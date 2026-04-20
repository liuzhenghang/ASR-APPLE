import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(DEFAULT_MODELS_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_MODELS_DIR / "hub"))

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, HttpUrl

from mlx_audio.stt import load

MODEL_ID = os.environ.get("ASR_MODEL_ID", "mlx-community/Qwen3-ASR-1.7B-8bit")
ALIGNER_ID = os.environ.get(
    "ASR_ALIGNER_ID", "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
)
ENABLE_ALIGN = os.environ.get("ASR_ENABLE_ALIGN", "1") not in ("0", "false", "False", "")
SEG_GAP_SEC = float(os.environ.get("ASR_SEG_GAP_SEC", "0.8"))
SEG_MAX_DURATION = float(os.environ.get("ASR_SEG_MAX_DURATION", "30"))
SEG_MAX_CHARS = int(os.environ.get("ASR_SEG_MAX_CHARS", "120"))
MAX_DOWNLOAD_BYTES = int(os.environ.get("ASR_MAX_DOWNLOAD_BYTES", str(500 * 1024 * 1024)))
DOWNLOAD_TIMEOUT = float(os.environ.get("ASR_DOWNLOAD_TIMEOUT", "60"))
MAX_QUEUE = int(os.environ.get("ASR_MAX_QUEUE", "5"))
MAX_CONCURRENCY = int(os.environ.get("ASR_MAX_CONCURRENCY", "1"))

_CJK_LANGS = {"chinese", "cantonese", "japanese", "korean"}

logger = logging.getLogger("asr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("HF_HOME=%s", os.environ.get("HF_HOME"))
    logger.info("loading mlx asr model: %s", MODEL_ID)
    _state["model"] = load(MODEL_ID)
    _state["aligner"] = None
    if ENABLE_ALIGN:
        try:
            logger.info("loading mlx forced-aligner: %s", ALIGNER_ID)
            _state["aligner"] = load(ALIGNER_ID)
            logger.info("aligner loaded")
        except Exception as e:
            logger.exception("load aligner failed, will skip post-alignment: %s", e)
            _state["aligner"] = None
    _state["semaphore"] = asyncio.Semaphore(MAX_CONCURRENCY)
    _state["counter_lock"] = asyncio.Lock()
    _state["in_flight"] = 0
    logger.info(
        "model loaded (max_queue=%d, max_concurrency=%d, align=%s)",
        MAX_QUEUE,
        MAX_CONCURRENCY,
        bool(_state["aligner"]),
    )
    try:
        yield
    finally:
        _state.clear()


app = FastAPI(title="MLX Qwen3-ASR API", lifespan=lifespan)


class AsrUrlReq(BaseModel):
    url: HttpUrl
    language: Optional[str] = None
    only_text: Optional[bool] = False


async def _acquire_slot():
    lock: asyncio.Lock = _state["counter_lock"]
    async with lock:
        if _state["in_flight"] >= MAX_QUEUE:
            raise HTTPException(
                status_code=429,
                detail=f"queue full ({MAX_QUEUE}), please retry later",
            )
        _state["in_flight"] += 1


async def _release_slot():
    lock: asyncio.Lock = _state["counter_lock"]
    async with lock:
        _state["in_flight"] = max(0, _state["in_flight"] - 1)


def _suffix_from_name(name: Optional[str], fallback: str = ".wav") -> str:
    if not name:
        return fallback
    _, ext = os.path.splitext(name)
    return ext if ext else fallback


def _suffix_from_url(url: str, content_type: Optional[str]) -> str:
    path = urlparse(url).path
    _, ext = os.path.splitext(path)
    if ext:
        return ext
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        mapping = {
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/mp4": ".m4a",
            "audio/x-m4a": ".m4a",
            "audio/aac": ".aac",
            "audio/ogg": ".ogg",
            "audio/flac": ".flac",
            "audio/x-flac": ".flac",
            "audio/webm": ".webm",
        }
        if ct in mapping:
            return mapping[ct]
    return ".wav"


def _result_to_dict(result: Any) -> dict[str, Any]:
    text = getattr(result, "text", None)
    segments = getattr(result, "segments", None) or []
    language = getattr(result, "language", None)
    total_time = getattr(result, "total_time", None)

    norm_segments = []
    for seg in segments:
        if isinstance(seg, dict):
            norm_segments.append(
                {
                    "text": seg.get("text"),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                }
            )
        else:
            norm_segments.append(
                {
                    "text": getattr(seg, "text", None),
                    "start": getattr(seg, "start", None),
                    "end": getattr(seg, "end", None),
                }
            )

    return {
        "text": text,
        "segments": norm_segments,
        "language": language,
        "total_time": total_time,
    }


def _normalize_language(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    return lang.strip().capitalize()


def _is_cjk_language(lang: Optional[str]) -> bool:
    if not lang:
        return False
    return lang.strip().lower() in _CJK_LANGS


def _align_to_segments(
    audio_path: str,
    text: str,
    language: Optional[str],
) -> list[dict[str, Any]]:
    """用 ForcedAligner 拿到 word/char 级时间戳，按停顿切段。失败返回 []。"""
    aligner = _state.get("aligner")
    if aligner is None or not text or not text.strip():
        return []
    try:
        norm_lang = _normalize_language(language)
        kwargs: dict[str, Any] = {"audio": audio_path, "text": text}
        if norm_lang:
            kwargs["language"] = norm_lang
        items = aligner.generate(**kwargs)
    except Exception:
        logger.exception("forced-align failed, fallback to raw text")
        return []

    norm: list[dict[str, Any]] = []
    for it in items or []:
        t = getattr(it, "text", None) if not isinstance(it, dict) else it.get("text")
        s = getattr(it, "start_time", None) if not isinstance(it, dict) else it.get("start_time")
        e = getattr(it, "end_time", None) if not isinstance(it, dict) else it.get("end_time")
        if t is None or s is None or e is None:
            continue
        norm.append({"text": str(t), "start": float(s), "end": float(e)})

    if not norm:
        return []

    cjk = _is_cjk_language(language)
    joiner = "" if cjk else " "

    segments: list[dict[str, Any]] = []
    cur_words: list[str] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    prev_end: Optional[float] = None

    def flush():
        if not cur_words:
            return
        segments.append(
            {
                "text": joiner.join(cur_words).strip(),
                "start": cur_start,
                "end": cur_end,
            }
        )

    for item in norm:
        w, s, e = item["text"], item["start"], item["end"]
        gap = (s - prev_end) if prev_end is not None else 0.0
        cur_chars = len(joiner.join(cur_words))
        cur_dur = (cur_end - cur_start) if cur_start is not None and cur_end is not None else 0.0
        should_split = cur_words and (
            gap >= SEG_GAP_SEC
            or cur_dur >= SEG_MAX_DURATION
            or cur_chars >= SEG_MAX_CHARS
        )
        if should_split:
            flush()
            cur_words = []
            cur_start = None
            cur_end = None
        if cur_start is None:
            cur_start = s
        cur_end = e
        cur_words.append(w)
        prev_end = e
    flush()
    return [seg for seg in segments if seg["text"]]


async def _transcribe(path: str, language: Optional[str]) -> dict[str, Any]:
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    sem: asyncio.Semaphore = _state["semaphore"]
    kwargs: dict[str, Any] = {}
    norm_lang = _normalize_language(language)
    if norm_lang:
        kwargs["language"] = norm_lang
    async with sem:
        try:
            result = await asyncio.to_thread(model.generate, path, **kwargs)
        except Exception as e:
            logger.exception("transcribe failed")
            raise HTTPException(status_code=500, detail=f"transcribe failed: {e}") from e

        out = _result_to_dict(result)
        detected_lang = out.get("language") or norm_lang
        raw_text = (out.get("text") or "").strip()

        if _state.get("aligner") is not None and raw_text:
            try:
                segs = await asyncio.to_thread(
                    _align_to_segments, path, raw_text, detected_lang
                )
            except Exception:
                logger.exception("align task failed")
                segs = []
            if segs:
                out["segments"] = segs
                out["text"] = "\n".join(s["text"] for s in segs)
    return out


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "loaded": _state.get("model") is not None,
        "aligner": ALIGNER_ID if _state.get("aligner") is not None else None,
        "in_flight": _state.get("in_flight", 0),
        "max_queue": MAX_QUEUE,
        "max_concurrency": MAX_CONCURRENCY,
        "seg_gap_sec": SEG_GAP_SEC,
        "seg_max_duration": SEG_MAX_DURATION,
        "seg_max_chars": SEG_MAX_CHARS,
    }


@app.post("/asr_file")
async def asr_file(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    only_text: Optional[bool] = Form(False),
):
    await _acquire_slot()
    suffix = _suffix_from_name(audio_file.filename)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    try:
        total = 0
        while True:
            chunk = await audio_file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                raise HTTPException(status_code=413, detail="file too large")
            tmp.write(chunk)
        tmp.close()
        if total == 0:
            raise HTTPException(status_code=400, detail="empty audio_file")
        result = await _transcribe(tmp_path, language)
        if only_text:
            return Response(content=result["text"], media_type="text/plain")
        return result
    finally:
        try:
            if not tmp.closed:
                tmp.close()
        except Exception:
            pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        await _release_slot()


@app.post("/asr_url")
async def asr_url(req: AsrUrlReq):
    await _acquire_slot()
    url = str(req.url)
    tmp_path: Optional[str] = None
    try:
        try:
            async with httpx.AsyncClient(
                timeout=DOWNLOAD_TIMEOUT, follow_redirects=True
            ) as client:
                async with client.stream("GET", url) as resp:
                    if resp.status_code >= 400:
                        raise HTTPException(
                            status_code=400,
                            detail=f"download failed: http {resp.status_code}",
                        )
                    content_type = resp.headers.get("content-type")
                    suffix = _suffix_from_url(url, content_type)
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp_path = tmp.name
                    total = 0
                    try:
                        async for chunk in resp.aiter_bytes(1024 * 1024):
                            total += len(chunk)
                            if total > MAX_DOWNLOAD_BYTES:
                                raise HTTPException(
                                    status_code=413, detail="remote file too large"
                                )
                            tmp.write(chunk)
                    finally:
                        tmp.close()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=400, detail=f"download error: {e}") from e

        if total == 0:
            raise HTTPException(status_code=400, detail="empty remote file")
        result = await _transcribe(tmp_path, req.language)
        if req.only_text:
            return Response(content=result["text"], media_type="text/plain")
        return result
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        await _release_slot()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.environ.get("ASR_HOST", "0.0.0.0"),
        port=int(os.environ.get("ASR_PORT", "8000")),
        reload=False,
    )
