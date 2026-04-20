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
from pydantic import BaseModel, HttpUrl

from mlx_audio.stt import load

MODEL_ID = os.environ.get("ASR_MODEL_ID", "mlx-community/Qwen3-ASR-1.7B-8bit")
MAX_DOWNLOAD_BYTES = int(os.environ.get("ASR_MAX_DOWNLOAD_BYTES", str(500 * 1024 * 1024)))
DOWNLOAD_TIMEOUT = float(os.environ.get("ASR_DOWNLOAD_TIMEOUT", "60"))
MAX_QUEUE = int(os.environ.get("ASR_MAX_QUEUE", "5"))
MAX_CONCURRENCY = int(os.environ.get("ASR_MAX_CONCURRENCY", "1"))

logger = logging.getLogger("asr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("HF_HOME=%s", os.environ.get("HF_HOME"))
    logger.info("loading mlx asr model: %s", MODEL_ID)
    _state["model"] = load(MODEL_ID)
    _state["semaphore"] = asyncio.Semaphore(MAX_CONCURRENCY)
    _state["counter_lock"] = asyncio.Lock()
    _state["in_flight"] = 0
    logger.info(
        "model loaded (max_queue=%d, max_concurrency=%d)", MAX_QUEUE, MAX_CONCURRENCY
    )
    try:
        yield
    finally:
        _state.clear()


app = FastAPI(title="MLX Qwen3-ASR API", lifespan=lifespan)


class AsrUrlReq(BaseModel):
    url: HttpUrl
    language: Optional[str] = None


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


async def _transcribe(path: str, language: Optional[str]) -> dict[str, Any]:
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    sem: asyncio.Semaphore = _state["semaphore"]
    kwargs: dict[str, Any] = {}
    if language:
        kwargs["language"] = language
    async with sem:
        try:
            result = await asyncio.to_thread(model.generate, path, **kwargs)
        except Exception as e:
            logger.exception("transcribe failed")
            raise HTTPException(status_code=500, detail=f"transcribe failed: {e}") from e
    return _result_to_dict(result)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "loaded": _state.get("model") is not None,
        "in_flight": _state.get("in_flight", 0),
        "max_queue": MAX_QUEUE,
        "max_concurrency": MAX_CONCURRENCY,
    }


@app.post("/asr_file")
async def asr_file(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None),
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
        return await _transcribe(tmp_path, language)
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
        return await _transcribe(tmp_path, req.language)
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
