import asyncio
import logging
import multiprocessing as mp
import os
import queue as _queue
import tempfile
import uuid
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

from worker import worker_main

MODEL_ID = os.environ.get("ASR_MODEL_ID", "mlx-community/Qwen3-ASR-1.7B-8bit")
ALIGNER_ID = os.environ.get(
    "ASR_ALIGNER_ID", "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
)
ENABLE_ALIGN = os.environ.get("ASR_ENABLE_ALIGN", "1") not in ("0", "false", "False", "")
SEG_GAP_SEC = float(os.environ.get("ASR_SEG_GAP_SEC", "0.8"))
SEG_MAX_DURATION = float(os.environ.get("ASR_SEG_MAX_DURATION", "30"))
SEG_MAX_CHARS = int(os.environ.get("ASR_SEG_MAX_CHARS", "120"))
ALIGN_MAX_CHARS = int(os.environ.get("ASR_ALIGN_MAX_CHARS", "3000"))
MAX_DOWNLOAD_BYTES = int(os.environ.get("ASR_MAX_DOWNLOAD_BYTES", str(500 * 1024 * 1024)))
DOWNLOAD_TIMEOUT = float(os.environ.get("ASR_DOWNLOAD_TIMEOUT", "60"))
MAX_QUEUE = int(os.environ.get("ASR_MAX_QUEUE", "5"))
MAX_CONCURRENCY = int(os.environ.get("ASR_MAX_CONCURRENCY", "1"))
ASR_TIMEOUT = float(os.environ.get("ASR_TIMEOUT", "180"))
ALIGN_TIMEOUT = float(os.environ.get("ASR_ALIGN_TIMEOUT", "60"))
WORKER_READY_TIMEOUT = float(os.environ.get("ASR_WORKER_READY_TIMEOUT", "600"))

_CJK_LANGS = {"chinese", "cantonese", "japanese", "korean"}

# ForcedAligner 已知支持语言（不在列表里就不调 aligner，直接走标点切分 fallback）
_ALIGNER_SUPPORTED_LANGS = {
    "cantonese", "chinese", "english", "french", "german", "italian",
    "japanese", "korean", "portuguese", "russian", "spanish",
}

# 多语言句末标点（中/英/日/阿拉伯/西语等）
_SENT_END_PUNCT = set("。！？；.!?;…؟؛।")
# 次级分隔（句子太长时再按这些切）
_SUB_SEP_PUNCT = set("，、,،")

logger = logging.getLogger("asr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_state: dict[str, Any] = {}


class WorkerBusy(Exception):
    pass


class WorkerManager:
    """常驻子进程 + 串行 RPC。超时时 terminate 子进程并自动重启。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ctx = mp.get_context("spawn")
        self.proc: Optional[mp.process.BaseProcess] = None
        self.req_q = None
        self.resp_q = None
        self.aligner_ready = False
        self.lock = asyncio.Lock()
        self.restart_count = 0

    def _start_sync(self) -> None:
        self.req_q = self.ctx.Queue()
        self.resp_q = self.ctx.Queue()
        ready = self.ctx.Event()
        aligner_ready = self.ctx.Event()
        self.proc = self.ctx.Process(
            target=worker_main,
            args=(self.cfg, self.req_q, self.resp_q, ready, aligner_ready),
            daemon=True,
        )
        self.proc.start()
        logger.info(
            "worker: spawned pid=%d, waiting model load (<= %.0fs)...",
            self.proc.pid,
            WORKER_READY_TIMEOUT,
        )
        if not ready.wait(timeout=WORKER_READY_TIMEOUT):
            self._kill_sync()
            raise RuntimeError(
                f"worker did not become ready in {WORKER_READY_TIMEOUT:.0f}s"
            )
        self.aligner_ready = aligner_ready.is_set()
        logger.info(
            "worker: ready (pid=%d, aligner=%s)",
            self.proc.pid,
            self.aligner_ready,
        )

    def _kill_sync(self) -> None:
        proc = self.proc
        old_resp_q = self.resp_q
        self.proc = None
        self.req_q = None
        self.resp_q = None
        if proc is not None:
            try:
                if proc.is_alive():
                    logger.warning("worker: terminating pid=%d", proc.pid)
                    proc.terminate()
                    proc.join(timeout=5)
                if proc.is_alive():
                    logger.warning("worker: kill -9 pid=%d", proc.pid)
                    proc.kill()
                    proc.join(timeout=5)
            except Exception:
                logger.exception("worker: kill error")
        # 唤醒可能卡在 resp_q.get() 的后台线程，避免线程泄漏
        if old_resp_q is not None:
            try:
                old_resp_q.put_nowait(
                    {"id": "__killed__", "ok": False, "error": "worker killed"}
                )
            except Exception:
                pass

    async def start(self) -> None:
        await asyncio.to_thread(self._start_sync)

    async def stop(self) -> None:
        await asyncio.to_thread(self._kill_sync)

    async def _ensure_alive(self) -> None:
        if self.proc is None or not self.proc.is_alive():
            logger.warning("worker: not alive, restarting...")
            self.restart_count += 1
            await asyncio.to_thread(self._start_sync)

    async def call(self, op: str, args: dict, timeout: float) -> Any:
        async with self.lock:
            await self._ensure_alive()
            tid = uuid.uuid4().hex
            req_q = self.req_q
            resp_q = self.resp_q
            # 清空可能遗留的响应（正常情况下应该没有，保险起见）
            try:
                while True:
                    resp_q.get_nowait()
            except _queue.Empty:
                pass
            await asyncio.to_thread(req_q.put, {"id": tid, "op": op, "args": args})
            try:
                resp = await asyncio.wait_for(
                    asyncio.to_thread(resp_q.get), timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.error(
                    "worker: timeout on op=%s after %.0fs, killing worker",
                    op,
                    timeout,
                )
                await asyncio.to_thread(self._kill_sync)
                raise
            if resp.get("id") == "__killed__" or not resp.get("ok"):
                err = resp.get("error") or "worker error"
                raise RuntimeError(err)
            return resp["result"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("HF_HOME=%s", os.environ.get("HF_HOME"))
    cfg = {
        "model_id": MODEL_ID,
        "aligner_id": ALIGNER_ID,
        "enable_align": ENABLE_ALIGN,
        "hf_home": os.environ.get("HF_HOME"),
    }
    wm = WorkerManager(cfg)
    await wm.start()
    _state["worker"] = wm
    _state["semaphore"] = asyncio.Semaphore(MAX_CONCURRENCY)
    _state["counter_lock"] = asyncio.Lock()
    _state["in_flight"] = 0
    logger.info(
        "service ready (max_queue=%d, max_concurrency=%d, align=%s)",
        MAX_QUEUE,
        MAX_CONCURRENCY,
        wm.aligner_ready,
    )
    try:
        yield
    finally:
        try:
            await wm.stop()
        except Exception:
            logger.exception("worker stop error")
        _state.clear()


app = FastAPI(title="MLX Qwen3-ASR API", lifespan=lifespan)


class AsrUrlReq(BaseModel):
    url: HttpUrl
    language: Optional[str] = None
    more_info: Optional[bool] = False


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


def _coerce_lang(lang: Any) -> Optional[str]:
    """ASR/aligner 返回的 language 可能是 str / list / tuple，统一成单个 str。"""
    if lang is None:
        return None
    if isinstance(lang, (list, tuple)):
        lang = lang[0] if lang else None
    if lang is None:
        return None
    if not isinstance(lang, str):
        lang = str(lang)
    lang = lang.strip()
    return lang or None


def _normalize_language(lang: Any) -> Optional[str]:
    s = _coerce_lang(lang)
    return s.capitalize() if s else None


def _is_cjk_language(lang: Any) -> bool:
    s = _coerce_lang(lang)
    if not s:
        return False
    return s.lower() in _CJK_LANGS


def _aligner_supports(language: Any) -> bool:
    s = _coerce_lang(language)
    if not s:
        return True
    return s.lower() in _ALIGNER_SUPPORTED_LANGS


def _split_text_by_punct(text: str, language: Optional[str]) -> list[str]:
    """无时间戳的兜底切分：先按句末标点切，长句再按逗号切，还长就按长度硬切。"""
    text = (text or "").strip()
    if not text:
        return []

    def flush_chunk(buf: list[str], out: list[str]):
        s = "".join(buf).strip()
        if s:
            out.append(s)

    primary: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in _SENT_END_PUNCT:
            flush_chunk(buf, primary)
            buf = []
    flush_chunk(buf, primary)

    cjk = _is_cjk_language(language)
    joiner = "" if cjk else " "

    def too_long(s: str) -> bool:
        return len(s) > SEG_MAX_CHARS

    def hard_chunks(s: str) -> list[str]:
        if cjk:
            return [s[i : i + SEG_MAX_CHARS] for i in range(0, len(s), SEG_MAX_CHARS)]
        words = s.split()
        chunks: list[str] = []
        cur: list[str] = []
        cur_len = 0
        for w in words:
            add = len(w) + (1 if cur else 0)
            if cur and cur_len + add > SEG_MAX_CHARS:
                chunks.append(joiner.join(cur))
                cur = [w]
                cur_len = len(w)
            else:
                cur.append(w)
                cur_len += add
        if cur:
            chunks.append(joiner.join(cur))
        return chunks

    result: list[str] = []
    for sent in primary:
        if not too_long(sent):
            result.append(sent)
            continue
        sub_buf: list[str] = []
        subs: list[str] = []
        for ch in sent:
            sub_buf.append(ch)
            if ch in _SUB_SEP_PUNCT:
                flush_chunk(sub_buf, subs)
                sub_buf = []
        flush_chunk(sub_buf, subs)
        for s in subs:
            if too_long(s):
                result.extend(hard_chunks(s))
            else:
                result.append(s)
    return [s for s in (x.strip() for x in result) if s]


def _items_to_segments(
    items: list[dict[str, Any]], language: Optional[str]
) -> list[dict[str, Any]]:
    if not items:
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

    for item in items:
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


async def _run_align(
    worker: "WorkerManager",
    audio_path: str,
    text: str,
    language: Optional[str],
) -> list[dict[str, Any]]:
    if not worker.aligner_ready:
        logger.info("align: skip (aligner not loaded)")
        return []
    if not text or not text.strip():
        logger.info("align: skip (empty text)")
        return []
    if len(text) > ALIGN_MAX_CHARS:
        logger.info(
            "align: skip (text_len=%d > ASR_ALIGN_MAX_CHARS=%d)",
            len(text),
            ALIGN_MAX_CHARS,
        )
        return []
    if not _aligner_supports(language):
        logger.info(
            "align: skip (language %r not in aligner-supported %s)",
            language,
            sorted(_ALIGNER_SUPPORTED_LANGS),
        )
        return []
    norm_lang = _normalize_language(language)
    args: dict[str, Any] = {"audio": audio_path, "text": text}
    if norm_lang:
        args["language"] = norm_lang
    logger.info("align: run aligner (lang=%s, text_len=%d)", norm_lang, len(text))
    try:
        items = await worker.call("align", args, timeout=ALIGN_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error(
            "align: timeout after %.0fs, worker killed", ALIGN_TIMEOUT
        )
        return []
    except Exception as e:
        logger.error("align: failed (%s)", e)
        return []
    logger.info("align: got %d aligned items", len(items or []))
    return _items_to_segments(items or [], language)


async def _transcribe(path: str, language: Optional[str]) -> dict[str, Any]:
    worker: Optional[WorkerManager] = _state.get("worker")
    if worker is None:
        raise HTTPException(status_code=503, detail="worker not ready")
    sem: asyncio.Semaphore = _state["semaphore"]
    args: dict[str, Any] = {"path": path}
    norm_lang = _normalize_language(language)
    if norm_lang:
        args["language"] = norm_lang
    async with sem:
        logger.info(
            "asr: start (path=%s, lang=%s, timeout=%.0fs)",
            os.path.basename(path),
            norm_lang,
            ASR_TIMEOUT,
        )
        try:
            out = await worker.call("asr", args, timeout=ASR_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(
                "asr: timeout after %.0fs (path=%s, lang=%s), worker restarted",
                ASR_TIMEOUT,
                os.path.basename(path),
                norm_lang,
            )
            raise HTTPException(
                status_code=504,
                detail=f"asr timeout after {ASR_TIMEOUT:.0f}s, worker killed and restarted",
            )
        except Exception as e:
            logger.exception("asr: worker call failed")
            raise HTTPException(status_code=500, detail=f"transcribe failed: {e}") from e

        out["language"] = _coerce_lang(out.get("language"))
        detected_lang = out.get("language") or norm_lang
        raw_text = (out.get("text") or "").strip()
        raw_segments = out.get("segments") or []
        logger.info(
            "asr: done (lang=%s, text_len=%d, raw_segments=%d, preview=%r)",
            detected_lang,
            len(raw_text),
            len(raw_segments),
            raw_text[:80],
        )

        segs: list[dict[str, Any]] = []

        if worker.aligner_ready and raw_text:
            segs = await _run_align(worker, path, raw_text, detected_lang)

        if not segs and len(raw_segments) > 1:
            logger.info("seg: use native asr segments (%d)", len(raw_segments))
            segs = [
                {
                    "text": (s.get("text") or "").strip(),
                    "start": s.get("start"),
                    "end": s.get("end"),
                }
                for s in raw_segments
                if (s.get("text") or "").strip()
            ]

        if not segs and raw_text:
            punct_segs = _split_text_by_punct(raw_text, detected_lang)
            if len(punct_segs) > 1:
                logger.info(
                    "seg: fallback to punctuation split (%d segments)",
                    len(punct_segs),
                )
                segs = [{"text": s, "start": None, "end": None} for s in punct_segs]
            else:
                logger.info(
                    "seg: punctuation split produced %d segments, keep raw text",
                    len(punct_segs),
                )

        if segs:
            out["segments"] = segs
            out["text"] = "\n".join(s["text"] for s in segs)
            logger.info("seg: final %d segments", len(segs))
    return out


@app.get("/health")
async def health():
    worker: Optional[WorkerManager] = _state.get("worker")
    worker_alive = bool(worker and worker.proc and worker.proc.is_alive())
    return {
        "status": "ok" if worker_alive else "worker_down",
        "model": MODEL_ID,
        "aligner": ALIGNER_ID if (worker and worker.aligner_ready) else None,
        "worker_alive": worker_alive,
        "worker_pid": worker.proc.pid if worker_alive else None,
        "worker_restarts": worker.restart_count if worker else 0,
        "in_flight": _state.get("in_flight", 0),
        "max_queue": MAX_QUEUE,
        "max_concurrency": MAX_CONCURRENCY,
        "seg_gap_sec": SEG_GAP_SEC,
        "seg_max_duration": SEG_MAX_DURATION,
        "seg_max_chars": SEG_MAX_CHARS,
        "align_max_chars": ALIGN_MAX_CHARS,
        "asr_timeout": ASR_TIMEOUT,
        "align_timeout": ALIGN_TIMEOUT,
    }


@app.post("/asr_file")
async def asr_file(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    more_info: Optional[bool] = Form(False),
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
        if more_info:
            return result
        return Response(content=result["text"], media_type="text/plain")
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
        if req.more_info:
            return result
        return Response(content=result["text"], media_type="text/plain")
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
