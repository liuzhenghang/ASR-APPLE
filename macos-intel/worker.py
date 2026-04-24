"""Intel Mac (x86_64) 子进程 worker.

引擎: faster-whisper (CTranslate2) CPU int8。
Intel Mac 上的 AMD 显卡通过 MPS/Metal 跑 Whisper 并不稳定（PyTorch 对 Intel Mac 的 AMD 卡
早就停更，whisper.cpp 的 Metal backend 编译链路又折腾），所以这里默认走 CPU int8，
Intel 10 代 及以上 AVX2 跑 large-v3-turbo 足够实用。

默认模型: deepdml/faster-whisper-large-v3-turbo-ct2

与 app.py 的契约对齐:
- worker_main(cfg, req_q, resp_q, ready_ev, aligner_ready_ev)
- op=asr: 输入 {path, language?, word_timestamps?}, 返回 {text, segments, language, total_time}
- op=align: 本平台不支持，直接 raise
"""
from __future__ import annotations

import os
import signal
import traceback
from typing import Any, Optional

_LANG_ALIAS = {
    "chinese": "zh", "mandarin": "zh", "zh-cn": "zh", "zh-tw": "zh", "zh_cn": "zh", "zh_tw": "zh",
    "cantonese": "yue",
    "english": "en",
    "japanese": "ja",
    "korean": "ko",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "spanish": "es",
    "portuguese": "pt",
    "russian": "ru",
    "arabic": "ar",
    "hindi": "hi",
    "thai": "th",
    "vietnamese": "vi",
    "indonesian": "id",
    "turkish": "tr",
    "dutch": "nl",
    "polish": "pl",
    "czech": "cs",
    "ukrainian": "uk",
    "swedish": "sv",
    "norwegian": "no",
    "danish": "da",
    "finnish": "fi",
    "hebrew": "he",
    "hungarian": "hu",
    "romanian": "ro",
    "greek": "el",
    "bulgarian": "bg",
    "malay": "ms",
    "tamil": "ta",
    "bengali": "bn",
    "urdu": "ur",
    "persian": "fa",
}


def _to_iso_lang(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    s = str(lang).strip().lower()
    if not s:
        return None
    return _LANG_ALIAS.get(s, s)


def worker_main(cfg: dict, req_q, resp_q, ready_ev, aligner_ready_ev) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    hf_home = cfg.get("hf_home")
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")

    model_id = cfg["model_id"]
    device = (os.environ.get("ASR_DEVICE") or "cpu").lower()
    compute_type = os.environ.get("ASR_COMPUTE_TYPE") or ("int8" if device == "cpu" else "int8_float16")
    cpu_threads = int(os.environ.get("ASR_CPU_THREADS") or "0")
    num_workers = int(os.environ.get("ASR_NUM_WORKERS") or "1")

    print(
        f"[worker pid={os.getpid()}] loading faster-whisper: id={model_id} "
        f"device={device} compute_type={compute_type} threads={cpu_threads}",
        flush=True,
    )

    from faster_whisper import WhisperModel

    download_root = os.path.join(hf_home or os.path.expanduser("~/.cache/huggingface"), "hub")
    os.makedirs(download_root, exist_ok=True)

    model = WhisperModel(
        model_id,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        download_root=download_root,
    )
    print(f"[worker pid={os.getpid()}] whisper ready", flush=True)

    # align 不支持，aligner_ready_ev 不 set
    ready_ev.set()

    while True:
        try:
            task = req_q.get()
        except (EOFError, KeyboardInterrupt):
            break
        if task is None:
            break
        tid = task.get("id")
        op = task.get("op")
        args = dict(task.get("args") or {})
        try:
            if op == "asr":
                path = args.pop("path")
                lang_in = args.pop("language", None)
                want_word_ts = bool(args.pop("word_timestamps", False))

                iso = _to_iso_lang(lang_in)

                segments_iter, info = model.transcribe(
                    path,
                    language=iso,
                    task="transcribe",
                    word_timestamps=want_word_ts,
                    vad_filter=False,
                    beam_size=int(os.environ.get("ASR_BEAM_SIZE") or "5"),
                )

                segs: list[dict] = []
                full_texts: list[str] = []
                for seg in segments_iter:
                    text = (seg.text or "").strip()
                    if not text:
                        continue
                    try:
                        s = float(seg.start) if seg.start is not None else None
                        e = float(seg.end) if seg.end is not None else None
                    except (TypeError, ValueError):
                        s, e = None, None
                    segs.append({"text": text, "start": s, "end": e})
                    full_texts.append(text)

                detected = getattr(info, "language", None) or iso
                out = {
                    "text": " ".join(full_texts).strip(),
                    "segments": segs,
                    "language": detected,
                    "total_time": getattr(info, "duration", None),
                }
            elif op == "align":
                raise RuntimeError(
                    "align not supported on macos-intel (set ASR_ENABLE_ALIGN=0)"
                )
            else:
                raise ValueError(f"unknown op: {op}")
            resp_q.put({"id": tid, "ok": True, "result": out})
        except Exception as e:
            resp_q.put(
                {
                    "id": tid,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "tb": traceback.format_exc(),
                }
            )
