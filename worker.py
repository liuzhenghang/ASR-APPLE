"""独立子进程，负责真正加载 mlx 模型并执行推理。

主进程通过 multiprocessing Queue 与它通信，超时可直接 terminate 本进程，
强制释放 GPU/内存，主进程再起一个新的即可。
"""
from __future__ import annotations

import os
import signal
import sys
import traceback
from typing import Any


def _stt_output_to_dict(r: Any) -> dict:
    text = getattr(r, "text", None)
    if isinstance(text, (list, tuple)):
        text = text[0] if text else None
    segs_raw = getattr(r, "segments", None) or []
    segs = []
    for s in segs_raw:
        if isinstance(s, dict):
            segs.append(
                {
                    "text": s.get("text"),
                    "start": s.get("start"),
                    "end": s.get("end"),
                }
            )
        else:
            segs.append(
                {
                    "text": getattr(s, "text", None),
                    "start": getattr(s, "start", None),
                    "end": getattr(s, "end", None),
                }
            )
    lang = getattr(r, "language", None)
    if isinstance(lang, (list, tuple)):
        lang = lang[0] if lang else None
    return {
        "text": text,
        "segments": segs,
        "language": lang,
        "total_time": getattr(r, "total_time", None),
    }


def _align_result_to_list(r: Any) -> list:
    out = []
    for it in r or []:
        if isinstance(it, dict):
            t = it.get("text")
            s = it.get("start_time")
            e = it.get("end_time")
        else:
            t = getattr(it, "text", None)
            s = getattr(it, "start_time", None)
            e = getattr(it, "end_time", None)
        if t is None or s is None or e is None:
            continue
        out.append({"text": str(t), "start": float(s), "end": float(e)})
    return out


def _safe_call(fn, *args, **kwargs):
    """带 kwargs 兼容降级：如果某个 kwarg 不支持，丢弃它再试一次。"""
    try:
        return fn(*args, **kwargs)
    except TypeError as e:
        msg = str(e)
        dropped = None
        for k in list(kwargs.keys()):
            if k in msg:
                dropped = k
                kwargs.pop(k)
                break
        if dropped is None:
            raise
        print(
            f"[worker pid={os.getpid()}] dropped unsupported kwarg '{dropped}': {e}",
            flush=True,
        )
        return _safe_call(fn, *args, **kwargs)


def worker_main(cfg: dict, req_q, resp_q, ready_ev, aligner_ready_ev) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    hf_home = cfg.get("hf_home")
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")

    print(f"[worker pid={os.getpid()}] loading asr: {cfg['model_id']}", flush=True)
    from mlx_audio.stt import load

    asr = load(cfg["model_id"])
    print(f"[worker pid={os.getpid()}] asr loaded", flush=True)

    aligner = None
    if cfg.get("enable_align"):
        try:
            print(
                f"[worker pid={os.getpid()}] loading aligner: {cfg['aligner_id']}",
                flush=True,
            )
            aligner = load(cfg["aligner_id"])
            aligner_ready_ev.set()
            print(f"[worker pid={os.getpid()}] aligner loaded", flush=True)
        except Exception as e:
            print(
                f"[worker pid={os.getpid()}] aligner load failed: {e}",
                file=sys.stderr,
                flush=True,
            )

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
                r = _safe_call(asr.generate, path, **args)
                out = _stt_output_to_dict(r)
            elif op == "align":
                if aligner is None:
                    raise RuntimeError("aligner not available in worker")
                r = _safe_call(aligner.generate, **args)
                out = _align_result_to_list(r)
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
