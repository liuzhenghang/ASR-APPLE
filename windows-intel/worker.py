"""Windows + Intel (i7-12900H / Iris Xe iGPU) 子进程 worker.

使用 OpenVINO GenAI 的 WhisperPipeline 运行 whisper-large-v3-turbo (INT8, OV-IR 格式)。
默认模型: bweng/whisper-large-v3-turbo-int8-ov
默认设备: GPU (Intel iGPU)，可通过 ASR_DEVICE 切 CPU

与 app.py 的契约对齐（和 macos-arm/worker.py 一致）:
- worker_main(cfg, req_q, resp_q, ready_ev, aligner_ready_ev)
- op=asr: 输入 {path, language?, word_timestamps?}, 返回 {text, segments, language, total_time}
- op=align: 本平台不支持，直接 raise
"""
from __future__ import annotations

import os
import signal
import sys
import traceback
from typing import Any, Optional


_WHISPER_LANG_ALIAS = {
    "chinese": "zh", "mandarin": "zh", "zh-cn": "zh", "zh-tw": "zh", "zh_cn": "zh", "zh_tw": "zh",
    "cantonese": "yue", "yue": "yue",
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


def _to_whisper_lang_token(lang: Optional[str]) -> Optional[str]:
    """把 app.py 传来的 language（zh / chinese / en 等）转成 <|zh|> 这种 Whisper token。"""
    if not lang:
        return None
    s = str(lang).strip().lower()
    if not s:
        return None
    if s.startswith("<|") and s.endswith("|>"):
        return s
    s = _WHISPER_LANG_ALIAS.get(s, s)
    return f"<|{s}|>"


def _strip_lang_token(tok: Any) -> Optional[str]:
    if tok is None:
        return None
    s = str(tok).strip()
    if s.startswith("<|") and s.endswith("|>"):
        s = s[2:-2]
    return s.lower() or None


def _ensure_model_local(model_id: str, hf_home: Optional[str]) -> str:
    """下载模型到 hf_home/hub 下，返回本地目录路径。若已存在则直接返回。"""
    import huggingface_hub as hf_hub

    cache_root = os.path.join(hf_home or os.path.expanduser("~/.cache/huggingface"), "hub")
    os.makedirs(cache_root, exist_ok=True)
    local_dir = os.path.join(cache_root, "models--" + model_id.replace("/", "--"))

    sentinel = os.path.join(local_dir, "openvino_encoder_model.xml")
    if os.path.isfile(sentinel):
        print(
            f"[worker pid={os.getpid()}] model already cached at {local_dir}",
            flush=True,
        )
        return local_dir

    print(
        f"[worker pid={os.getpid()}] downloading {model_id} -> {local_dir}",
        flush=True,
    )
    hf_hub.snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    return local_dir


def _load_audio_16k(path: str):
    """读音频 -> 16kHz mono float32 numpy 一维数组。OpenVINO WhisperPipeline 要求这个输入。"""
    import librosa

    raw, _sr = librosa.load(path, sr=16000, mono=True)
    return raw


def _whisper_result_to_dict(result: Any, fallback_lang: Optional[str]) -> dict:
    """把 ov_genai WhisperDecodedResults 映射成 app.py 期望的 dict。"""
    texts = getattr(result, "texts", None)
    if texts:
        text = str(texts[0]).strip()
    else:
        text = str(result).strip()

    segs: list[dict] = []
    chunks = getattr(result, "chunks", None) or []
    for ch in chunks:
        t = getattr(ch, "text", None)
        s = getattr(ch, "start_ts", None)
        e = getattr(ch, "end_ts", None)
        if t is None:
            continue
        t_str = str(t).strip()
        if not t_str:
            continue
        try:
            s_f = float(s) if s is not None else None
            e_f = float(e) if e is not None else None
        except (TypeError, ValueError):
            s_f, e_f = None, None
        # OpenVINO 对未知时间戳有时返回 -1，这里归一化成 None
        if s_f is not None and s_f < 0:
            s_f = None
        if e_f is not None and e_f < 0:
            e_f = None
        segs.append({"text": t_str, "start": s_f, "end": e_f})

    lang = _strip_lang_token(getattr(result, "language", None)) or fallback_lang
    return {
        "text": text,
        "segments": segs,
        "language": lang,
        "total_time": None,
    }


def worker_main(cfg: dict, req_q, resp_q, ready_ev, aligner_ready_ev) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    hf_home = cfg.get("hf_home")
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")

    model_id = cfg["model_id"]
    device = (os.environ.get("ASR_DEVICE") or "GPU").upper()

    print(
        f"[worker pid={os.getpid()}] preparing whisper: id={model_id} device={device}",
        flush=True,
    )

    local_dir = _ensure_model_local(model_id, hf_home)

    # Windows: 显式把 openvino/libs 加进 DLL 搜索路径。
    # 正常情况下 openvino 包自己会做，但如果发生 ImportError: DLL load failed
    # 通常原因是缺 VC++ Redistributable，或者 pip 清华源把 openvino 的 dll 下崩了。
    if sys.platform == "win32":
        try:
            import openvino as _ov_probe

            _libs_dir = os.path.join(os.path.dirname(_ov_probe.__file__), "libs")
            if os.path.isdir(_libs_dir):
                os.add_dll_directory(_libs_dir)
                os.environ["PATH"] = _libs_dir + os.pathsep + os.environ.get("PATH", "")
        except ImportError as e:
            print(
                f"[worker pid={os.getpid()}] openvino import failed: {e}\n"
                f"  -> 大概率缺 Visual C++ Redistributable (x64): https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                f"  -> 或者 pip 清华源下载不完整，重装: pip install -U --no-cache-dir openvino openvino-tokenizers openvino-genai -i https://pypi.org/simple",
                flush=True,
            )
            raise

    print(f"[worker pid={os.getpid()}] importing openvino_genai...", flush=True)
    import openvino_genai as ov_genai

    # OpenVINO 编译缓存，避免每次启动 iGPU kernel 重新编译
    ov_cache_dir = os.path.join(hf_home or ".", "ov_cache")
    os.makedirs(ov_cache_dir, exist_ok=True)

    print(
        f"[worker pid={os.getpid()}] loading WhisperPipeline on {device} (cache={ov_cache_dir})...",
        flush=True,
    )
    try:
        pipe = ov_genai.WhisperPipeline(local_dir, device, CACHE_DIR=ov_cache_dir)
    except TypeError:
        # 极老版本不认 kwargs，退化成 dict 配置
        pipe = ov_genai.WhisperPipeline(local_dir, device, {"CACHE_DIR": ov_cache_dir})
    except Exception as e:
        if device != "CPU":
            print(
                f"[worker pid={os.getpid()}] device={device} load failed: {e}; "
                f"falling back to CPU",
                file=sys.stderr,
                flush=True,
            )
            device = "CPU"
            pipe = ov_genai.WhisperPipeline(local_dir, device, CACHE_DIR=ov_cache_dir)
        else:
            raise

    print(f"[worker pid={os.getpid()}] whisper ready on {device}", flush=True)

    # align 不支持；aligner_ready_ev 不 set
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

                audio = _load_audio_16k(path)

                gen_kwargs: dict[str, Any] = {
                    "task": "transcribe",
                    "return_timestamps": True,
                }
                lang_token = _to_whisper_lang_token(lang_in)
                if lang_token:
                    gen_kwargs["language"] = lang_token

                # OpenVINO GenAI 某些版本不认 word_timestamps；传了不支持就降级
                try:
                    if want_word_ts:
                        gen_kwargs["return_timestamps"] = True
                    result = pipe.generate(audio, **gen_kwargs)
                except TypeError as e:
                    msg = str(e)
                    dropped = None
                    for k in list(gen_kwargs.keys()):
                        if k in msg:
                            dropped = k
                            gen_kwargs.pop(k)
                            break
                    if dropped is None:
                        raise
                    print(
                        f"[worker pid={os.getpid()}] dropped unsupported kwarg '{dropped}': {e}",
                        flush=True,
                    )
                    result = pipe.generate(audio, **gen_kwargs)

                out = _whisper_result_to_dict(result, _strip_lang_token(lang_token))
            elif op == "align":
                raise RuntimeError(
                    "align not supported on windows-intel (set ASR_ENABLE_ALIGN=0)"
                )
            else:
                raise ValueError(f"unknown op: {op}")
            resp_q.put({"id": tid, "ok": True, "result": out})
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            # 老版 optimum-intel 导的模型没 beam_idx 端口，命中就在错误里明说
            if "beam_idx" in str(e):
                err_msg += (
                    "\n>> 模型 IR 和 openvino-genai 版本不匹配 (老模型缺 stateful 端口 beam_idx)。\n"
                    ">> 换成 FluidInference/whisper-large-v3-turbo-int8-ov-npu 或用 "
                    "optimum-cli export openvino --model openai/whisper-large-v3-turbo --weight-format int8 ./out 重新导出。"
                )
            resp_q.put(
                {
                    "id": tid,
                    "ok": False,
                    "error": err_msg,
                    "tb": traceback.format_exc(),
                }
            )
