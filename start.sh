#!/usr/bin/env bash
set -euo pipefail

# ---------- 可调参数 ----------
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ASR_HOST="${ASR_HOST:-0.0.0.0}"
ASR_PORT="${ASR_PORT:-8000}"
ASR_MODEL_ID="${ASR_MODEL_ID:-mlx-community/Qwen3-ASR-1.7B-8bit}"
ASR_ALIGNER_ID="${ASR_ALIGNER_ID:-mlx-community/Qwen3-ForcedAligner-0.6B-8bit}"
ASR_ENABLE_ALIGN="${ASR_ENABLE_ALIGN:-1}"
ASR_SEG_GAP_SEC="${ASR_SEG_GAP_SEC:-0.8}"
ASR_SEG_MAX_DURATION="${ASR_SEG_MAX_DURATION:-30}"
ASR_SEG_MAX_CHARS="${ASR_SEG_MAX_CHARS:-120}"
ASR_ALIGN_MAX_CHARS="${ASR_ALIGN_MAX_CHARS:-3000}"
ASR_MAX_QUEUE="${ASR_MAX_QUEUE:-5}"
ASR_MAX_CONCURRENCY="${ASR_MAX_CONCURRENCY:-1}"
ASR_TIMEOUT="${ASR_TIMEOUT:-180}"
ASR_ALIGN_TIMEOUT="${ASR_ALIGN_TIMEOUT:-60}"
ASR_WORKER_READY_TIMEOUT="${ASR_WORKER_READY_TIMEOUT:-600}"
ASR_MAX_NEW_TOKENS="${ASR_MAX_NEW_TOKENS:-2048}"
# 可选: 走镜像加速 HF 下载
# export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---------- 切到脚本所在目录 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- 模型缓存目录：项目内 models/ ----------
MODELS_DIR="$SCRIPT_DIR/models"
mkdir -p "$MODELS_DIR"
export HF_HOME="$MODELS_DIR"
export HUGGINGFACE_HUB_CACHE="$MODELS_DIR/hub"

# ---------- 检测/创建虚拟环境 ----------
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "[start.sh] 未检测到虚拟环境，使用 $PYTHON_BIN 创建: $VENV_DIR"
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[start.sh] 找不到 $PYTHON_BIN，请先安装 Python3" >&2
    exit 1
  fi
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ---------- 处理参数 ----------
FORCE_INSTALL=0
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    install) FORCE_INSTALL=1 ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done

echo "[start.sh] python=$(python --version 2>&1) at $(which python)"

# ---------- 安装依赖 ----------
REQ_FILE="requirements.txt"
STAMP_FILE="$VENV_DIR/.requirements.stamp"

need_install=0
if [ "$FORCE_INSTALL" = "1" ]; then
  echo "[start.sh] 强制重新安装依赖..."
  need_install=1
elif [ ! -f "$STAMP_FILE" ]; then
  need_install=1
elif [ "$REQ_FILE" -nt "$STAMP_FILE" ]; then
  need_install=1
fi

if [ "$need_install" = "1" ]; then
  echo "[start.sh] 安装/更新依赖..."
  python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
  python -m pip install -r "$REQ_FILE" -i https://pypi.tuna.tsinghua.edu.cn/simple
  touch "$STAMP_FILE"
else
  echo "[start.sh] 依赖已是最新，跳过安装 (删除 $STAMP_FILE 可强制重装)"
fi

# ---------- 导出运行参数 ----------
export ASR_HOST ASR_PORT ASR_MODEL_ID ASR_ALIGNER_ID ASR_ENABLE_ALIGN \
  ASR_SEG_GAP_SEC ASR_SEG_MAX_DURATION ASR_SEG_MAX_CHARS ASR_ALIGN_MAX_CHARS \
  ASR_MAX_QUEUE ASR_MAX_CONCURRENCY \
  ASR_TIMEOUT ASR_ALIGN_TIMEOUT ASR_WORKER_READY_TIMEOUT \
  ASR_MAX_NEW_TOKENS

echo "[start.sh] ========================================"
echo "[start.sh] MODEL        : $ASR_MODEL_ID"
echo "[start.sh] ALIGNER      : $ASR_ALIGNER_ID (enable=$ASR_ENABLE_ALIGN)"
echo "[start.sh] SEG          : gap=${ASR_SEG_GAP_SEC}s max_dur=${ASR_SEG_MAX_DURATION}s max_chars=${ASR_SEG_MAX_CHARS}"
echo "[start.sh] MODELS_DIR   : $MODELS_DIR"
echo "[start.sh] HOST:PORT    : $ASR_HOST:$ASR_PORT"
echo "[start.sh] MAX_QUEUE    : $ASR_MAX_QUEUE"
echo "[start.sh] MAX_CONCURR. : $ASR_MAX_CONCURRENCY"
echo "[start.sh] TIMEOUT      : asr=${ASR_TIMEOUT}s align=${ASR_ALIGN_TIMEOUT}s"
echo "[start.sh] MAX_NEW_TOK  : $ASR_MAX_NEW_TOKENS (0=unlimited)"
echo "[start.sh] ========================================"

# ---------- 启动 ----------
exec python app.py "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
