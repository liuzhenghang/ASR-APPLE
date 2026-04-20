# ASR-APPLE

基于 MLX + Qwen3-ASR 的语音识别 HTTP 服务，使用 FastAPI 构建，支持 Apple Silicon (M 系列芯片)。

## 特性

- 两种识别接口：音频文件上传 / 音频 URL 下载
- 队列管理，支持并发限流
- 模型启动时自动加载，支持配置
- 内置健康检查接口

## 快速开始

```bash
# 一键启动（自动创建虚拟环境、安装依赖）
./start.sh
```

服务默认监听 `0.0.0.0:8000`。

## API 接口

### 健康检查

```
GET /health
```

返回模型加载状态、队列使用情况等。

### 音频文件识别

```
POST /asr_file
Content-Type: multipart/form-data

- audio_file: 音频文件 (必填)
- language: 语言代码 (可选)
```

### 音频 URL 识别

```
POST /asr_url
Content-Type: application/json

{
  "url": "https://example.com/audio.wav",
  "language": "zh"  // 可选
}
```

### 响应格式

```json
{
  "text": "完整识别文本",
  "segments": [
    {"text": "片段文本", "start": 0.0, "end": 2.5}
  ],
  "language": "zh",
  "total_time": 2.5
}
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ASR_MODEL_ID` | `mlx-community/Qwen3-ASR-1.7B-8bit` | Hugging Face 模型 ID |
| `ASR_HOST` | `0.0.0.0` | 监听地址 |
| `ASR_PORT` | `8000` | 监听端口 |
| `ASR_MAX_QUEUE` | `5` | 最大排队数 |
| `ASR_MAX_CONCURRENCY` | `1` | 最大并发数 |
| `ASR_MAX_DOWNLOAD_BYTES` | `524288000` | 最大下载大小 (500MB) |
| `ASR_DOWNLOAD_TIMEOUT` | `60` | 下载超时时间 (秒) |
| `HF_ENDPOINT` | - | Hugging Face 镜像地址 |

## 手动安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## 目录结构

```
.
├── app.py              # FastAPI 应用主文件
├── start.sh            # 一键启动脚本
├── requirements.txt    # Python 依赖
└── models/             # 模型缓存目录 (自动创建)
```
