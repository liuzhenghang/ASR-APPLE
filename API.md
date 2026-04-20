# ASR-APPLE API 接口文档

> 基于 MLX + Qwen3-ASR 的语音识别 HTTP 服务  
> 服务地址：`http://0.0.0.0:8000`

---

## 1. 健康检查

### `GET /health`

检查模型加载状态与队列使用情况。

**请求参数**：无

**响应示例**：

```json
{
  "status": "ok",
  "model": "mlx-community/Qwen3-ASR-1.7B-8bit",
  "loaded": true,
  "in_flight": 1,
  "max_queue": 5,
  "max_concurrency": 1
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 服务状态，固定 `"ok"` |
| `model` | string | 当前加载的模型 ID |
| `loaded` | bool | 模型是否已加载完成 |
| `in_flight` | int | 当前排队/处理中的请求数 |
| `max_queue` | int | 最大排队数（环境变量 `ASR_MAX_QUEUE`） |
| `max_concurrency` | int | 最大并发数（环境变量 `ASR_MAX_CONCURRENCY`） |

---

## 2. 音频文件识别

### `POST /asr_file`

上传本地音频文件进行语音识别。

**Content-Type**: `multipart/form-data`

**请求参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audio_file` | file | 是 | 音频文件（支持 wav、mp3、m4a、flac 等格式） |
| `language` | string | 否 | 语言代码，如 `zh`、`en` |
| `only_text` | bool | 否 | 是否只返回 text 字段，默认 `false` |

**请求示例**（curl）：

```bash
curl -X POST http://localhost:8000/asr_file \
  -F "audio_file=@audio.wav" \
  -F "language=zh"
```

---

## 3. 音频 URL 识别

### `POST /asr_url`

通过音频 URL 下载后进行语音识别。

**Content-Type**: `application/json`

**请求体**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `url` | string | 是 | 音频文件的 HTTP/HTTPS URL |
| `language` | string | 否 | 语言代码，如 `zh`、`en` |
| `only_text` | bool | 否 | 是否只返回 text 字段，默认 `false` |

**请求示例**：

```json
{
  "url": "https://example.com/audio.wav",
  "language": "zh"
}
```

```bash
curl -X POST http://localhost:8000/asr_url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/audio.wav", "language": "zh"}'
```

---

## 4. 统一响应格式（成功）

`/asr_file` 和 `/asr_url` 成功时返回相同结构：

```json
{
  "text": "完整识别文本",
  "segments": [
    {
      "text": "第一句识别文本",
      "start": 0.0,
      "end": 2.5
    },
    {
      "text": "第二句识别文本",
      "start": 2.5,
      "end": 5.0
    }
  ],
  "language": "zh",
  "total_time": 5.0
}
```

当 `only_text=true` 时，响应为纯文本，`Content-Type: text/plain`：

```
完整识别文本
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | string | 完整识别文本 |
| `segments` | array | 分段识别结果 |
| `segments[].text` | string | 分段文本 |
| `segments[].start` | float | 分段起始时间（秒） |
| `segments[].end` | float | 分段结束时间（秒） |
| `language` | string | 检测到的语言代码 |
| `total_time` | float | 音频总时长（秒） |

---

## 5. 错误响应

所有接口出错时返回：

```json
{
  "detail": "错误描述信息"
}
```

**HTTP 状态码说明**：

| 状态码 | 场景 |
|--------|------|
| `400` | URL 下载失败、音频文件为空、远程文件为空 |
| `413` | 音频文件过大（超过 500MB） |
| `429` | 队列已满（达到 `ASR_MAX_QUEUE`），请稍后重试 |
| `500` | 模型识别失败 |
| `503` | 模型尚未加载完成 |

---

## 6. 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ASR_MODEL_ID` | `mlx-community/Qwen3-ASR-1.7B-8bit` | Hugging Face 模型 ID |
| `ASR_HOST` | `0.0.0.0` | 监听地址 |
| `ASR_PORT` | `8000` | 监听端口 |
| `ASR_MAX_QUEUE` | `5` | 最大排队数 |
| `ASR_MAX_CONCURRENCY` | `1` | 最大并发处理数 |
| `ASR_MAX_DOWNLOAD_BYTES` | `524288000` | 最大下载大小（500MB） |
| `ASR_DOWNLOAD_TIMEOUT` | `60` | URL 下载超时时间（秒） |
| `HF_ENDPOINT` | — | Hugging Face 镜像地址 |

---

## 7. 流控说明

- 服务使用信号量 + 计数器实现队列限流
- `ASR_MAX_CONCURRENCY`：同时执行模型识别的最大数量
- `ASR_MAX_QUEUE`：允许排队等候的最大请求数（含正在处理的）
- 当 `in_flight >= ASR_MAX_QUEUE` 时，新请求返回 `429`
- 排队释放遵循 FIFO 顺序
