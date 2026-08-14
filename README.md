# Vision MCP Server

为缺少多模态能力的 LLM（如 DeepSeek）提供图片理解能力。通过 OpenAI-compatible API 将图片转发至视觉模型，以 MCP 工具形式暴露 `describe_image`，可接入任意支持 MCP 协议的 AI Agent（Claude Code、Cline、Continue.dev 等）。

## 特性

- **多种图片来源**：本地路径（绝对/相对）、`http(s)://` 网络图片 URL、`data:` URI，无需先下载到本地
- **多图分析**：一次传入多张图片进行对比、总结
- **自动压缩**：超过 `VISION_MAX_UPLOAD_MB` 的图片自动缩放/转 JPEG（最长边 2048px），降低上传体积与 API 拒绝率
- **多后端（provider）**：可同时配置多个视觉 API，按需切换（工具参数或启动参数）
- **健壮性**：瞬时错误自动重试（尊重 `Retry-After` 头 + 指数退避抖动）、响应结构防御性校验、错误按 HTTP 状态分类给出中文提示
- **诊断工具**：`vision_ping` 支持探测视觉 API 连通性（`GET /models`）
- **配置友好**：所有配置通过项目根目录 `.env` 管理，修改后重启 Agent 即生效，**无需重新注册 MCP**

---

## 一、安装

### 方式 A：项目内安装（推荐用于开发调试）

```bash
cd /path/to/vision_mcp
uv venv && uv pip install -e .
```

### 方式 B：全局安装（推荐日常使用，注册命令不依赖工作目录）

```bash
uv tool install -e /path/to/vision_mcp
```

安装后 `vision-mcp` 进入 PATH，可在任意目录直接运行（此方式注册最省心）。

---

## 二、配置

所有后端配置通过项目根目录 `.env` 文件管理，**无需在 MCP 注册时反复传入环境变量**。修改配置后重启 Agent 即可生效。

```bash
cp .env.example .env
```

`.env` 文件内容：

```
VISION_API_BASE=https://api.siliconflow.cn/v1
VISION_API_KEY=sk-your-key-here
VISION_MODEL=Qwen/Qwen2.5-VL-72B-Instruct
VISION_MAX_TOKENS=2000
```

### 配置变量总表

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `VISION_API_BASE` | 是 | `http://localhost:8000/v1` | API 地址，不含 `/chat/completions` 后缀 |
| `VISION_API_KEY` | 按需 | `not-needed` | API 密钥（本地部署留空即可） |
| `VISION_MODEL` | 是 | `qwen-vl-plus` | 视觉模型名称 |
| `VISION_MAX_TOKENS` | 否 | `2000` | 单次响应最大 token 数（可被工具参数覆盖） |
| `VISION_TIMEOUT` | 否 | `120` | API 请求超时（秒） |
| `VISION_MAX_RETRIES` | 否 | `2` | 瞬时错误（429/5xx/超时）最大重试次数 |
| `VISION_MAX_IMAGE_MB` | 否 | `20` | 单张图片最大体积（MB），超过拒绝 |
| `VISION_MAX_UPLOAD_MB` | 否 | `8` | 超过该体积的图片自动压缩后再上传 |
| `VISION_PROVIDER` | 否 | `default` | 默认使用的 provider 名称 |

> 非法配置值（如 `VISION_MAX_TOKENS=abc`）不会导致启动失败，会告警并回退默认值。启动日志会打印配置摘要（不含 API Key），并提示未找到 `.env` 的情况。

### 多后端（provider）配置

适合同时连接「公网 + 本地」多个视觉服务。命名 provider 的环境变量格式：

```
VISION_PROVIDER_<NAME>_API_BASE=...
VISION_PROVIDER_<NAME>_API_KEY=...
VISION_PROVIDER_<NAME>_MODEL=...
```

示例（`.env` 中）：

```
# 默认 provider（公网）
VISION_API_BASE=https://api.siliconflow.cn/v1
VISION_API_KEY=sk-xxx
VISION_MODEL=Qwen/Qwen2.5-VL-72B-Instruct

# 命名 provider：local
VISION_PROVIDER_LOCAL_API_BASE=http://localhost:11434/v1
VISION_PROVIDER_LOCAL_API_KEY=not-needed
VISION_PROVIDER_LOCAL_MODEL=llava:13b

VISION_PROVIDER=default   # 默认使用哪个
```

切换后端有三种方式（优先级从高到低）：

1. 调用工具时传 `provider` 参数（如 `provider="local"`），仅本次生效
2. 启动时 `vision-mcp --provider local`，本次会话生效
3. 修改 `VISION_PROVIDER` 环境变量，重启 Agent 生效

### 启动参数

```
vision-mcp [--env-file PATH] [--provider NAME] [--version]

--env-file PATH   从指定 .env 文件加载配置（优先级最高，可替代自动发现的 .env）
--provider NAME   默认使用的 provider 名称
```

> `.env` 查找顺序：当前工作目录 → 项目根目录。若你的 Agent 不在项目根目录启动（如 Claude Code 的 user 级注册），请用 `--env-file /path/to/vision_mcp/.env` 显式指定。

---

## 三、注册到 AI Agent

### 方式 A：使用项目内 `.mcp.json`（Cline、Continue.dev 等）

项目已内置 `.mcp.json`，支持此格式的 Agent 会自动识别：

```json
{
  "mcpServers": {
    "vision": {
      "command": "uv",
      "args": ["run", "vision-mcp"]
    }
  }
}
```

> ⚠️ 该写法要求 Agent 以**项目根目录**为工作目录（Cline/Continue 以项目根打开时满足）。若你的客户端不满足，请改用绝对路径：
> ```json
> { "mcpServers": { "vision": { "command": "uv", "args": ["run", "--project", "/path/to/vision_mcp", "vision-mcp"] } } }
> ```

### 方式 B：Claude Code 手动注册

**推荐（已全局安装）**：

```bash
claude mcp add-json -s user vision '{
  "command": "vision-mcp",
  "args": []
}'
```

**未全局安装**（必须带绝对路径，否则 `uv run` 找不到包）：

```bash
claude mcp add-json -s user vision '{
  "command": "uv",
  "args": ["run", "--project", "/path/to/vision_mcp", "vision-mcp"]
}'
```

### 方式 C：`uv tool install` + 任意注册

按「一、安装」方式 B 全局安装后，任何注册命令都只需 `"command": "vision-mcp"`，无 cwd 依赖。

> 所有方式均**无需携带 `env` 字段**，配置已由 `.env` 文件管理。

---

## 四、验证

```bash
vision-mcp --version          # 输出版本号
vision-mcp --help             # 查看启动参数
```

确认 Agent 中 MCP Server 状态为已连接（Claude Code: `claude mcp list` → `vision: vision-mcp - ✓ Connected`）。连接后调用 `vision_ping`（`probe_api: true` 可同时检查视觉 API 连通性）。

---

## 五、工作原理

```
User: "看看这张截图"
  → AI Agent (DeepSeek, 无视觉)
    → 调用 describe_image 工具
      → Vision MCP Server: 读取本地图片 / 下载 URL → Base64 → POST 视觉模型 API
        ← 返回文字描述
  → AI Agent 基于描述回答用户
```

---

## 六、工具说明

### describe_image — 理解图片内容

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `image_path` | 否* | `string` | 本地图片路径（绝对或相对，相对基于 Agent 启动目录），兼容旧版单图用法 |
| `image_url` | 否* | `string` | 网络图片 URL（http/https），自动下载 |
| `images` | 否* | `string[]` | 多张图片（路径/URL 混合），用于对比或总结 |
| `prompt` | 否 | `string` | 描述侧重，如 "提取所有文字"、"描述图表趋势" |
| `max_tokens` | 否 | `int` | 覆盖环境变量 `VISION_MAX_TOKENS` |
| `provider` | 否 | `string` | 指定后端（需在 `.env` 中配置，如 `"local"`） |

\* 三个图片来源参数至少提供一个；同时提供时自动合并。也支持 `data:` URI。

- **支持格式**：PNG / JPG / JPEG / JFIF / GIF / WebP / BMP / HEIC / AVIF / TIFF
- **单文件限制**：≤ `VISION_MAX_IMAGE_MB`（默认 20 MB）
- **自动压缩**：超过 `VISION_MAX_UPLOAD_MB`（默认 8 MB）的图片自动缩放/转 JPEG 后上传
- **多图**：多张图按顺序放入一次请求（OpenAI-compatible 多模态格式）

### vision_ping — 诊断连通性

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `msg` | 否 | `string` | 任意测试字符串，原样回显 |
| `probe_api` | 否 | `bool` | 是否额外请求 `GET /models` 检查默认 provider 的 API 连通性 |

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call",\
  "params":{"name":"vision_ping","arguments":{"msg":"hello"}}}' | vision-mcp
```

---

## 七、使用示例

```
User: 看看 @error_screenshot.png 里的报错信息
User: 分析 @architecture.png 的系统设计有什么问题
User: 把 @data_table.png 转成 markdown 表格
User: @chart.png 描述数据变化趋势，控制在 200 字以内
User: 对比 @before.png 和 @after.png 的界面差异        # 多图
User: 分析 https://example.com/chart.png 的数据趋势     # 网络图片
```

---

## 八、更新配置

修改 `.env` 文件后**重启 AI Agent** 即可生效，无需重新注册 MCP。

如需更换 Provider 的 API 地址，可参考以下配置：

| Provider | `VISION_API_BASE` | 推荐 `VISION_MODEL` |
|----------|-------------------|--------------------|
| SiliconFlow | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-VL-72B-Instruct` |
| 本地 vLLM | `http://10.0.0.5:8000/v1` | `Qwen2.5-VL-72B-Instruct` |
| 本地 Ollama | `http://localhost:11434/v1` | `llava:13b` |
| One-API 网关 | `https://your-gateway.com/v1` | `gpt-4o` |

---

## 常见问题

### Failed to connect？

1. 确认 `vision-mcp --version` 可执行
2. 检查 API 连通性：`curl $VISION_API_BASE/models`
3. 运行 `vision-mcp` 查看 stderr 日志（启动时会打印配置摘要与 `.env` 加载情况）
4. Agent 内调用 `vision_ping`，`probe_api: true` 一键排查

### 注册后连不上，日志显示找不到 vision-mcp？

`uv run vision-mcp` 依赖工作目录。请改用绝对路径：`uv run --project /path/to/vision_mcp vision-mcp`，或按「方式 C」全局安装后直接注册 `vision-mcp`。

### 返回乱码或空内容？

- 尝试将图片转为 PNG 格式
- 缩小图片尺寸（Base64 后体积增大约 33%）
- 更换更强大的视觉模型

### 图片数据会留存吗？

不会。图片仅通过 Base64 编码后 HTTPS 发送至视觉 API，Server 不做本地存储或缓存。使用公网 API 时注意勿传入敏感图片。

### 如何临时切换视觉服务？

调用工具时传 `provider` 参数（需先在 `.env` 配置好命名 provider），或在启动命令加 `--provider <名称>`。

### HEIC 图片无法识别？

扩展名识别支持 HEIC，但上传前压缩需要 [pillow-heif](https://pypi.org/project/pillow-heif/)；未安装时会原样上传，由视觉 API 决定是否支持。
