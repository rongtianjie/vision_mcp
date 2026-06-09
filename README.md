# Vision MCP Server

为缺少多模态能力的 LLM（如 DeepSeek）提供图片理解能力。通过 OpenAI-compatible API 将图片转发至视觉模型，以 MCP 工具形式暴露 `describe_image`，可接入任意支持 MCP 协议的 AI Agent（Claude Code、Cline、Continue.dev 等）。

---

## 一、配置方式

所有后端配置（API 地址、密钥、模型名）通过项目根目录的 `.env` 文件管理，**无需在 MCP 注册时反复传入环境变量**。修改配置后重启 Agent 即可生效。

### 1. 安装

```bash
uv venv && uv pip install -e .
```

### 2. 配置后端

复制配置模板并填入实际值：

```bash
cp .env.example .env
```

`.env` 文件内容：

```
VISION_API_BASE=https://api.siliconflow.cn/v1      # 视觉模型 API 地址
VISION_API_KEY=sk-your-key-here                      # API 密钥
VISION_MODEL=Qwen/Qwen3.6-35B-A3B                    # 模型名称
VISION_MAX_TOKENS=2000                                # 最大输出 token 数（可选）
```

> `.env` 由 Server 启动时自动加载，修改后重启 Agent 即可生效，**无需重新注册 MCP**。

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `VISION_API_BASE` | 是 | `http://localhost:8000/v1` | API 地址，不含 `/chat/completions` 后缀 |
| `VISION_API_KEY` | 按需 | `not-needed` | API 密钥（本地部署留空即可） |
| `VISION_MODEL` | 是 | `qwen-vl-plus` | 视觉模型名称 |
| `VISION_MAX_TOKENS` | 否 | `2000` | 单次响应最大 token 数 |

### 3. 注册到 AI Agent

**方式 A（推荐）— 直接使用 `.mcp.json`**

项目已内置 `.mcp.json` 文件，支持此格式的 Agent（Cline、Continue.dev 等）会自动识别：

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

**方式 B — Claude Code 手动注册**

```bash
claude mcp add-json -s user vision '{
  "command": "uv",
  "args": ["run", "vision-mcp"]
}'
```

> 所有方式均**无需携带 `env` 字段**，配置已由 `.env` 文件管理。

### 4. 验证

```bash
vision-mcp --help
```

确认 Agent 中 MCP Server 状态为已连接（Claude Code: `claude mcp list` → `vision: vision-mcp - ✓ Connected`）。

---

## 二、工作原理

```
User: "看看这张截图"
  → AI Agent (DeepSeek, 无视觉)
    → 调用 describe_image 工具
      → Vision MCP Server: 读取本地图片 → Base64 → POST 视觉模型 API
        ← 返回文字描述
  → AI Agent 基于描述回答用户
```

---

## 三、工具说明

### describe_image — 理解图片内容

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `image_path` | 是 | `string` | 图片本地**绝对路径** |
| `prompt` | 否 | `string` | 描述侧重，如 "提取所有文字"、"描述图表趋势" |
| `max_tokens` | 否 | `int` | 覆盖环境变量 `VISION_MAX_TOKENS`，按需控制输出长度 |

- **支持格式**：PNG / JPG / JPEG / GIF / WebP / BMP
- **单文件限制**：≤ 20 MB

### vision_ping — 诊断连通性

返回服务器状态信息，用于排查 MCP 通信是否正常。

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call",\
  "params":{"name":"vision_ping","arguments":{"msg":"hello"}}}' | vision-mcp
```

---

## 四、使用示例

```
User: 看看 @error_screenshot.png 里的报错信息
User: 分析 @architecture.png 的系统设计有什么问题
User: 把 @data_table.png 转成 markdown 表格
User: @chart.png 描述数据变化趋势，控制在 200 字以内
```

---

## 五、更新配置

修改 `.env` 文件后**重启 AI Agent** 即可生效，无需重新注册 MCP。

如需更换 Provider 的 API 地址，可参考以下配置：

| Provider | `VISION_API_BASE` | 推荐 `VISION_MODEL` |
|----------|-------------------|--------------------|
| SiliconFlow | `https://api.siliconflow.cn/v1` | `Qwen/Qwen3.6-35B-A3B` |
| 本地 vLLM | `http://10.0.0.5:8000/v1` | `Qwen3-VL-32B-Instruct` |
| 本地 Ollama | `http://localhost:11434/v1` | `llava:13b` |
| One-API 网关 | `https://your-gateway.com/v1` | `gpt-4o` |

---

## 常见问题

### Failed to connect？

1. 确认 `vision-mcp --help` 可执行
2. 检查 API 连通性：`curl $VISION_API_BASE/models`
3. 运行 `vision-mcp` 查看 stderr 日志

### 返回乱码或空内容？

- 尝试将图片转为 PNG 格式
- 缩小图片尺寸（Base64 后体积增大约 33%）
- 更换更强大的视觉模型

### 图片数据会留存吗？

不会。图片仅通过 Base64 编码后 HTTPS 发送至视觉 API，Server 不做本地存储或缓存。使用公网 API 时注意勿传入敏感图片。
