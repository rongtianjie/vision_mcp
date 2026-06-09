# Vision MCP Server

为缺少多模态能力的 LLM（如 DeepSeek）提供图片理解能力。通过 OpenAI-compatible API 将图片转发至视觉模型，在 Code Agent 中以 MCP 工具形式暴露 `describe_image`。

---

## 一、快速上手

### 1. 安装

```bash
uv venv && uv pip install -e .
```

### 2. 注册到 Code Agent

```bash
claude mcp add-json -s user vision '{
  "command": "vision-mcp",
  "args": [],
  "env": {
    "VISION_API_BASE": "https://api.siliconflow.cn/v1",
    "VISION_API_KEY": "sk-your-key-here",
    "VISION_MODEL": "Qwen/Qwen3.6-35B-A3B",
    "VISION_MAX_TOKENS": "2000"
  }
}'
```

> `vision-mcp` 命令由 `pyproject.toml` 注册，`uv pip install -e .` 后自动可用。注册后**重启 Code Agent** 生效。

### 3. 验证

```bash
vision-mcp --help
claude mcp list           # → vision: vision-mcp - ✓ Connected
```

---

## 二、环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `VISION_API_BASE` | 是 | `http://localhost:8000/v1` | API 地址，不含 `/chat/completions` 后缀 |
| `VISION_API_KEY` | 按需 | `not-needed` | API 密钥（本地部署留空即可） |
| `VISION_MODEL` | 是 | `qwen-vl-plus` | 视觉模型名称 |
| `VISION_MAX_TOKENS` | 否 | `2000` | 单次响应最大 token 数 |

> 环境变量通过注册命令的 `env` 字段固化到 Agent 配置中。修改后需**重新执行注册命令**。

### Provider 配置参考

| Provider | API_BASE | 推荐模型 |
|----------|----------|---------|
| SiliconFlow | `https://api.siliconflow.cn/v1` | `Qwen/Qwen3.6-35B-A3B` |
| 本地 vLLM | `http://10.0.0.5:8000/v1` | `Qwen3-VL-32B-Instruct` |
| 本地 Ollama | `http://localhost:11434/v1` | `llava:13b` |
| One-API 网关 | `https://your-gateway.com/v1` | `gpt-4o` |

---

## 三、工作原理

```
User: "看看这张截图"
  → Code Agent (DeepSeek, 无视觉)
    → 调用 describe_image 工具
      → Vision MCP Server: 读取本地图片 → Base64 → POST 视觉模型 API
        ← 返回文字描述
  → Code Agent 基于描述回答用户
```

---

## 四、工具说明

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

## 五、使用示例

```
User: 看看 @error_screenshot.png 里的报错信息
User: 分析 @architecture.png 的系统设计有什么问题
User: 把 @data_table.png 转成 markdown 表格
User: @chart.png 描述数据变化趋势，控制在 200 字以内
```

---

## 六、更新配置

修改环境变量后重新注册：

```bash
claude mcp remove vision -s user
claude mcp add-json -s user vision '{
  "command": "vision-mcp",
  "args": [],
  "env": {
    "VISION_API_BASE": "...",
    "VISION_API_KEY": "...",
    "VISION_MODEL": "...",
    "VISION_MAX_TOKENS": "4096"
  }
}'
```

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
