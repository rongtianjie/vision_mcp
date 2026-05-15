# Vision MCP Server

为缺少多模态能力的 LLM（如 DeepSeek）提供图片理解能力。通过 OpenAI-compatible API 将图片转发至视觉模型，在 Claude Code 中以 MCP 工具形式暴露 `describe_image` 工具。

## 工作原理

```
User: "看看这张截图"
  → Claude (DeepSeek, 无视觉)
    → 调用 describe_image 工具
      → MCP Server: 读取图片 → Base64 编码 → 请求视觉模型 API
        ← 返回文字描述
  → Claude 基于描述回答
```

## 环境要求

- Python 3.10+
- 任一提供 `/chat/completions` 的视觉模型 API（SiliconFlow、vLLM、Ollama 等）

## 安装

```bash
cd vision-mcp
pip install -e .          # 可编辑模式，推荐
vision-mcp --help         # 验证安装
```

## 配置

### 1. 准备环境变量

项目提供 `.env.example` 作为模板，复制后填入实际值：

```bash
cp .env.example .env
```

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `VISION_API_BASE` | 是 | `http://localhost:8000/v1` | API 地址，不含 `/chat/completions` 后缀 |
| `VISION_API_KEY` | 按需 | `not-needed` | API 密钥（本地部署可留空） |
| `VISION_MODEL` | 是 | `qwen-vl-plus` | 模型名称 |
| `VISION_MAX_TOKENS` | 否 | `2000` | 单次响应最大 token |

### 2. Provider 配置参考

**SiliconFlow**

```bash
VISION_API_BASE=https://api.siliconflow.cn/v1
VISION_API_KEY=sk-your-key-here
VISION_MODEL=Qwen/Qwen3.6-35B-A3B
```

**本地 vLLM**

```bash
VISION_API_BASE=http://10.0.0.5:8000/v1
VISION_API_KEY=not-needed
VISION_MODEL=Qwen3-VL-32B-Instruct
```

**本地 Ollama**

```bash
VISION_API_BASE=http://localhost:11434/v1
VISION_API_KEY=not-needed
VISION_MODEL=llava:13b
```

**One-API 网关（代理 GPT-4o）**

```bash
VISION_API_BASE=https://your-gateway.com/v1
VISION_API_KEY=sk-your-gateway-key
VISION_MODEL=gpt-4o
```

### 3. 注册到 Claude Code

```bash
claude mcp add-json -s user vision '{
  "command": "vision-mcp",
  "args": [],
  "env": {
    "VISION_API_BASE": "https://api.siliconflow.cn/v1",
    "VISION_API_KEY": "sk-your-key-here",
    "VISION_MODEL": "Qwen/Qwen3.6-35B-A3B"
  }
}'
```

> `-s user` 注册为全局可用。`-s local` 仅当前项目可用，或使用 `.mcp.json` 在团队内共享（注意 API Key 会暴露）。

### 4. 验证

```bash
claude mcp list
# 应显示: vision: vision-mcp - ✓ Connected
```

重启 Claude Code 后生效。

### 更新配置

```bash
claude mcp remove vision -s user
claude mcp add-json -s user vision '{ ... }'
```

## 工具

### describe_image

| 参数 | 必填 | 说明 |
|------|------|------|
| `image_path` | 是 | 图片本地绝对路径 |
| `prompt` | 否 | 指定描述侧重点，如 "提取所有文字"、"描述图表数据趋势" |

支持的格式：PNG / JPG / JPEG / GIF / WebP / BMP，单文件不超过 20 MB。

### vision_ping

诊断工具，返回服务状态，用于排查 MCP 通信是否正常。

## 示例

```
User: 看看 @error_screenshot.png 里的报错信息
User: 分析 @architecture.png 的系统设计有什么问题
User: 把 @data_table.png 转成 markdown 表格
```

## 常见问题

### Failed to connect？

1. 确认使用 `pip install -e .` 安装，否则可能报 `ModuleNotFoundError`
2. 确认 `vision-mcp --help` 可正常执行
3. 检查 API 连通性：`curl $VISION_API_BASE/models`
4. 直接运行 `vision-mcp` 查看 stderr 错误信息

### 返回乱码或空内容？

- 将图片转为 PNG 格式
- 缩小图片尺寸（Base64 编码后体积增大约 33%，可能超 API 限制）
- 更换视觉模型

### 如何切换模型？

修改配置中的 `VISION_MODEL`，然后重新注册：

```bash
claude mcp remove vision -s user
claude mcp add-json -s user vision '{ ... }'
```

### 图片数据会留存吗？

图片经 Base64 编码后通过 HTTPS 发送至配置的 API 服务端，MCP Server 不做本地存储或缓存。使用公网 API 时注意不要传入敏感图片。
