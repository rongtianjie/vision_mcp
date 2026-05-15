# Vision MCP Server

为没有多模态能力的 LLM（如 DeepSeek）提供图片理解能力。通过 OpenAI-compatible API 调用外部视觉模型（Qwen-VL、GPT-4o 等），在 Claude Code 中以 MCP 工具的形式呈现。

## 工作原理

```
User: "看看这张截图"
  → Claude (DeepSeek, 无视觉)
    → 调用 describe_image 工具
      → MCP Server 读取图片 → Base64 编码
        → 请求视觉模型 API
          ← 返回文字描述
    → Claude 基于描述回答用户
```

## 环境要求

- Python 3.10+
- 一个 OpenAI-compatible 的视觉模型 API（如 SiliconFlow、vLLM、Ollama 等）

## 安装

### 从源码安装（可编辑模式，推荐）

```bash
# 1. 进入项目目录
cd vision-mcp

# 2. 可编辑模式安装（修改源码无需重新安装）
pip install -e .

# 3. 验证
vision-mcp --help
```

### 从 GitHub 安装（如果有仓库）

```bash
pip install git+https://github.com/your-org/vision-mcp.git
```

## 配置视觉模型 API

### 支持的 Provider

任何 OpenAI-compatible 的 `/chat/completions` 接口都可以使用。以下是验证过的 provider：

| Provider | 模型示例 | 说明 |
|----------|---------|------|
| [SiliconFlow](https://siliconflow.cn) | `Qwen/Qwen3.6-35B-A3B`、`Qwen/Qwen3-VL-32B-Instruct` | 国内首推，稳定快速 |
| vLLM 自部署 | 任意 VL 模型 | 内网部署，无数据外泄 |
| Ollama | `llava`、`minicpm-v` | 本地运行，免费 |
| OpenAI 兼容代理 | `gpt-4o`、`gpt-4-vision` | 通过 one-api 等网关 |

### 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `VISION_API_BASE` | 是 | `http://localhost:8000/v1` | API 基础 URL，无需带 `/chat/completions` |
| `VISION_API_KEY` | 按需 | `not-needed` | API 密钥，本地部署可留空 |
| `VISION_MODEL` | 是 | `qwen-vl-plus` | 模型名称 |
| `VISION_MAX_TOKENS` | 否 | `2000` | 最大输出 token 数 |

### 各 Provider 配置示例

**SiliconFlow：**

```bash
VISION_API_BASE=https://api.siliconflow.cn/v1
VISION_API_KEY=sk-your-key-here
VISION_MODEL=Qwen/Qwen3.6-35B-A3B
```

**本地 vLLM：**

```bash
VISION_API_BASE=http://10.0.0.5:8000/v1
VISION_API_KEY=not-needed
VISION_MODEL=Qwen3-VL-32B-Instruct
```

**本地 Ollama：**

```bash
VISION_API_BASE=http://localhost:11434/v1
VISION_API_KEY=not-needed
VISION_MODEL=llava:13b
```

**One-API 网关（代理 GPT-4o）：**

```bash
VISION_API_BASE=https://your-gateway.com/v1
VISION_API_KEY=sk-your-gateway-key
VISION_MODEL=gpt-4o
```

## 快速配置

### 1. 创建配置文件

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API 信息：

```env
VISION_API_BASE=https://api.siliconflow.cn/v1
VISION_API_KEY=sk-your-key-here
VISION_MODEL=Qwen/Qwen3.6-35B-A3B
VISION_MAX_TOKENS=2048
```

### 2. 注册到 Claude Code

**直接注册（环境变量写在 MCP 配置中）：**

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

### 3. 验证

```bash
claude mcp list
# 应显示: vision: vision-mcp - ✓ Connected
```

重启 Claude Code 即可使用。

### 更新已有注册

```bash
# 先删除旧配置
claude mcp remove vision -s user

# 再重新注册
claude mcp add-json -s user vision '{ ... 新配置 ... }'
```

### 其他注册方式

| 方式 | 命令 | 说明 |
|------|------|------|
| 用户级 | `-s user` | 所有项目可用（推荐） |
| 项目级 | `-s local` | 仅当前项目可用 |
| 项目共享 | `.mcp.json` | 团队成员共用（注意 API Key 会暴露） |

## 可用工具

### `describe_image`

| 参数 | 必填 | 说明 |
|------|------|------|
| `image_path` | 是 | 图片文件的本地绝对路径 |
| `prompt` | 否 | 描述侧重点，如 "描述图表中的数据趋势"、"提取所有文字" |

支持的图片格式：PNG、JPG、JPEG、GIF、WebP、BMP

图片大小限制：20MB 以内

### `vision_ping`

诊断工具，返回服务状态信息，用于排查 MCP 通信是否正常。

## 使用示例

```
# 截了一张报错图
User: "帮我看看 @error_screenshot.png 里的报错信息是什么"

# 分析架构图
User: "用 describe_image 看一下 @architecture.png，这个系统设计有什么问题"

# 提取表格数据
User: "读取 @data_table.png，把表格转成 markdown"
```

## 常见问题

### Q: `claude mcp list` 显示 Failed to connect？

1. 确认使用 `pip install -e .`（可编辑模式），而非 `pip install .`，否则可能找不到模块
2. 确认 `vision-mcp --help` 可正常执行
3. 检查 `VISION_API_BASE` 是否可访问：`curl $VISION_API_BASE/models`
4. 直接运行 `vision-mcp` 看 stderr 有无报错（如 `ModuleNotFoundError: No module named 'vision_mcp'` 说明需要 `pip install -e .`）

### Q: 工具返回乱码或空？

可能是视觉模型不支持该图片格式。尝试：
- 将图片转为 PNG 格式
- 缩小图片尺寸（大图 base64 编码后可能超出 API 限制）
- 换一个视觉模型

### Q: 如何换模型？

修改 `.env` 或 MCP 注册配置中的 `VISION_MODEL`，然后重新注册：

```bash
claude mcp remove vision -s user
claude mcp add-json -s user vision '{ ... 新配置 ... }'
```

### Q: 图片数据会存在哪里？

图片以 Base64 格式通过 HTTPS 发送到配置的 API 服务器，不会被 MCP Server 本地存储或缓存。如果使用公网 API（如 SiliconFlow），请勿传入敏感图片。
