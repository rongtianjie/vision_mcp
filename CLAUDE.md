# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Vision MCP Server — exposes an MCP tool (`describe_image`) that lets non-multimodal LLMs (e.g., DeepSeek) understand images by proxying them to a vision-capable model via any OpenAI-compatible `/chat/completions` API.

## Commands

```bash
# Install (editable)
pip install -e .

# Verify CLI
vision-mcp --help

# Test the MCP server locally (reads stdin/stdout per MCP stdio protocol)
vision-mcp
```

There are no lint, test, or type-check scripts configured yet.

## Architecture

```
vision_mcp/
  __init__.py    → re-exports main() from server.py
  __main__.py    → calls main() — enables `python -m vision_mcp`
  server.py      → all logic: config, two MCP tools, entry point
```

`server.py` is the single substantive file (~120 lines). It uses FastMCP from the `mcp` SDK:

- **Configuration**: Reads `VISION_API_BASE`, `VISION_API_KEY`, `VISION_MODEL`, `VISION_MAX_TOKENS` from env vars at module level.
- **`vision_ping`** tool: diagnostic echo — verifies the MCP transport is working.
- **`describe_image`** tool: reads a local image path, encodes it as Base64 data URI, POSTs it as an `image_url` content block to `{API_BASE}/chat/completions`, and returns the model's text response.
- **`main()`**: runs the server over stdio via `asyncio.run(mcp.run_stdio_async())`.

Dependencies (from `pyproject.toml`): `httpx>=0.24`, `mcp>=1.0`. The CLI entry point `vision-mcp` maps to `vision_mcp:main`.

## Registration with Claude Code

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
