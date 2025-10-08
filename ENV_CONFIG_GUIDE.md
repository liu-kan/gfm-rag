# GFM-RAG 环境变量配置扩展使用指南

## 概述

GFM-RAG 现在支持通过环境变量动态配置LLM服务，无需修改YAML配置文件即可切换不同的模型服务提供商。

## 新增功能

### 支持的环境变量

| 环境变量 | 描述 | 示例值 | 必需 |
|----------|------|--------|------|
| `GFMRAG_CHAT_PROVIDER` | LLM服务提供商 | `openai`, `third-party`, `ollama` | 否 |
| `GFMRAG_CHAT_MODEL_NAME` | 模型名称 | `gpt-4o-mini`, `Qwen3-VL-30B` | 否 |
| `GFMRAG_CHAT_BASE_URL` | 第三方服务URL | `http://localhost:8000/v1` | 否 |
| `GFMRAG_CHAT_KEY` | API密钥 | `sk-your-api-key` | 否 |

### 配置优先级

1. **函数参数** (最高优先级)
2. **环境变量** 
3. **YAML配置文件默认值** (最低优先级)

## 使用示例

### 场景1：使用第三方本地LLM服务

```bash
# 设置环境变量
export GFMRAG_CHAT_PROVIDER="third-party"
export GFMRAG_CHAT_MODEL_NAME="Qwen3-VL-30B-A3B-Instruct-FP8"
export GFMRAG_CHAT_BASE_URL="http://192.168.110.11:8888/v1"
export GFMRAG_CHAT_KEY="your-api-key"  # 可选

# 运行系统
python -m gfmrag.workflow.stage1_index_dataset
```

### 场景2：使用OpenAI服务

```bash
# 设置环境变量
export GFMRAG_CHAT_PROVIDER="openai"
export GFMRAG_CHAT_MODEL_NAME="gpt-4o-mini"
export GFMRAG_CHAT_KEY="sk-your-openai-key"

# 运行系统
python -m gfmrag.workflow.stage1_index_dataset
```

### 场景3：使用本地Ollama服务

```bash
# 设置环境变量
export GFMRAG_CHAT_PROVIDER="ollama"
export GFMRAG_CHAT_MODEL_NAME="llama3"

# 运行系统
python -m gfmrag.workflow.stage1_index_dataset
```

## 配置验证

### 快速验证脚本

运行简单的配置测试：

```bash
python3 test_env_config.py
```

### 完整验证脚本

运行详细的配置验证（需要安装依赖）：

```bash
python scripts/validate_env_config.py
```

## 向后兼容性

- 如果未设置环境变量，系统将使用YAML配置文件中的默认值
- 现有的配置和代码无需修改即可继续工作
- 支持渐进式迁移到环境变量配置

## 修改的文件

### YAML配置文件
- `gfmrag/workflow/config/ner_model/llm_ner_model.yaml`
- `gfmrag/workflow/config/openie_model/llm_openie_model.yaml`

### 模型类
- `gfmrag/kg_construction/ner_model/llm_ner_model.py`
- `gfmrag/kg_construction/openie_model/llm_openie_model.py`

### 工具函数
- `gfmrag/kg_construction/langchain_util.py`

### 新增文件
- `scripts/validate_env_config.py` - 完整的配置验证工具
- `test_env_config.py` - 快速配置测试
- `ENV_CONFIG_GUIDE.md` - 本使用指南

## 故障排除

### 常见问题

1. **环境变量不生效**
   - 检查环境变量是否正确设置：`echo $GFMRAG_CHAT_PROVIDER`
   - 确保在同一shell会话中运行程序

2. **第三方服务连接失败**
   - 验证 `GFMRAG_CHAT_BASE_URL` 的URL格式和可达性
   - 检查网络连接和防火墙设置

3. **API认证失败**
   - 验证 `GFMRAG_CHAT_KEY` 的有效性
   - 某些服务可能不需要API密钥，留空即可

### 调试步骤

1. 运行配置验证脚本
2. 检查环境变量设置
3. 测试网络连接
4. 验证API密钥

## 最佳实践

1. **开发环境**：使用 `.env` 文件管理环境变量
2. **生产环境**：通过部署系统设置环境变量
3. **容器化**：在 `Dockerfile` 或 `docker-compose.yml` 中设置
4. **CI/CD**：在构建脚本中配置不同环境的变量

## 示例配置文件

### .env 文件示例

```bash
# GFM-RAG LLM 配置
GFMRAG_CHAT_PROVIDER=third-party
GFMRAG_CHAT_MODEL_NAME=Qwen3-VL-30B-A3B-Instruct-FP8
GFMRAG_CHAT_BASE_URL=http://192.168.110.11:8888/v1
GFMRAG_CHAT_KEY=your-api-key
```

### Docker Compose 示例

```yaml
version: '3.8'
services:
  gfmrag:
    build: .
    environment:
      - GFMRAG_CHAT_PROVIDER=third-party
      - GFMRAG_CHAT_MODEL_NAME=Qwen3-VL-30B-A3B-Instruct-FP8
      - GFMRAG_CHAT_BASE_URL=http://llm-service:8888/v1
      - GFMRAG_CHAT_KEY=your-api-key
```

这些配置扩展使得GFM-RAG系统更加灵活，支持在不同环境中无缝切换LLM服务提供商。