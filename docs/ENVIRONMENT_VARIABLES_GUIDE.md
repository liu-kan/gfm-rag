# GFM-RAG 环境变量配置指南

本文档详细介绍如何使用 GFM-RAG 的环境变量功能来配置 Chat 和 Embedding 服务。

## 概述

GFM-RAG 现在支持通过环境变量配置第三方自建 LLM 和 Embedding 服务，使其能够在 `python -m gfmrag.workflow.stage1_index_dataset` 执行过程中无缝集成各种服务提供商。

### 主要特性

- ✅ **环境变量优先级**: 环境变量 > YAML 配置文件 > 默认值
- ✅ **独立认证密钥**: Chat 和 Embedding 服务可使用不同的认证凭据
- ✅ **无认证模式**: 支持本地服务（如 Ollama、本地 API 服务器）
- ✅ **第三方服务支持**: 兼容 OpenAI API 的第三方服务（vLLM、llama.cpp 等）
- ✅ **配置验证**: 自动验证配置的正确性和兼容性
- ✅ **向后兼容**: 保持与现有配置系统的完全兼容

## 支持的环境变量

### Chat 服务配置

| 环境变量 | 描述 | 示例值 | 必需 |
|---------|------|--------|------|
| `GFMRAG_CHAT_PROVIDER` | Chat服务提供商 | `openai`, `third-party`, `ollama` | 可选 |
| `GFMRAG_CHAT_MODEL_NAME` | Chat模型名称 | `gpt-3.5-turbo`, `llama3` | 可选 |
| `GFMRAG_CHAT_BASE_URL` | 第三方服务API端点 | `http://localhost:8000/v1` | 第三方服务必需 |
| `GFMRAG_CHAT_KEY` | Chat服务认证密钥 | `sk-xxx`, 空值表示无认证 | 可选 |

### Embedding 服务配置

| 环境变量 | 描述 | 示例值 | 必需 |
|---------|------|--------|------|
| `GFMRAG_EMBEDDING_PROVIDER` | Embedding服务提供商 | `openai`, `third-party`, `huggingface` | 可选 |
| `GFMRAG_EMBEDDING_MODEL_NAME` | Embedding模型名称 | `text-embedding-ada-002` | 可选 |
| `GFMRAG_EMBEDDING_BASE_URL` | 第三方服务API端点 | `http://localhost:8001/v1` | 第三方服务必需 |
| `GFMRAG_EMBEDDING_KEY` | Embedding服务认证密钥 | `sk-xxx`, 空值表示无认证 | 可选 |

## 支持的服务提供商

### 标准提供商
- `openai`: OpenAI 官方 API
- `together`: Together AI 服务  
- `nvidia`: NVIDIA AI 端点
- `anthropic`: Anthropic Claude API
- `huggingface`: HuggingFace 模型

### 自建服务提供商
- `third-party`: OpenAI 兼容的第三方服务
- `ollama`: 本地 Ollama 服务（通常无需认证）
- `llama.cpp`: llama.cpp 服务器（可选认证）
- `vllm`: vLLM 推理服务（可选认证）

## 使用示例

### 示例 1: 使用 OpenAI 官方服务

```bash
export GFMRAG_CHAT_PROVIDER=\"openai\"
export GFMRAG_CHAT_MODEL_NAME=\"gpt-3.5-turbo\"
export GFMRAG_CHAT_KEY=\"sk-your-openai-key\"
export GFMRAG_EMBEDDING_PROVIDER=\"openai\"
export GFMRAG_EMBEDDING_KEY=\"sk-your-openai-key\"

python -m gfmrag.workflow.stage1_index_dataset
```

### 示例 2: 使用本地 Ollama + OpenAI Embedding

```bash
export GFMRAG_CHAT_PROVIDER=\"ollama\"
export GFMRAG_CHAT_MODEL_NAME=\"llama3\"
# GFMRAG_CHAT_KEY 留空，ollama 通常无需认证
export GFMRAG_EMBEDDING_PROVIDER=\"openai\"
export GFMRAG_EMBEDDING_KEY=\"sk-your-openai-key\"

python -m gfmrag.workflow.stage1_index_dataset
```

### 示例 3: 使用第三方 vLLM 服务（无认证）

```bash
export GFMRAG_CHAT_PROVIDER=\"third-party\"
export GFMRAG_CHAT_MODEL_NAME=\"meta-llama/Llama-2-7b-chat-hf\"
export GFMRAG_CHAT_BASE_URL=\"http://localhost:8000/v1\"
# GFMRAG_CHAT_KEY 未设置，使用无认证模式
export GFMRAG_EMBEDDING_PROVIDER=\"huggingface\"
export GFMRAG_EMBEDDING_MODEL_NAME=\"sentence-transformers/all-MiniLM-L6-v2\"

python -m gfmrag.workflow.stage1_index_dataset
```

### 示例 4: 混合第三方服务（不同认证）

```bash
export GFMRAG_CHAT_PROVIDER=\"third-party\"
export GFMRAG_CHAT_MODEL_NAME=\"custom-chat-model\"
export GFMRAG_CHAT_BASE_URL=\"http://chat-service:8000/v1\"
export GFMRAG_CHAT_KEY=\"chat-specific-key\"
export GFMRAG_EMBEDDING_PROVIDER=\"third-party\"
export GFMRAG_EMBEDDING_MODEL_NAME=\"custom-embedding-model\"
export GFMRAG_EMBEDDING_BASE_URL=\"http://embedding-service:8001/v1\"
# GFMRAG_EMBEDDING_KEY 未设置，embedding服务使用无认证模式

python -m gfmrag.workflow.stage1_index_dataset
```

## 认证密钥管理

### 独立密钥配置

Chat 和 Embedding 服务支持独立的认证密钥管理：

- **Chat 服务认证**: 优先使用 `GFMRAG_CHAT_KEY`，如果未设置则尝试使用 `OPENAI_API_KEY` 等标准密钥
- **Embedding 服务认证**: 优先使用 `GFMRAG_EMBEDDING_KEY`，如果未设置则尝试使用对应的标准密钥

### 无认证模式

当 API 密钥为空或未设置时，系统将进入无认证模式：

- ✅ 适用于本地服务（Ollama、本地 API 服务器）
- ✅ 适用于不需要认证的第三方服务
- ✅ API 请求中不会包含认证信息

### 认证状态判断规则

```python
# 认证密钥有效性判断
if GFMRAG_CHAT_KEY 存在且非空:
    在Chat API请求中包含认证头
else:
    发送无认证的API请求

if GFMRAG_EMBEDDING_KEY 存在且非空:
    在Embedding API请求中包含认证头
else:
    发送无认证的API请求
```

## 配置验证

### 使用验证脚本

项目提供了一个配置验证脚本来检查环境变量配置：

```bash
python scripts/validate_env_config.py
```

验证内容包括：
- ✅ 环境变量语法检查
- ✅ URL 格式验证
- ✅ 提供商兼容性检查
- ✅ 认证设置验证
- ✅ 配置完整性检查

### 手动验证

也可以在 Python 代码中手动验证配置：

```python
from gfmrag.config_manager import get_config_manager

config_manager = get_config_manager()

# 验证配置
errors = config_manager.validate_config()
if errors:
    print(\"配置错误:\", errors)
else:
    print(\"配置验证通过\")

# 查看环境变量摘要
summary = config_manager.get_environment_variables_summary()
print(\"环境变量设置:\", summary)
```

## 集成到现有代码

### 在 Python 代码中使用

```python
from gfmrag.langchain_factory import create_chat_model, create_embedding_model
from gfmrag.kg_construction.langchain_util import init_langchain_model_from_env

# 方法1: 使用工厂函数（完全依赖环境变量）
chat_model = create_chat_model()
embedding_model = create_embedding_model()

# 方法2: 使用便捷函数
chat_model = init_langchain_model_from_env()

# 方法3: 部分覆盖环境变量
chat_model = create_chat_model(provider=\"openai\", model_name=\"gpt-4\")
```

### 在工作流中使用

环境变量配置会自动被 `stage1_index_dataset` 工作流识别和使用：

```bash
# 设置环境变量
export GFMRAG_CHAT_PROVIDER=\"third-party\"
export GFMRAG_CHAT_BASE_URL=\"http://localhost:8000/v1\"

# 运行工作流（会自动使用环境变量配置）
python -m gfmrag.workflow.stage1_index_dataset
```

## 故障排除

### 常见问题

1. **第三方服务连接失败**
   - 检查 `GFMRAG_CHAT_BASE_URL` 是否正确
   - 确认服务是否运行并可访问
   - 验证端口和协议（http/https）

2. **认证失败**
   - 确认 API 密钥的有效性
   - 检查是否设置了正确的环境变量名
   - 对于本地服务，确认是否需要认证

3. **模型不存在**
   - 验证模型名称是否正确
   - 确认服务提供商支持该模型
   - 检查模型是否已正确加载

### 调试技巧

1. **启用详细日志**
   ```bash
   export GFMRAG_LOGGING_LEVEL=\"DEBUG\"
   ```

2. **使用验证脚本**
   ```bash
   python scripts/validate_env_config.py
   ```

3. **检查配置摘要**
   ```python
   from gfmrag.config_manager import log_environment_config
   log_environment_config()
   ```

## 部署建议

### Docker 环境

在 Docker 容器中使用环境变量：

```dockerfile
# Dockerfile
ENV GFMRAG_CHAT_PROVIDER=third-party
ENV GFMRAG_CHAT_BASE_URL=http://llm-service:8000/v1
ENV GFMRAG_EMBEDDING_PROVIDER=huggingface
```

### CI/CD 流水线

在 CI/CD 中设置环境变量：

```yaml
# GitHub Actions 示例
env:
  GFMRAG_CHAT_PROVIDER: \"openai\"
  GFMRAG_CHAT_KEY: ${{ secrets.OPENAI_API_KEY }}
  GFMRAG_EMBEDDING_PROVIDER: \"openai\"
  GFMRAG_EMBEDDING_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### 生产环境安全

- 🔒 使用密钥管理系统存储 API 密钥
- 🔒 避免在代码中硬编码密钥
- 🔒 定期轮换 API 密钥
- 🔒 监控 API 使用情况

## 性能优化

### 模型缓存

环境变量配置的模型会自动缓存，相同配置的模型实例会被复用：

```python
# 第一次创建会缓存
model1 = create_chat_model()

# 第二次创建会从缓存返回
model2 = create_chat_model()

assert model1 is model2  # True
```

### 连接池

对于高并发场景，建议：
- 使用连接池减少连接开销
- 设置合适的超时时间
- 配置重试机制

## 迁移指南

### 从配置文件迁移

如果你当前使用 YAML 配置文件：

1. **查看现有配置**
   ```yaml
   # config.yaml
   chat:
     provider: openai
     model_name: gpt-3.5-turbo
   ```

2. **转换为环境变量**
   ```bash
   export GFMRAG_CHAT_PROVIDER=\"openai\"
   export GFMRAG_CHAT_MODEL_NAME=\"gpt-3.5-turbo\"
   ```

3. **验证配置**
   ```bash
   python scripts/validate_env_config.py
   ```

### 兼容性说明

- ✅ 现有代码无需修改
- ✅ 配置文件仍然有效
- ✅ 环境变量优先级更高
- ✅ 可以混合使用两种配置方式

## 参考资料

- [配置管理器 API 文档](gfmrag/config_manager.py)
- [模型工厂 API 文档](gfmrag/langchain_factory.py)
- [环境变量验证脚本](scripts/validate_env_config.py)
- [单元测试示例](tests/test_env_variables.py)
- [环境变量示例文件](.env.gfmrag.example)