# LangChain配置优化使用指南

## 概述

GFM-RAG项目的LangChain配置优化提供了统一的配置管理体系，支持多种LLM和Embedding服务提供商，包括OpenAI官方、第三方OpenAI兼容服务（如vLLM、llama-server）等。该优化方案具有以下特性：

- **统一配置管理**：支持环境变量、YAML文件等多种配置方式
- **多服务提供商支持**：OpenAI、NVIDIA、Together、Ollama、第三方服务等
- **增强的错误处理**：重试机制、备用方案、优雅降级
- **配置验证**：多级验证确保配置正确性
- **向后兼容**：保持与现有代码的兼容性

## 快速开始

### 1. 基础配置

首先设置环境变量：

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量
export OPENAI_API_KEY="your-openai-api-key"
export GFMRAG_CHAT_PROVIDER="openai"
export GFMRAG_CHAT_MODEL_NAME="gpt-3.5-turbo"
```

### 2. 使用升级后的LangChain工具

```python
from gfmrag.kg_construction.langchain_util import init_langchain_model

# 创建OpenAI模型（向后兼容）
model = init_langchain_model(
    llm="openai",
    model_name="gpt-3.5-turbo",
    temperature=0.0
)

# 创建第三方兼容模型（新功能）
model = init_langchain_model(
    llm="third-party",
    model_name="llama-2-7b-chat",
    base_url="http://localhost:8000/v1",
    api_key="placeholder"
)
```

### 3. 使用配置管理器

```python
from gfmrag.config_manager import get_config_manager

# 获取配置管理器
config_manager = get_config_manager()

# 获取Chat配置
chat_config = config_manager.get_chat_config()
print(f"Chat提供商: {chat_config.provider}")
print(f"模型名称: {chat_config.model_name}")

# 获取Embedding配置
embedding_config = config_manager.get_embedding_config()
print(f"Embedding提供商: {embedding_config.provider}")
```

## 配置方式

### 1. 环境变量配置（推荐）

环境变量具有最高优先级，适合生产环境：

```bash
# 全局配置
export GFMRAG_DEFAULT_PROVIDER="openai"
export GFMRAG_TIMEOUT=60
export GFMRAG_MAX_RETRIES=3

# Chat服务配置
export GFMRAG_CHAT_PROVIDER="openai"
export GFMRAG_CHAT_MODEL_NAME="gpt-3.5-turbo"
export GFMRAG_CHAT_API_KEY="your-api-key"
export GFMRAG_CHAT_BASE_URL=""  # 可选，用于第三方服务
export GFMRAG_CHAT_TEMPERATURE=0.0

# Embedding服务配置
export GFMRAG_EMBEDDING_PROVIDER="openai"
export GFMRAG_EMBEDDING_MODEL_NAME="text-embedding-ada-002"
export GFMRAG_EMBEDDING_BATCH_SIZE=32
```

### 2. YAML文件配置

创建配置文件`config.yaml`：

```yaml
global:
  default_provider: openai
  timeout: 60
  max_retries: 3
  fallback_enabled: true

chat:
  provider: openai
  model_name: gpt-3.5-turbo
  temperature: 0.0
  max_tokens: 4096

embedding:
  provider: openai
  model_name: text-embedding-ada-002
  batch_size: 32
  normalize: true
```

使用配置文件：

```python
from gfmrag.config_manager import ConfigurationManager

# 加载配置文件
config_manager = ConfigurationManager("config.yaml")
chat_config = config_manager.get_chat_config()
```

### 3. 代码中配置

```python
from gfmrag.config_manager import ChatConfig, EmbeddingConfig
from gfmrag.langchain_factory import LangChainModelFactory

# 直接创建配置
chat_config = ChatConfig(
    provider="openai",
    model_name="gpt-4",
    temperature=0.5,
    api_key="your-api-key"
)

# 使用工厂创建模型
factory = LangChainModelFactory()
model = factory.create_chat_model(chat_config)
```

## 支持的服务提供商

### 1. OpenAI官方服务

```python
# 环境变量配置
export GFMRAG_CHAT_PROVIDER="openai"
export GFMRAG_CHAT_MODEL_NAME="gpt-4"
export OPENAI_API_KEY="your-openai-api-key"

# 或代码配置
chat_config = ChatConfig(
    provider="openai",
    model_name="gpt-4",
    api_key="your-openai-api-key"
)
```

支持的Chat模型：
- `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`
- `gpt-3.5-turbo`
- `o1-preview`, `o1-mini`

支持的Embedding模型：
- `text-embedding-ada-002`
- `text-embedding-3-small`, `text-embedding-3-large`

### 2. 第三方OpenAI兼容服务

#### vLLM服务

```bash
# 启动vLLM服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --host localhost \
    --port 8000

# 配置环境变量
export GFMRAG_CHAT_PROVIDER="third-party"
export GFMRAG_CHAT_BASE_URL="http://localhost:8000/v1"
export GFMRAG_CHAT_MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
export GFMRAG_CHAT_API_KEY="placeholder"
```

#### llama-server（llama.cpp）

```bash
# 启动llama-server
./llama-server \
    -m models/llama-2-7b-chat.gguf \
    --host localhost \
    --port 8080

# 配置环境变量
export GFMRAG_CHAT_PROVIDER="third-party"
export GFMRAG_CHAT_BASE_URL="http://localhost:8080/v1"
export GFMRAG_CHAT_MODEL_NAME="llama-2-7b-chat"
export GFMRAG_CHAT_API_KEY="placeholder"
```

### 3. NVIDIA AI服务

```bash
export GFMRAG_CHAT_PROVIDER="nvidia"
export GFMRAG_CHAT_MODEL_NAME="meta/llama3-70b-instruct"
export NVIDIA_API_KEY="your-nvidia-api-key"
```

### 4. Together AI服务

```bash
export GFMRAG_CHAT_PROVIDER="together"
export GFMRAG_CHAT_MODEL_NAME="meta-llama/Llama-2-70b-chat-hf"
export TOGETHER_API_KEY="your-together-api-key"
```

### 5. Ollama本地服务

```bash
# 启动Ollama
ollama serve

# 拉取模型
ollama pull llama3

# 配置环境变量
export GFMRAG_CHAT_PROVIDER="ollama"
export GFMRAG_CHAT_MODEL_NAME="llama3"
export GFMRAG_CHAT_BASE_URL="http://localhost:11434"
```

## 高级功能

### 1. 错误处理和重试

```python
from gfmrag.error_handler import with_error_handling, RetryConfig

# 自定义重试配置
retry_config = RetryConfig(
    max_retries=5,
    base_delay=2.0,
    max_delay=60.0,
    backoff_factor=2.0
)

@with_error_handling(retry_config=retry_config)
def create_model_with_retry():
    return init_langchain_model("openai", "gpt-3.5-turbo")

model = create_model_with_retry()
```

### 2. 备用方案

```python
from gfmrag.error_handler import FallbackManager

# 注册备用配置
fallback_manager = FallbackManager()
fallback_config = ChatConfig(
    provider="together",
    model_name="meta-llama/Llama-2-7b-chat-hf"
)
fallback_manager.register_fallback_config("chat", fallback_config)
```

### 3. 配置验证

```python
from gfmrag.config_validator import ConfigValidator

validator = ConfigValidator()

# 验证Chat配置
chat_config = config_manager.get_chat_config()
result = validator.validate_chat_config(chat_config)

if not result.is_valid:
    print("配置验证失败:")
    for error in result.errors:
        print(f"  - {error}")
```

### 4. 性能监控

```python
from gfmrag.embedding_factory import EmbeddingModelFactory

factory = EmbeddingModelFactory()

# 性能基准测试
benchmark_result = factory.benchmark_model(
    test_texts=["测试文本1", "测试文本2", "测试文本3"],
    num_runs=5
)

print(f"平均处理时间: {benchmark_result['avg_time']:.3f}秒")
print(f"吞吐量: {benchmark_result['throughput']:.2f} 文本/秒")
```

## 迁移指南

### 从旧版本迁移

如果您使用的是旧版本的`langchain_util.py`，迁移步骤如下：

1. **保持现有代码不变**（向后兼容）：
   ```python
   # 旧代码继续工作
   model = init_langchain_model("openai", "gpt-3.5-turbo")
   ```

2. **逐步采用新功能**：
   ```python
   # 使用新的base_url参数
   model = init_langchain_model(
       llm="third-party",
       model_name="llama-2-7b-chat",
       base_url="http://localhost:8000/v1"
   )
   ```

3. **使用配置管理器**：
   ```python
   # 新的推荐方式
   from gfmrag.config_manager import get_config_manager
   from gfmrag.langchain_factory import LangChainModelFactory
   
   config_manager = get_config_manager()
   factory = LangChainModelFactory()
   
   chat_config = config_manager.get_chat_config()
   model = factory.create_chat_model(chat_config)
   ```

### 配置文件迁移

1. **创建新的配置文件**：
   ```bash
   # 使用提供的模板
   cp configs/langchain_config.yaml my_config.yaml
   ```

2. **设置环境变量**：
   ```bash
   # 使用更新后的环境变量
   cp .env.example .env
   # 编辑 .env 文件
   ```

## 故障排除

### 常见问题

1. **第三方服务连接失败**
   ```python
   # 检查服务是否运行
   import requests
   response = requests.get("http://localhost:8000/v1/models")
   print(response.status_code)
   
   # 验证配置
   from gfmrag.config_validator import ConfigValidator
   validator = ConfigValidator()
   result = validator.validate_connection(chat_config)
   ```

2. **API密钥无效**
   ```python
   # 验证API权限
   result = validator.validate_permissions(chat_config)
   if not result.is_valid:
       print("API权限验证失败:", result.errors)
   ```

3. **模型不存在**
   ```python
   # 获取可用模型列表
   from gfmrag.providers.third_party_provider import ThirdPartyProvider
   provider = ThirdPartyProvider()
   models = provider.get_available_models(chat_config)
   print("可用模型:", models)
   ```

### 日志诊断

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或通过环境变量
export GFMRAG_LOGGING_LEVEL=DEBUG
```

### 配置验证

```python
from gfmrag.config_validator import ConfigValidator

validator = ConfigValidator()
results = validator.comprehensive_validation(
    global_config=config_manager.global_config,
    chat_config=config_manager.get_chat_config(),
    embedding_config=config_manager.get_embedding_config(),
    test_connections=True,
    test_permissions=True
)

# 生成详细报告
report = validator.generate_validation_report(results)
print(report)
```

## 最佳实践

### 1. 生产环境配置

- 使用环境变量管理敏感信息（API密钥）
- 启用备用方案和重试机制
- 配置性能监控和日志
- 定期验证配置和连接

### 2. 开发环境配置

- 使用配置文件管理非敏感配置
- 启用详细日志便于调试
- 使用本地服务（Ollama、vLLM）减少API调用

### 3. 安全建议

- 不要在代码中硬编码API密钥
- 使用HTTPS URL（避免HTTP）
- 定期轮换API密钥
- 限制API密钥权限

### 4. 性能优化

- 合理设置批处理大小
- 启用模型缓存
- 使用连接池
- 监控API使用量

## 参考资料

- [LangChain官方文档](https://python.langchain.com/)
- [OpenAI API文档](https://platform.openai.com/docs/)
- [vLLM部署指南](https://docs.vllm.ai/)
- [llama.cpp使用指南](https://github.com/ggerganov/llama.cpp)

## 支持

如果遇到问题或需要帮助，请：

1. 检查本文档的故障排除部分
2. 查看项目的Issue页面
3. 运行配置验证工具进行诊断
4. 提供详细的错误日志和配置信息