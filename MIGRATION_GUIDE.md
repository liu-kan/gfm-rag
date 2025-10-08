# LangChain配置优化迁移指南

## 概述

本指南帮助您从旧版本的LangChain集成迁移到新的优化版本。新版本提供了更强大的配置管理、多服务提供商支持和错误处理机制，同时保持向后兼容性。

## 版本对比

### 旧版本特性
- 基础的LangChain模型初始化
- 有限的服务提供商支持
- 简单的环境变量配置
- 基础错误处理

### 新版本特性
- ✅ 统一配置管理系统
- ✅ 多种服务提供商支持（OpenAI、vLLM、llama-server等）
- ✅ 增强的错误处理和重试机制
- ✅ 配置验证和监控
- ✅ 向后兼容性保证
- ✅ 性能优化和缓存

## 迁移步骤

### 第一阶段：评估现有配置

1. **检查当前使用情况**
   ```bash
   # 查找所有使用langchain_util的地方
   grep -r "import.*langchain_util" .
   grep -r "from.*langchain_util" .
   ```

2. **记录当前配置**
   ```bash
   # 列出相关环境变量
   env | grep -E "(OPENAI|TOGETHER|NVIDIA)"
   ```

### 第二阶段：更新环境变量

1. **备份现有环境变量**
   ```bash
   # 备份当前环境变量
   env > backup_env_vars.txt
   ```

2. **更新.env文件**
   ```bash
   # 使用新的环境变量模板
   cp .env.example .env.new
   
   # 对比差异
   diff .env .env.new
   ```

3. **迁移环境变量映射**

   | 旧环境变量 | 新环境变量 | 说明 |
   |------------|------------|------|
   | `OPENAI_API_KEY` | `OPENAI_API_KEY` | 保持不变 |
   | - | `GFMRAG_CHAT_PROVIDER` | 新增：指定Chat服务提供商 |
   | - | `GFMRAG_CHAT_MODEL_NAME` | 新增：指定Chat模型名称 |
   | - | `GFMRAG_CHAT_BASE_URL` | 新增：支持第三方服务URL |
   | - | `GFMRAG_EMBEDDING_PROVIDER` | 新增：指定Embedding服务提供商 |

   **迁移示例**：
   ```bash
   # 旧配置
   export OPENAI_API_KEY="sk-..."
   
   # 新配置（向后兼容 + 新功能）
   export OPENAI_API_KEY="sk-..."
   export GFMRAG_CHAT_PROVIDER="openai"
   export GFMRAG_CHAT_MODEL_NAME="gpt-3.5-turbo"
   export GFMRAG_EMBEDDING_PROVIDER="openai"
   export GFMRAG_EMBEDDING_MODEL_NAME="text-embedding-ada-002"
   ```

### 第三阶段：代码迁移

#### 3.1 保持现有代码不变（推荐）

**现有代码继续工作，无需修改**：

```python
# 这些代码在新版本中仍然正常工作
from gfmrag.kg_construction.langchain_util import init_langchain_model

# OpenAI模型
model = init_langchain_model("openai", "gpt-3.5-turbo")

# Together模型
model = init_langchain_model("together", "meta-llama/Llama-2-7b-chat-hf")

# Ollama模型
model = init_langchain_model("ollama", "llama3")
```

#### 3.2 逐步采用新功能（可选）

**添加第三方服务支持**：

```python
# 支持vLLM服务
model = init_langchain_model(
    llm="third-party",
    model_name="llama-2-7b-chat",
    base_url="http://localhost:8000/v1",
    api_key="placeholder"
)

# 支持llama-server
model = init_langchain_model(
    llm="third-party", 
    model_name="llama-2-7b-chat",
    base_url="http://localhost:8080/v1",
    api_key="placeholder"
)
```

#### 3.3 使用新的配置管理（高级）

**完全采用新架构**：

```python
# 新方式：使用配置管理器
from gfmrag.config_manager import get_config_manager
from gfmrag.langchain_factory import LangChainModelFactory

# 获取配置
config_manager = get_config_manager()
chat_config = config_manager.get_chat_config()

# 创建模型
factory = LangChainModelFactory()
model = factory.create_chat_model(chat_config)
```

### 第四阶段：配置文件创建（可选）

1. **创建配置文件**
   ```bash
   # 复制配置模板
   cp configs/langchain_config.yaml my_config.yaml
   ```

2. **自定义配置**
   ```yaml
   # my_config.yaml
   global:
     default_provider: openai
     timeout: 60
     
   chat:
     provider: openai
     model_name: gpt-3.5-turbo
     temperature: 0.0
     
   embedding:
     provider: openai
     model_name: text-embedding-ada-002
   ```

3. **使用配置文件**
   ```python
   from gfmrag.config_manager import ConfigurationManager
   
   # 加载自定义配置
   config_manager = ConfigurationManager("my_config.yaml")
   ```

### 第五阶段：验证和测试

1. **运行配置验证**
   ```python
   from gfmrag.config_validator import ConfigValidator
   
   validator = ConfigValidator()
   config_manager = get_config_manager()
   
   # 验证配置
   results = validator.comprehensive_validation(
       chat_config=config_manager.get_chat_config(),
       embedding_config=config_manager.get_embedding_config(),
       test_connections=True
   )
   
   # 查看验证结果
   report = validator.generate_validation_report(results)
   print(report)
   ```

2. **运行现有测试**
   ```bash
   # 确保现有功能正常
   python -m pytest tests/ -v
   
   # 运行新的集成测试
   python -m pytest tests/test_langchain_integration.py -v
   ```

## 特定场景迁移

### 场景1：当前使用OpenAI官方服务

**迁移前**：
```python
model = init_langchain_model("openai", "gpt-3.5-turbo")
```

**迁移后**：
```python
# 方式1：保持不变（推荐）
model = init_langchain_model("openai", "gpt-3.5-turbo")

# 方式2：使用环境变量配置
export GFMRAG_CHAT_PROVIDER="openai"
export GFMRAG_CHAT_MODEL_NAME="gpt-3.5-turbo"
# 代码简化为：
config_manager = get_config_manager()
chat_config = config_manager.get_chat_config()
model = LangChainModelFactory().create_chat_model(chat_config)
```

### 场景2：需要添加vLLM支持

**新增配置**：
```bash
# 环境变量配置
export GFMRAG_CHAT_PROVIDER="third-party"
export GFMRAG_CHAT_BASE_URL="http://localhost:8000/v1"
export GFMRAG_CHAT_MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
export GFMRAG_CHAT_API_KEY="placeholder"
```

**代码调用**：
```python
# 直接调用
model = init_langchain_model(
    llm="third-party",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    base_url="http://localhost:8000/v1"
)

# 或使用配置管理器
config_manager = get_config_manager()
model = LangChainModelFactory().create_chat_model()
```

### 场景3：需要错误处理和重试

**添加错误处理**：
```python
from gfmrag.error_handler import with_error_handling, RetryConfig

# 配置重试策略
retry_config = RetryConfig(
    max_retries=5,
    base_delay=2.0,
    max_delay=60.0
)

@with_error_handling(retry_config=retry_config, enable_fallback=True)
def create_model():
    return init_langchain_model("openai", "gpt-3.5-turbo")

# 调用时自动处理错误和重试
model = create_model()
```

### 场景4：多环境配置管理

**开发环境**：
```yaml
# dev_config.yaml
global:
  default_provider: ollama
  logging_level: DEBUG
  
chat:
  provider: ollama
  model_name: llama3
  base_url: http://localhost:11434
```

**生产环境**：
```yaml
# prod_config.yaml  
global:
  default_provider: openai
  logging_level: WARNING
  fallback_enabled: true
  
chat:
  provider: openai
  model_name: gpt-4o-mini
```

**代码**：
```python
import os
from gfmrag.config_manager import ConfigurationManager

# 根据环境选择配置
env = os.getenv("ENV", "dev")
config_file = f"{env}_config.yaml"
config_manager = ConfigurationManager(config_file)
```

## 常见迁移问题

### 问题1：现有代码不工作

**症状**：导入错误或模型创建失败

**解决方案**：
```python
# 检查导入路径
from gfmrag.kg_construction.langchain_util import init_langchain_model

# 确保环境变量正确设置
import os
print("OPENAI_API_KEY:", "设置" if os.getenv("OPENAI_API_KEY") else "未设置")
```

### 问题2：第三方服务连接失败

**症状**：ConnectionError或超时错误

**解决方案**：
```python
# 1. 检查服务是否运行
import requests
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    print("服务状态:", response.status_code)
except Exception as e:
    print("服务连接失败:", e)

# 2. 验证配置
from gfmrag.config_validator import ConfigValidator
validator = ConfigValidator()
result = validator.validate_connection(chat_config)
if not result.is_valid:
    print("连接验证失败:", result.errors)
```

### 问题3：API密钥配置混乱

**症状**：认证错误或权限不足

**解决方案**：
```python
# 清理环境变量
unset OPENAI_API_KEY
unset GFMRAG_CHAT_API_KEY

# 重新设置
export OPENAI_API_KEY="your-actual-key"

# 验证设置
from gfmrag.config_manager import get_config_manager
config = get_config_manager().get_chat_config()
print("API Key:", "已设置" if config.api_key else "未设置")
```

## 回滚计划

如果迁移过程中出现问题，可以快速回滚：

### 1. 代码回滚
```bash
# 恢复旧版本文件（如果有备份）
git checkout HEAD~1 -- gfmrag/kg_construction/langchain_util.py
```

### 2. 环境变量回滚
```bash
# 恢复备份的环境变量
source backup_env_vars.txt
```

### 3. 依赖回滚
```bash
# 如果需要，回滚依赖版本
pip install langchain==<old_version>
```

## 迁移检查清单

- [ ] 备份现有配置和代码
- [ ] 更新环境变量文件
- [ ] 测试现有代码在新版本下的兼容性
- [ ] 验证新功能（如第三方服务支持）
- [ ] 运行全部测试套件
- [ ] 配置监控和日志
- [ ] 准备回滚计划
- [ ] 团队培训和文档更新

## 迁移时间表建议

| 阶段 | 时间 | 活动 |
|------|------|------|
| 第1周 | 评估阶段 | 分析现有代码，制定迁移计划 |
| 第2周 | 准备阶段 | 更新环境变量，创建配置文件 |
| 第3周 | 测试阶段 | 在开发环境测试新功能 |
| 第4周 | 部署阶段 | 逐步部署到生产环境 |

## 后续优化建议

迁移完成后，可以考虑以下优化：

1. **启用性能监控**
   ```python
   from gfmrag.embedding_factory import EmbeddingModelFactory
   factory = EmbeddingModelFactory()
   benchmark = factory.benchmark_model()
   ```

2. **配置备用方案**
   ```python
   from gfmrag.error_handler import FallbackManager
   fallback_manager = FallbackManager()
   # 注册备用配置
   ```

3. **定期配置验证**
   ```python
   # 设置定期任务验证配置和连接
   from gfmrag.config_validator import ConfigValidator
   validator = ConfigValidator()
   # 运行健康检查
   ```

## 获取帮助

如果在迁移过程中遇到问题：

1. 查看详细的[使用指南](docs/langchain_optimization_guide.md)
2. 运行诊断工具检查配置
3. 查看项目Issue页面寻找类似问题
4. 提供详细的错误日志和配置信息寻求帮助

## 总结

新版本的LangChain配置优化提供了更强大和灵活的功能，同时保持了向后兼容性。按照本迁移指南，您可以：

- **零风险**：现有代码继续工作
- **渐进式**：逐步采用新功能
- **可回滚**：出现问题可快速恢复
- **高收益**：获得更好的性能和稳定性

建议采用渐进式迁移策略，先确保现有功能正常，再逐步启用新特性。