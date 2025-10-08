# 环境变量配置快速开始

本文档提供 GFM-RAG 环境变量配置的快速入门指南。

## 5分钟快速配置

### 步骤 1: 选择您的配置场景

#### 场景 A: OpenAI 官方服务
```bash
export GFMRAG_CHAT_PROVIDER=\"openai\"
export GFMRAG_CHAT_MODEL_NAME=\"gpt-3.5-turbo\"
export GFMRAG_CHAT_KEY=\"sk-your-openai-key\"
export GFMRAG_EMBEDDING_PROVIDER=\"openai\"
export GFMRAG_EMBEDDING_KEY=\"sk-your-openai-key\"
```

#### 场景 B: 本地 Ollama + OpenAI Embedding
```bash
export GFMRAG_CHAT_PROVIDER=\"ollama\"
export GFMRAG_CHAT_MODEL_NAME=\"llama3\"
# 无需设置 GFMRAG_CHAT_KEY，Ollama 无需认证
export GFMRAG_EMBEDDING_PROVIDER=\"openai\"
export GFMRAG_EMBEDDING_KEY=\"sk-your-openai-key\"
```

#### 场景 C: 第三方 vLLM 服务（无认证）
```bash
export GFMRAG_CHAT_PROVIDER=\"third-party\"
export GFMRAG_CHAT_MODEL_NAME=\"llama-2-7b-chat\"
export GFMRAG_CHAT_BASE_URL=\"http://localhost:8000/v1\"
# 无需设置 GFMRAG_CHAT_KEY，本地服务无认证
export GFMRAG_EMBEDDING_PROVIDER=\"huggingface\"
export GFMRAG_EMBEDDING_MODEL_NAME=\"sentence-transformers/all-MiniLM-L6-v2\"
```

### 步骤 2: 验证配置

```bash
python scripts/validate_env_config.py
```

应该看到类似输出：
```
✅ 所有检查通过！配置验证成功。
```

### 步骤 3: 运行工作流

```bash
python -m gfmrag.workflow.stage1_index_dataset
```

## 常用环境变量

| 变量 | 用途 | 示例 |
|------|------|------|
| `GFMRAG_CHAT_PROVIDER` | Chat 服务商 | `openai`, `ollama`, `third-party` |
| `GFMRAG_CHAT_MODEL_NAME` | Chat 模型 | `gpt-3.5-turbo`, `llama3` |
| `GFMRAG_CHAT_BASE_URL` | 第三方 API 地址 | `http://localhost:8000/v1` |
| `GFMRAG_CHAT_KEY` | Chat API 密钥 | `sk-xxx` (留空=无认证) |
| `GFMRAG_EMBEDDING_PROVIDER` | Embedding 服务商 | `openai`, `huggingface` |
| `GFMRAG_EMBEDDING_KEY` | Embedding API 密钥 | `sk-xxx` (留空=无认证) |

## 故障排除

### 问题：第三方服务连接失败
**解决方案：**
1. 检查服务是否运行：`curl http://localhost:8000/v1/models`
2. 确认 URL 格式正确
3. 验证防火墙设置

### 问题：认证失败
**解决方案：**
1. 确认 API 密钥有效
2. 检查环境变量名拼写
3. 对于本地服务，确认是否需要认证

### 问题：模型不存在
**解决方案：**
1. 验证模型名称
2. 确认服务支持该模型
3. 检查模型是否已加载

## 高级配置

### 使用 .env 文件

创建 `.env` 文件：
```bash
# 复制示例文件
cp .env.gfmrag.example .env

# 编辑配置
nano .env
```

### 在代码中使用

```python
from gfmrag.langchain_factory import create_chat_model, create_embedding_model

# 自动使用环境变量
chat_model = create_chat_model()
embedding_model = create_embedding_model()
```

## 需要帮助？

- 📖 详细文档：[环境变量配置指南](ENVIRONMENT_VARIABLES_GUIDE.md)
- 🔧 配置验证：`python scripts/validate_env_config.py`
- 📝 示例配置：[.env.gfmrag.example](.env.gfmrag.example)
- 🧪 测试示例：[tests/test_env_variables.py](tests/test_env_variables.py)