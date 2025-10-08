"""
GFMRAG环境变量功能单元测试

该测试模块验证环境变量功能的正确性，包括：
- 环境变量解析和优先级
- 配置管理器环境变量支持
- 模型工厂环境变量集成
- 无认证模式支持
- 第三方服务配置
"""

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

from gfmrag.config_manager import (
    ConfigurationManager, 
    ChatConfig, 
    EmbeddingConfig,
    reset_config_manager,
    create_config_from_env
)
from gfmrag.langchain_factory import LangChainModelFactory


class TestEnvironmentVariableSupport:
    """测试环境变量支持功能"""
    
    def setup_method(self):
        """每个测试前的设置"""
        # 重置全局配置管理器
        reset_config_manager()
        
        # 清理环境变量
        self.original_env = {}
        for key in list(os.environ.keys()):
            if key.startswith('GFMRAG_'):
                self.original_env[key] = os.environ.pop(key)
    
    def teardown_method(self):
        """每个测试后的清理"""
        # 恢复原始环境变量
        for key in list(os.environ.keys()):
            if key.startswith('GFMRAG_'):
                del os.environ[key]
        
        for key, value in self.original_env.items():
            os.environ[key] = value
        
        # 重置配置管理器
        reset_config_manager()
    
    def test_basic_environment_variable_parsing(self):
        """测试基本环境变量解析"""
        # 设置环境变量
        os.environ['GFMRAG_CHAT_PROVIDER'] = 'openai'
        os.environ['GFMRAG_CHAT_MODEL_NAME'] = 'gpt-4'
        os.environ['GFMRAG_CHAT_KEY'] = 'sk-test-key'
        
        config_manager = ConfigurationManager()
        chat_config = config_manager.get_chat_config()
        
        assert chat_config.provider == 'openai'
        assert chat_config.model_name == 'gpt-4'
        assert chat_config.api_key == 'sk-test-key'
    
    def test_environment_variable_priority(self):
        """测试环境变量优先级高于配置文件"""
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
chat:
  provider: together
  model_name: llama-2-7b
  api_key: file-key
""")
            config_file = f.name
        
        try:
            # 设置环境变量（应该覆盖配置文件）
            os.environ['GFMRAG_CHAT_PROVIDER'] = 'openai'
            os.environ['GFMRAG_CHAT_MODEL_NAME'] = 'gpt-3.5-turbo'
            os.environ['GFMRAG_CHAT_KEY'] = 'env-key'
            
            config_manager = ConfigurationManager(config_file)
            chat_config = config_manager.get_chat_config()
            
            # 环境变量应该优先
            assert chat_config.provider == 'openai'
            assert chat_config.model_name == 'gpt-3.5-turbo'
            assert chat_config.api_key == 'env-key'
            
        finally:
            os.unlink(config_file)
    
    def test_no_authentication_mode(self):
        """测试无认证模式"""
        # 设置第三方服务但不设置API密钥
        os.environ['GFMRAG_CHAT_PROVIDER'] = 'third-party'
        os.environ['GFMRAG_CHAT_MODEL_NAME'] = 'llama-2-7b'
        os.environ['GFMRAG_CHAT_BASE_URL'] = 'http://localhost:8000/v1'
        # 注意：GFMRAG_CHAT_KEY 未设置
        
        config_manager = ConfigurationManager()
        chat_config = config_manager.get_chat_config()
        
        assert chat_config.provider == 'third-party'
        assert chat_config.model_name == 'llama-2-7b'
        assert chat_config.base_url == 'http://localhost:8000/v1'
        assert chat_config.api_key is None  # 应该是None，表示无认证
    
    def test_empty_string_api_key_handling(self):
        """测试空字符串API密钥处理"""
        # 设置空字符串API密钥（应该被解释为无认证）
        os.environ['GFMRAG_CHAT_PROVIDER'] = 'ollama'
        os.environ['GFMRAG_CHAT_MODEL_NAME'] = 'llama3'
        os.environ['GFMRAG_CHAT_KEY'] = ''  # 空字符串
        
        config_manager = ConfigurationManager()
        chat_config = config_manager.get_chat_config()
        
        assert chat_config.provider == 'ollama'
        assert chat_config.model_name == 'llama3'
        assert chat_config.api_key is None  # 空字符串应该转换为None
    
    def test_separate_chat_and_embedding_keys(self):
        """测试Chat和Embedding独立API密钥"""
        # 设置不同的API密钥
        os.environ['GFMRAG_CHAT_PROVIDER'] = 'openai'
        os.environ['GFMRAG_CHAT_KEY'] = 'chat-key'
        os.environ['GFMRAG_EMBEDDING_PROVIDER'] = 'openai'
        os.environ['GFMRAG_EMBEDDING_KEY'] = 'embedding-key'
        
        config_manager = ConfigurationManager()
        chat_config = config_manager.get_chat_config()
        embedding_config = config_manager.get_embedding_config()
        
        assert chat_config.api_key == 'chat-key'
        assert embedding_config.api_key == 'embedding-key'
    
    def test_mixed_authentication_scenario(self):
        """测试混合认证场景"""
        # Chat需要认证，Embedding不需要认证
        os.environ['GFMRAG_CHAT_PROVIDER'] = 'openai'
        os.environ['GFMRAG_CHAT_KEY'] = 'openai-key'
        os.environ['GFMRAG_EMBEDDING_PROVIDER'] = 'third-party'
        os.environ['GFMRAG_EMBEDDING_BASE_URL'] = 'http://localhost:8001/v1'
        # GFMRAG_EMBEDDING_KEY 未设置，表示无认证
        
        config_manager = ConfigurationManager()
        chat_config = config_manager.get_chat_config()
        embedding_config = config_manager.get_embedding_config()
        
        assert chat_config.provider == 'openai'
        assert chat_config.api_key == 'openai-key'
        assert embedding_config.provider == 'third-party'
        assert embedding_config.base_url == 'http://localhost:8001/v1'
        assert embedding_config.api_key is None
    
    def test_environment_variable_type_conversion(self):
        """测试环境变量类型转换"""
        # 设置各种类型的环境变量
        os.environ['GFMRAG_CHAT_PROVIDER'] = 'openai'
        os.environ['GFMRAG_CHAT_TEMPERATURE'] = '0.7'  # 字符串形式的浮点数
        os.environ['GFMRAG_CHAT_MAX_RETRIES'] = '5'     # 字符串形式的整数
        os.environ['GFMRAG_EMBEDDING_BATCH_SIZE'] = '64'
        os.environ['GFMRAG_EMBEDDING_NORMALIZE'] = 'true'  # 字符串形式的布尔值
        
        config_manager = ConfigurationManager()
        chat_config = config_manager.get_chat_config()
        embedding_config = config_manager.get_embedding_config()
        
        assert chat_config.temperature == 0.7
        assert chat_config.max_retries == 5
        assert embedding_config.batch_size == 64
        assert embedding_config.normalize is True
    
    @patch('gfmrag.langchain_factory.ChatOpenAI')
    def test_model_factory_environment_integration(self, mock_chat_openai):
        """测试模型工厂环境变量集成"""
        # 设置环境变量
        os.environ['GFMRAG_CHAT_PROVIDER'] = 'third-party'
        os.environ['GFMRAG_CHAT_MODEL_NAME'] = 'custom-model'
        os.environ['GFMRAG_CHAT_BASE_URL'] = 'http://localhost:8000/v1'
        # 无API密钥，测试无认证模式
        
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model
        
        factory = LangChainModelFactory()
        model = factory.create_chat_model()
        
        # 验证模型创建时使用了正确的参数
        mock_chat_openai.assert_called_once()
        call_args = mock_chat_openai.call_args[1]
        
        assert call_args['model'] == 'custom-model'
        assert call_args['base_url'] == 'http://localhost:8000/v1'
        assert call_args['api_key'] == 'not-needed'  # 无认证模式的占位符
    
    def test_configuration_validation_with_env_vars(self):
        """测试配置验证与环境变量"""
        # 设置有效的环境变量配置
        os.environ['GFMRAG_CHAT_PROVIDER'] = 'openai'
        os.environ['GFMRAG_CHAT_MODEL_NAME'] = 'gpt-3.5-turbo'
        os.environ['GFMRAG_EMBEDDING_PROVIDER'] = 'openai'
        os.environ['GFMRAG_EMBEDDING_MODEL_NAME'] = 'text-embedding-ada-002'
        
        config_manager = ConfigurationManager()
        errors = config_manager.validate_config()
        
        # 应该没有验证错误
        assert len(errors) == 0
    
    def test_invalid_url_validation(self):
        """测试无效URL验证"""
        # 设置无效的URL
        os.environ['GFMRAG_CHAT_PROVIDER'] = 'third-party'
        os.environ['GFMRAG_CHAT_BASE_URL'] = 'invalid-url'
        
        config_manager = ConfigurationManager()
        errors = config_manager.validate_config()
        
        # 应该有URL格式错误
        assert len(errors) > 0
        assert any('URL格式无效' in str(error_list) for error_list in errors.values())
    
    def test_environment_variables_summary(self):
        """测试环境变量摘要功能"""
        # 设置一些环境变量
        os.environ['GFMRAG_CHAT_PROVIDER'] = 'openai'
        os.environ['GFMRAG_CHAT_KEY'] = 'secret-key'
        os.environ['GFMRAG_EMBEDDING_PROVIDER'] = 'huggingface'
        
        config_manager = ConfigurationManager()
        summary = config_manager.get_environment_variables_summary()
        
        # 验证摘要内容
        assert 'GFMRAG_CHAT_PROVIDER' in summary
        assert summary['GFMRAG_CHAT_PROVIDER'] == 'openai'
        
        # API密钥应该被隐藏
        assert 'GFMRAG_CHAT_KEY' in summary
        assert summary['GFMRAG_CHAT_KEY'] == '<已设置>'
        
        assert 'GFMRAG_EMBEDDING_PROVIDER' in summary
        assert summary['GFMRAG_EMBEDDING_PROVIDER'] == 'huggingface'
    
    def test_create_config_from_env_utility(self):
        """测试从环境变量创建配置的工具函数"""
        # 设置环境变量
        os.environ['GFMRAG_CHAT_PROVIDER'] = 'ollama'
        os.environ['GFMRAG_CHAT_MODEL_NAME'] = 'llama3'
        
        config_manager = create_config_from_env()
        chat_config = config_manager.get_chat_config()
        
        assert chat_config.provider == 'ollama'
        assert chat_config.model_name == 'llama3'
    
    def test_third_party_service_configuration(self):
        """测试第三方服务完整配置"""
        # 模拟vLLM服务配置
        os.environ['GFMRAG_CHAT_PROVIDER'] = 'third-party'
        os.environ['GFMRAG_CHAT_MODEL_NAME'] = 'meta-llama/Llama-2-7b-chat-hf'
        os.environ['GFMRAG_CHAT_BASE_URL'] = 'http://vllm-server:8000/v1'
        # 不设置API密钥，模拟本地服务
        
        # 模拟本地embedding服务
        os.environ['GFMRAG_EMBEDDING_PROVIDER'] = 'third-party'
        os.environ['GFMRAG_EMBEDDING_MODEL_NAME'] = 'sentence-transformers/all-MiniLM-L6-v2'
        os.environ['GFMRAG_EMBEDDING_BASE_URL'] = 'http://embedding-server:8001/v1'
        
        config_manager = ConfigurationManager()
        chat_config = config_manager.get_chat_config()
        embedding_config = config_manager.get_embedding_config()
        
        # 验证Chat配置
        assert chat_config.provider == 'third-party'
        assert chat_config.model_name == 'meta-llama/Llama-2-7b-chat-hf'
        assert chat_config.base_url == 'http://vllm-server:8000/v1'
        assert chat_config.api_key is None
        
        # 验证Embedding配置
        assert embedding_config.provider == 'third-party'
        assert embedding_config.model_name == 'sentence-transformers/all-MiniLM-L6-v2'
        assert embedding_config.base_url == 'http://embedding-server:8001/v1'
        assert embedding_config.api_key is None
    
    def test_fallback_to_traditional_api_keys(self):
        """测试回退到传统API密钥"""
        # 设置传统API密钥但不设置GFMRAG特定密钥
        os.environ['OPENAI_API_KEY'] = 'traditional-openai-key'
        os.environ['GFMRAG_CHAT_PROVIDER'] = 'openai'
        # 不设置 GFMRAG_CHAT_KEY
        
        config_manager = ConfigurationManager()
        chat_config = config_manager.get_chat_config()
        
        # 应该回退到传统API密钥
        assert chat_config.api_key == 'traditional-openai-key'
    
    def test_gfmrag_key_priority_over_traditional(self):
        """测试GFMRAG密钥优先于传统密钥"""
        # 同时设置两种密钥
        os.environ['OPENAI_API_KEY'] = 'traditional-key'
        os.environ['GFMRAG_CHAT_KEY'] = 'gfmrag-specific-key'
        os.environ['GFMRAG_CHAT_PROVIDER'] = 'openai'
        
        config_manager = ConfigurationManager()
        chat_config = config_manager.get_chat_config()
        
        # GFMRAG特定密钥应该优先
        assert chat_config.api_key == 'gfmrag-specific-key'


if __name__ == "__main__":
    pytest.main([__file__])