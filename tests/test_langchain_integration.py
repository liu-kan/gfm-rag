"""
LangChain配置优化集成测试

该测试套件验证LangChain配置优化的各个组件是否正常工作。
"""

import os
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from gfmrag.config_manager import ConfigurationManager, ChatConfig, EmbeddingConfig, GlobalConfig
from gfmrag.langchain_factory import LangChainModelFactory
from gfmrag.embedding_factory import EmbeddingModelFactory
from gfmrag.config_validator import ConfigValidator, ValidationResult
from gfmrag.error_handler import RetryHandler, FallbackManager, ErrorClassifier


class TestConfigurationManager:
    """测试配置管理器"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
        
        # 创建测试配置文件
        test_config = {
            "global": {
                "default_provider": "openai",
                "timeout": 30,
                "max_retries": 2,
                "fallback_enabled": True,
            },
            "chat": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.5,
                "max_tokens": 1000,
            },
            "embedding": {
                "provider": "openai",
                "model_name": "text-embedding-ada-002",
                "batch_size": 16,
                "normalize": True,
            },
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(test_config, f)
    
    def test_load_config_file(self):
        """测试配置文件加载"""
        config_manager = ConfigurationManager(self.config_file)
        
        # 验证配置加载
        assert config_manager._config_cache["global"]["default_provider"] == "openai"
        assert config_manager._config_cache["chat"]["model_name"] == "gpt-3.5-turbo"
        assert config_manager._config_cache["embedding"]["batch_size"] == 16
    
    def test_get_chat_config(self):
        """测试获取Chat配置"""
        config_manager = ConfigurationManager(self.config_file)
        chat_config = config_manager.get_chat_config()
        
        assert isinstance(chat_config, ChatConfig)
        assert chat_config.provider == "openai"
        assert chat_config.model_name == "gpt-3.5-turbo"
        assert chat_config.temperature == 0.5
        assert chat_config.max_tokens == 1000
    
    def test_get_embedding_config(self):
        """测试获取Embedding配置"""
        config_manager = ConfigurationManager(self.config_file)
        embedding_config = config_manager.get_embedding_config()
        
        assert isinstance(embedding_config, EmbeddingConfig)
        assert embedding_config.provider == "openai"
        assert embedding_config.model_name == "text-embedding-ada-002"
        assert embedding_config.batch_size == 16
        assert embedding_config.normalize is True
    
    @patch.dict(os.environ, {
        "GFMRAG_CHAT_MODEL_NAME": "gpt-4",
        "GFMRAG_CHAT_TEMPERATURE": "0.7",
    })
    def test_env_var_override(self):
        """测试环境变量覆盖"""
        config_manager = ConfigurationManager(self.config_file)
        chat_config = config_manager.get_chat_config()
        
        # 环境变量应该覆盖配置文件
        assert chat_config.model_name == "gpt-4"
        assert chat_config.temperature == 0.7
    
    def test_config_validation(self):
        """测试配置验证"""
        config_manager = ConfigurationManager(self.config_file)
        errors = config_manager.validate_config()
        
        # 应该没有验证错误
        assert len(errors) == 0 or all(len(error_list) == 0 for error_list in errors.values())


class TestLangChainModelFactory:
    """测试LangChain模型工厂"""
    
    def setup_method(self):
        """测试前设置"""
        self.factory = LangChainModelFactory()
    
    @patch('gfmrag.langchain_factory.ChatOpenAI')
    def test_create_openai_chat_model(self, mock_chat_openai):
        """测试创建OpenAI Chat模型"""
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model
        
        config = ChatConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            api_key="test-key"
        )
        
        model = self.factory.create_chat_model(config)
        
        assert model == mock_model
        mock_chat_openai.assert_called_once()
    
    @patch('gfmrag.langchain_factory.ChatOpenAI')
    def test_create_third_party_chat_model(self, mock_chat_openai):
        """测试创建第三方Chat模型"""
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model
        
        config = ChatConfig(
            provider="third-party",
            model_name="llama-2-7b-chat",
            base_url="http://localhost:8000/v1",
            api_key="placeholder"
        )
        
        model = self.factory.create_chat_model(config)
        
        assert model == mock_model
        mock_chat_openai.assert_called_once()
        
        # 验证调用参数
        call_args = mock_chat_openai.call_args[1]
        assert call_args["base_url"] == "http://localhost:8000/v1"
        assert call_args["model"] == "llama-2-7b-chat"
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试无效配置
        invalid_config = ChatConfig(
            provider="",  # 空提供商
            model_name="gpt-3.5-turbo"
        )
        
        with pytest.raises(ValueError, match="provider不能为空"):
            self.factory._validate_chat_config(invalid_config)
        
        # 测试无效温度
        invalid_temp_config = ChatConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=3.0  # 超出范围
        )
        
        with pytest.raises(ValueError, match="temperature必须在0-2之间"):
            self.factory._validate_chat_config(invalid_temp_config)


class TestEmbeddingModelFactory:
    """测试Embedding模型工厂"""
    
    def setup_method(self):
        """测试前设置"""
        self.factory = EmbeddingModelFactory()
    
    @patch('gfmrag.embedding_factory.OpenAIEmbeddings')
    def test_create_openai_embedding_model(self, mock_openai_embeddings):
        """测试创建OpenAI Embedding模型"""
        mock_model = MagicMock()
        mock_openai_embeddings.return_value = mock_model
        
        config = EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-ada-002",
            api_key="test-key"
        )
        
        model = self.factory.create_embedding_model(config)
        
        assert model == mock_model
        mock_openai_embeddings.assert_called_once()
    
    @patch('gfmrag.embedding_factory.HuggingFaceEmbeddings')
    def test_create_huggingface_embedding_model(self, mock_hf_embeddings):
        """测试创建HuggingFace Embedding模型"""
        mock_model = MagicMock()
        mock_hf_embeddings.return_value = mock_model
        
        config = EmbeddingConfig(
            provider="huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=16,
            normalize=True
        )
        
        model = self.factory.create_embedding_model(config)
        
        assert model == mock_model
        mock_hf_embeddings.assert_called_once()
    
    def test_model_caching(self):
        """测试模型缓存"""
        config = EmbeddingConfig(
            provider="huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        with patch('gfmrag.embedding_factory.HuggingFaceEmbeddings') as mock_hf:
            mock_model = MagicMock()
            mock_hf.return_value = mock_model
            
            # 第一次创建
            model1 = self.factory.create_embedding_model(config)
            
            # 第二次创建（应该从缓存返回）
            model2 = self.factory.create_embedding_model(config)
            
            # 应该是同一个对象
            assert model1 == model2
            
            # HuggingFaceEmbeddings应该只被调用一次
            assert mock_hf.call_count == 1


class TestConfigValidator:
    """测试配置验证器"""
    
    def setup_method(self):
        """测试前设置"""
        self.validator = ConfigValidator()
    
    def test_validate_chat_config_success(self):
        """测试Chat配置验证成功"""
        valid_config = ChatConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            timeout=60,
            max_retries=3
        )
        
        result = self.validator.validate_chat_config(valid_config)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_chat_config_failure(self):
        """测试Chat配置验证失败"""
        invalid_config = ChatConfig(
            provider="",  # 空提供商
            model_name="",  # 空模型名
            temperature=3.0,  # 无效温度
            timeout=-1,  # 无效超时
            max_retries=-1  # 无效重试次数
        )
        
        result = self.validator.validate_chat_config(invalid_config)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_validate_embedding_config_success(self):
        """测试Embedding配置验证成功"""
        valid_config = EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-ada-002",
            batch_size=32,
            timeout=60
        )
        
        result = self.validator.validate_embedding_config(valid_config)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_global_config(self):
        """测试全局配置验证"""
        valid_global_config = GlobalConfig(
            default_provider="openai",
            timeout=60,
            max_retries=3,
            logging_level="INFO"
        )
        
        result = self.validator.validate_global_config(valid_global_config)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0


class TestErrorHandler:
    """测试错误处理"""
    
    def setup_method(self):
        """测试前设置"""
        self.error_classifier = ErrorClassifier()
        self.retry_handler = RetryHandler()
        self.fallback_manager = FallbackManager()
    
    def test_error_classification(self):
        """测试错误分类"""
        # 测试连接错误
        connection_error = ConnectionError("Connection refused")
        error_type = self.error_classifier.classify_error(connection_error)
        assert error_type.value == "connection_error"
        
        # 测试超时错误
        timeout_error = TimeoutError("Request timeout")
        error_type = self.error_classifier.classify_error(timeout_error)
        assert error_type.value == "timeout_error"
    
    def test_retry_mechanism(self):
        """测试重试机制"""
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"
        
        # 模拟重试（减少延迟以加快测试）
        self.retry_handler.retry_config.base_delay = 0.01
        self.retry_handler.retry_config.max_delay = 0.1
        
        result = self.retry_handler.execute_with_retry(failing_function)
        
        assert result == "success"
        assert call_count == 3
    
    def test_fallback_configuration(self):
        """测试备用配置"""
        failed_config = ChatConfig(
            provider="custom",
            model_name="custom-model"
        )
        
        fallback_config = self.fallback_manager.get_fallback_chat_config(failed_config)
        
        assert fallback_config is not None
        assert fallback_config.provider != "custom"
        assert fallback_config.provider in ["openai", "together"]


class TestIntegration:
    """集成测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "integration_config.yaml"
        
        # 创建完整的集成测试配置
        integration_config = {
            "global": {
                "default_provider": "openai",
                "timeout": 60,
                "max_retries": 3,
                "fallback_enabled": True,
                "logging_level": "INFO",
            },
            "chat": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.0,
                "timeout": 60,
                "max_retries": 3,
            },
            "embedding": {
                "provider": "openai",
                "model_name": "text-embedding-ada-002",
                "batch_size": 32,
                "normalize": True,
                "timeout": 60,
            },
            "error_handling": {
                "retry": {
                    "max_retries": 3,
                    "base_delay": 1.0,
                },
                "fallback": {
                    "enabled": True,
                },
            },
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(integration_config, f)
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        # 1. 加载配置
        config_manager = ConfigurationManager(self.config_file)
        
        # 2. 验证配置
        validator = ConfigValidator()
        chat_config = config_manager.get_chat_config()
        embedding_config = config_manager.get_embedding_config()
        
        chat_validation = validator.validate_chat_config(chat_config)
        embedding_validation = validator.validate_embedding_config(embedding_config)
        
        assert chat_validation.is_valid
        assert embedding_validation.is_valid
        
        # 3. 创建模型工厂
        langchain_factory = LangChainModelFactory()
        embedding_factory = EmbeddingModelFactory()
        
        # 4. 模拟模型创建（使用mock避免实际API调用）
        with patch('gfmrag.langchain_factory.ChatOpenAI') as mock_chat, \
             patch('gfmrag.embedding_factory.OpenAIEmbeddings') as mock_embedding:
            
            mock_chat_model = MagicMock()
            mock_embedding_model = MagicMock()
            mock_chat.return_value = mock_chat_model
            mock_embedding.return_value = mock_embedding_model
            
            # 创建模型
            chat_model = langchain_factory.create_chat_model(chat_config)
            embedding_model = embedding_factory.create_embedding_model(embedding_config)
            
            assert chat_model == mock_chat_model
            assert embedding_model == mock_embedding_model
    
    @patch.dict(os.environ, {
        "GFMRAG_CHAT_PROVIDER": "third-party",
        "GFMRAG_CHAT_BASE_URL": "http://localhost:8000/v1",
        "GFMRAG_CHAT_MODEL_NAME": "llama-2-7b-chat",
    })
    def test_third_party_service_configuration(self):
        """测试第三方服务配置"""
        config_manager = ConfigurationManager(self.config_file)
        chat_config = config_manager.get_chat_config()
        
        # 验证环境变量覆盖了配置文件
        assert chat_config.provider == "third-party"
        assert chat_config.base_url == "http://localhost:8000/v1"
        assert chat_config.model_name == "llama-2-7b-chat"
        
        # 验证配置有效性
        validator = ConfigValidator()
        result = validator.validate_chat_config(chat_config)
        
        # 第三方服务配置应该通过基本验证
        assert result.is_valid
    
    def test_configuration_export_import(self):
        """测试配置导出导入"""
        # 1. 创建配置管理器
        config_manager = ConfigurationManager(self.config_file)
        
        # 2. 导出配置
        export_file = Path(self.temp_dir) / "exported_config.yaml"
        config_manager.export_config(export_file)
        
        # 3. 验证导出文件存在
        assert export_file.exists()
        
        # 4. 重新加载导出的配置
        new_config_manager = ConfigurationManager(export_file)
        
        # 5. 验证配置一致性
        original_chat = config_manager.get_chat_config()
        new_chat = new_config_manager.get_chat_config()
        
        assert original_chat.provider == new_chat.provider
        assert original_chat.model_name == new_chat.model_name
        assert original_chat.temperature == new_chat.temperature


if __name__ == "__main__":
    pytest.main([__file__, "-v"])