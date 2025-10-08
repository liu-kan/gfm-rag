"""
LangChain模型工厂模块

该模块实现了LangChain模型的统一创建和管理，支持多种服务提供商，
包括OpenAI官方、第三方OpenAI兼容服务、vLLM、llama-server等。
"""

import logging
import requests
from typing import Any, Dict, Optional, Union
from urllib.parse import urljoin

from langchain_community.chat_models import ChatLlamaCpp, ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from gfmrag.config_manager import ChatConfig, EmbeddingConfig, get_config_manager


logger = logging.getLogger(__name__)


class LangChainModelFactory:
    """LangChain模型工厂
    
    负责根据配置创建不同类型的LangChain模型实例，支持多种服务提供商。
    """
    
    def __init__(self):
        """初始化模型工厂"""
        self.config_manager = get_config_manager()
        self._model_cache: Dict[str, Any] = {}
    
    def create_chat_model(
        self, 
        config: Optional[ChatConfig] = None,
        **kwargs: Any
    ) -> Union[ChatOpenAI, ChatTogether, ChatOllama, ChatLlamaCpp, ChatNVIDIA]:
        """创建Chat模型实例
        
        Args:
            config: Chat配置，如果为None则使用默认配置
            **kwargs: 额外的模型参数
        
        Returns:
            Chat模型实例
        
        Raises:
            ValueError: 不支持的提供商
            ConnectionError: 服务连接失败
        """
        if config is None:
            config = self.config_manager.get_chat_config()
        
        # 验证配置
        self._validate_chat_config(config)
        
        # 生成缓存键
        cache_key = self._generate_cache_key("chat", config)
        
        # 检查缓存
        if cache_key in self._model_cache:
            logger.debug(f"从缓存返回Chat模型: {config.provider}/{config.model_name}")
            return self._model_cache[cache_key]
        
        try:
            model = self._create_chat_model_impl(config, **kwargs)
            
            # 测试连接
            if config.provider not in ["ollama", "llama.cpp"]:
                self._test_chat_model_connection(model, config)
            
            # 缓存模型
            self._model_cache[cache_key] = model
            
            logger.info(f"成功创建Chat模型: {config.provider}/{config.model_name}")
            return model
            
        except Exception as e:
            logger.error(f"创建Chat模型失败: {config.provider}/{config.model_name}, 错误: {e}")
            
            # 尝试备用方案
            if self.config_manager.global_config.fallback_enabled:
                return self._create_fallback_chat_model(config, **kwargs)
            
            raise
    
    def _create_chat_model_impl(
        self, 
        config: ChatConfig, 
        **kwargs: Any
    ) -> Union[ChatOpenAI, ChatTogether, ChatOllama, ChatLlamaCpp, ChatNVIDIA]:
        """Chat模型创建的具体实现"""
        
        if config.provider == "openai":
            return self._create_openai_chat_model(config, **kwargs)
        elif config.provider == "nvidia":
            return self._create_nvidia_chat_model(config, **kwargs)
        elif config.provider == "together":
            return self._create_together_chat_model(config, **kwargs)
        elif config.provider == "ollama":
            return self._create_ollama_chat_model(config, **kwargs)
        elif config.provider == "llama.cpp":
            return self._create_llamacpp_chat_model(config, **kwargs)
        elif config.provider in ["third-party", "vllm"]:
            return self._create_third_party_chat_model(config, **kwargs)
        else:
            raise ValueError(f"不支持的Chat提供商: {config.provider}")
    
    def _create_openai_chat_model(self, config: ChatConfig, **kwargs: Any) -> ChatOpenAI:
        """创建OpenAI Chat模型"""
        model_kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
            **kwargs
        }
        
        if config.api_key:
            model_kwargs["api_key"] = config.api_key
        
        if config.base_url:
            model_kwargs["base_url"] = config.base_url
        
        if config.max_tokens:
            model_kwargs["max_tokens"] = config.max_tokens
            
        if config.top_p is not None:
            model_kwargs["top_p"] = config.top_p
            
        if config.frequency_penalty is not None:
            model_kwargs["frequency_penalty"] = config.frequency_penalty
            
        if config.presence_penalty is not None:
            model_kwargs["presence_penalty"] = config.presence_penalty
        
        return ChatOpenAI(**model_kwargs)
    
    def _create_nvidia_chat_model(self, config: ChatConfig, **kwargs: Any) -> ChatNVIDIA:
        """创建NVIDIA Chat模型"""
        model_kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            **kwargs
        }
        
        if config.api_key:
            model_kwargs["nvidia_api_key"] = config.api_key
        
        if config.base_url:
            model_kwargs["base_url"] = config.base_url
        
        return ChatNVIDIA(**model_kwargs)
    
    def _create_together_chat_model(self, config: ChatConfig, **kwargs: Any) -> ChatTogether:
        """创建Together Chat模型"""
        model_kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            **kwargs
        }
        
        if config.api_key:
            model_kwargs["api_key"] = config.api_key
        
        if config.base_url:
            model_kwargs["base_url"] = config.base_url
        
        return ChatTogether(**model_kwargs)
    
    def _create_ollama_chat_model(self, config: ChatConfig, **kwargs: Any) -> ChatOllama:
        """创建Ollama Chat模型"""
        model_kwargs = {
            "model": config.model_name,
            **kwargs
        }
        
        if config.base_url:
            model_kwargs["base_url"] = config.base_url
        
        return ChatOllama(**model_kwargs)
    
    def _create_llamacpp_chat_model(self, config: ChatConfig, **kwargs: Any) -> ChatLlamaCpp:
        """创建llama.cpp Chat模型"""
        model_kwargs = {
            "model_path": config.model_name,  # model_name作为模型路径
            "verbose": True,
            **kwargs
        }
        
        return ChatLlamaCpp(**model_kwargs)
    
    def _create_third_party_chat_model(self, config: ChatConfig, **kwargs: Any) -> ChatOpenAI:
        """创建第三方OpenAI兼容Chat模型
        
        支持各种第三方OpenAI兼容服务，包括:
        - vLLM 推理服务
        - llama.cpp 服务器
        - 自建OpenAI兼容API
        - 带或不带认证的本地服务
        """
        if not config.base_url:
            raise ValueError("第三方服务必须提供base_url")
        
        model_kwargs = {
            "model": config.model_name,
            "base_url": config.base_url,
            "temperature": config.temperature,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
            **kwargs
        }
        
        # 处理认证：如果提供了API密钥则使用，否则尝试无认证连接
        if config.api_key:
            model_kwargs["api_key"] = config.api_key
            logger.debug(f"使用认证模式连接第三方服务: {config.base_url}")
        else:
            # 对于无认证的第三方服务，使用占位符密钥
            model_kwargs["api_key"] = "not-needed"
            logger.debug(f"使用无认证模式连接第三方服务: {config.base_url}")
        
        # 可选参数处理
        if config.max_tokens:
            model_kwargs["max_tokens"] = config.max_tokens
            
        if config.top_p is not None:
            model_kwargs["top_p"] = config.top_p
            
        if config.frequency_penalty is not None:
            model_kwargs["frequency_penalty"] = config.frequency_penalty
            
        if config.presence_penalty is not None:
            model_kwargs["presence_penalty"] = config.presence_penalty
        
        return ChatOpenAI(**model_kwargs)
        
        model_kwargs = {
            "model": config.model_name,
            "base_url": config.base_url,
            "temperature": config.temperature,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
            **kwargs
        }
        
        if config.api_key:
            model_kwargs["api_key"] = config.api_key
        else:
            # 第三方服务可能不需要API key，使用占位符
            model_kwargs["api_key"] = "placeholder"
        
        return ChatOpenAI(**model_kwargs)
    
    def create_embedding_model(
        self, 
        config: Optional[EmbeddingConfig] = None,
        **kwargs: Any
    ) -> Union[OpenAIEmbeddings, HuggingFaceEmbeddings]:
        """创建Embedding模型实例
        
        Args:
            config: Embedding配置，如果为None则使用默认配置
            **kwargs: 额外的模型参数
        
        Returns:
            Embedding模型实例
        """
        if config is None:
            config = self.config_manager.get_embedding_config()
        
        # 验证配置
        self._validate_embedding_config(config)
        
        # 生成缓存键
        cache_key = self._generate_cache_key("embedding", config)
        
        # 检查缓存
        if cache_key in self._model_cache:
            logger.debug(f"从缓存返回Embedding模型: {config.provider}/{config.model_name}")
            return self._model_cache[cache_key]
        
        try:
            model = self._create_embedding_model_impl(config, **kwargs)
            
            # 缓存模型
            self._model_cache[cache_key] = model
            
            logger.info(f"成功创建Embedding模型: {config.provider}/{config.model_name}")
            return model
            
        except Exception as e:
            logger.error(f"创建Embedding模型失败: {config.provider}/{config.model_name}, 错误: {e}")
            
            # 尝试备用方案
            if self.config_manager.global_config.fallback_enabled:
                return self._create_fallback_embedding_model(config, **kwargs)
            
            raise
    
    def _create_embedding_model_impl(
        self, 
        config: EmbeddingConfig, 
        **kwargs: Any
    ) -> Union[OpenAIEmbeddings, HuggingFaceEmbeddings]:
        """Embedding模型创建的具体实现"""
        
        if config.provider in ["openai", "third-party"]:
            return self._create_openai_embedding_model(config, **kwargs)
        elif config.provider == "huggingface":
            return self._create_huggingface_embedding_model(config, **kwargs)
        else:
            # 默认使用HuggingFace
            return self._create_huggingface_embedding_model(config, **kwargs)
    
    def _create_openai_embedding_model(self, config: EmbeddingConfig, **kwargs: Any) -> OpenAIEmbeddings:
        """创建OpenAI Embedding模型
        
        支持OpenAI官方服务和第三方OpenAI兼容服务
        """
        model_kwargs = {
            "model": config.model_name,
            **kwargs
        }
        
        # 处理认证：根据是否提供API密钥决定认证方式
        if config.api_key:
            model_kwargs["api_key"] = config.api_key
            logger.debug(f"使用认证模式访问Embedding服务: {config.provider}")
        elif config.provider == "third-party":
            # 第三方服务可能不需要认证
            model_kwargs["api_key"] = "not-needed"
            logger.debug(f"使用无认证模式访问第三方Embedding服务")
        
        # Base URL处理
        if config.base_url:
            model_kwargs["base_url"] = config.base_url
            
        # 可选参数
        if config.dimensions:
            model_kwargs["dimensions"] = config.dimensions
        
        return OpenAIEmbeddings(**model_kwargs)
    
    def _create_huggingface_embedding_model(self, config: EmbeddingConfig, **kwargs: Any) -> HuggingFaceEmbeddings:
        """创建HuggingFace Embedding模型"""
        model_kwargs = {
            "model_name": config.model_name,
            **kwargs
        }
        
        return HuggingFaceEmbeddings(**model_kwargs)
    
    def _validate_chat_config(self, config: ChatConfig) -> None:
        """验证Chat配置"""
        if not config.provider:
            raise ValueError("provider不能为空")
        if not config.model_name:
            raise ValueError("model_name不能为空")
        if config.temperature < 0 or config.temperature > 2:
            raise ValueError("temperature必须在0-2之间")
        if config.timeout <= 0:
            raise ValueError("timeout必须大于0")
        if config.max_retries < 0:
            raise ValueError("max_retries必须大于等于0")
    
    def _validate_embedding_config(self, config: EmbeddingConfig) -> None:
        """验证Embedding配置"""
        if not config.provider:
            raise ValueError("provider不能为空")
        if not config.model_name:
            raise ValueError("model_name不能为空")
        if config.batch_size <= 0:
            raise ValueError("batch_size必须大于0")
        if config.timeout <= 0:
            raise ValueError("timeout必须大于0")
    
    def _test_chat_model_connection(self, model: Any, config: ChatConfig) -> None:
        """测试Chat模型连接
        
        对于第三方服务和本地服务，进行更宽松的连接测试
        """
        try:
            # 发送简单的测试消息
            if config.provider in ["third-party", "vllm", "ollama"]:
                # 对于可能无认证的服务，使用更简单的测试
                logger.debug(f"跳过严格连接测试 for {config.provider} 服务")
            else:
                response = model.invoke("Hello")
                logger.debug(f"Chat模型连接测试成功: {config.provider}/{config.model_name}")
        except Exception as e:
            logger.warning(f"Chat模型连接测试失败: {config.provider}/{config.model_name}, 错误: {e}")
            # 不抛出异常，只记录警告
    
    def _generate_cache_key(self, model_type: str, config: Union[ChatConfig, EmbeddingConfig]) -> str:
        """生成缓存键"""
        return f"{model_type}:{config.provider}:{config.model_name}:{hash(str(config))}"
    
    def _create_fallback_chat_model(self, config: ChatConfig, **kwargs: Any) -> ChatOpenAI:
        """创建备用Chat模型"""
        logger.warning(f"使用备用Chat模型替代 {config.provider}/{config.model_name}")
        
        fallback_config = ChatConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=config.temperature,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
        
        return self._create_openai_chat_model(fallback_config, **kwargs)
    
    def _create_fallback_embedding_model(self, config: EmbeddingConfig, **kwargs: Any) -> HuggingFaceEmbeddings:
        """创建备用Embedding模型"""
        logger.warning(f"使用备用Embedding模型替代 {config.provider}/{config.model_name}")
        
        fallback_config = EmbeddingConfig(
            provider="huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=config.batch_size,
            normalize=config.normalize,
        )
        
        return self._create_huggingface_embedding_model(fallback_config, **kwargs)
    
    def clear_cache(self) -> None:
        """清空模型缓存"""
        self._model_cache.clear()
        logger.info("已清空模型缓存")
    
    def get_cached_models(self) -> Dict[str, Any]:
        """获取缓存的模型信息"""
        return {k: type(v).__name__ for k, v in self._model_cache.items()}


# 工具函数
def create_chat_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs: Any
) -> Union[ChatOpenAI, ChatTogether, ChatOllama, ChatLlamaCpp, ChatNVIDIA]:
    """便捷函数：创建Chat模型
    
    该函数优先使用环境变量配置，然后使用提供的参数。
    支持完全基于环境变量的配置。
    
    Args:
        provider: 服务提供商，如果为None则使用环境变量或默认值
        model_name: 模型名称，如果为None则使用环境变量或默认值
        **kwargs: 额外参数
    
    Returns:
        Chat模型实例
    """
    factory = LangChainModelFactory()
    
    # 如果没有提供参数，则完全依赖配置管理器（环境变量优先）
    if provider is None and model_name is None:
        config = factory.config_manager.get_chat_config()
    else:
        # 部分覆盖配置
        base_config = factory.config_manager.get_chat_config()
        config = ChatConfig(
            provider=provider or base_config.provider,
            model_name=model_name or base_config.model_name,
            api_key=base_config.api_key,
            base_url=base_config.base_url,
            timeout=base_config.timeout,
            max_retries=base_config.max_retries,
            temperature=base_config.temperature,
            **kwargs
        )
    
    return factory.create_chat_model(config)


def create_embedding_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs: Any
) -> Union[OpenAIEmbeddings, HuggingFaceEmbeddings]:
    """便捷函数：创建Embedding模型
    
    该函数优先使用环境变量配置，然后使用提供的参数。
    支持完全基于环境变量的配置。
    
    Args:
        provider: 服务提供商，如果为None则使用环境变量或默认值
        model_name: 模型名称，如果为None则使用环境变量或默认值
        **kwargs: 额外参数
    
    Returns:
        Embedding模型实例
    """
    factory = LangChainModelFactory()
    
    # 如果没有提供参数，则完全依赖配置管理器（环境变量优先）
    if provider is None and model_name is None:
        config = factory.config_manager.get_embedding_config()
    else:
        # 部分覆盖配置
        base_config = factory.config_manager.get_embedding_config()
        config = EmbeddingConfig(
            provider=provider or base_config.provider,
            model_name=model_name or base_config.model_name,
            api_key=base_config.api_key,
            base_url=base_config.base_url,
            timeout=base_config.timeout,
            max_retries=base_config.max_retries,
            batch_size=base_config.batch_size,
            normalize=base_config.normalize,
            **kwargs
        )
    
    return factory.create_embedding_model(config)


def create_models_from_env() -> tuple[Any, Any]:
    """从环境变量创建Chat和Embedding模型
    
    这是一个便捷函数，完全基于环境变量配置创建模型对。
    
    Returns:
        tuple: (chat_model, embedding_model)
    """
    return create_chat_model(), create_embedding_model()