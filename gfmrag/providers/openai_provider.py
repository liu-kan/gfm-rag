"""
OpenAI服务提供商实现

支持OpenAI官方API服务的Chat和Embedding模型。
"""

from typing import Any, Dict, Union

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from gfmrag.config_manager import ChatConfig, EmbeddingConfig
from .base_provider import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI服务提供商"""
    
    def __init__(self):
        super().__init__("openai")
    
    def initialize_chat(self, config: ChatConfig, **kwargs: Any) -> ChatOpenAI:
        """初始化OpenAI Chat模型
        
        Args:
            config: Chat配置
            **kwargs: 额外参数
        
        Returns:
            ChatOpenAI: OpenAI Chat模型实例
        """
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
        
        self.logger.info(f"初始化OpenAI Chat模型: {config.model_name}")
        return ChatOpenAI(**model_kwargs)
    
    def initialize_embedding(self, config: EmbeddingConfig, **kwargs: Any) -> OpenAIEmbeddings:
        """初始化OpenAI Embedding模型
        
        Args:
            config: Embedding配置
            **kwargs: 额外参数
        
        Returns:
            OpenAIEmbeddings: OpenAI Embedding模型实例
        """
        model_kwargs = {
            "model": config.model_name,
            **kwargs
        }
        
        if config.api_key:
            model_kwargs["api_key"] = config.api_key
        
        if config.base_url:
            model_kwargs["base_url"] = config.base_url
            
        if config.dimensions:
            model_kwargs["dimensions"] = config.dimensions
        
        self.logger.info(f"初始化OpenAI Embedding模型: {config.model_name}")
        return OpenAIEmbeddings(**model_kwargs)
    
    def _validate_specific_config(self, config: Union[ChatConfig, EmbeddingConfig]) -> bool:
        """OpenAI特定的配置验证"""
        # 验证模型名称格式
        if isinstance(config, ChatConfig):
            if not config.model_name.startswith(("gpt-", "o1-")):
                self.logger.warning(f"OpenAI Chat模型名称可能不正确: {config.model_name}")
        elif isinstance(config, EmbeddingConfig):
            if not config.model_name.startswith("text-embedding-"):
                self.logger.warning(f"OpenAI Embedding模型名称可能不正确: {config.model_name}")
        
        return True
    
    def _test_chat_connection(self, config: ChatConfig) -> bool:
        """测试OpenAI Chat连接"""
        try:
            model = self.initialize_chat(config)
            # 发送简单测试消息
            response = model.invoke("Hello")
            self.logger.debug(f"OpenAI Chat连接测试成功: {config.model_name}")
            return True
        except Exception as e:
            self.logger.warning(f"OpenAI Chat连接测试失败: {e}")
            return False
    
    def _test_embedding_connection(self, config: EmbeddingConfig) -> bool:
        """测试OpenAI Embedding连接"""
        try:
            model = self.initialize_embedding(config)
            # 发送简单测试文本
            embeddings = model.embed_query("test")
            self.logger.debug(f"OpenAI Embedding连接测试成功: {config.model_name}")
            return True
        except Exception as e:
            self.logger.warning(f"OpenAI Embedding连接测试失败: {e}")
            return False
    
    def get_supported_models(self) -> Dict[str, Any]:
        """获取OpenAI支持的模型列表"""
        return {
            "chat_models": [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-3.5-turbo",
                "o1-preview",
                "o1-mini",
            ],
            "embedding_models": [
                "text-embedding-ada-002",
                "text-embedding-3-small", 
                "text-embedding-3-large",
            ],
        }