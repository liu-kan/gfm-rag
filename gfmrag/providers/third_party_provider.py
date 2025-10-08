"""
第三方OpenAI兼容服务提供商实现

支持各种OpenAI兼容的第三方服务，如vLLM、llama-server等。
"""

from typing import Any, Dict, Union
from urllib.parse import urljoin

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import requests

from gfmrag.config_manager import ChatConfig, EmbeddingConfig
from .base_provider import BaseProvider


class ThirdPartyProvider(BaseProvider):
    """第三方OpenAI兼容服务提供商"""
    
    def __init__(self):
        super().__init__("third-party")
    
    def initialize_chat(self, config: ChatConfig, **kwargs: Any) -> ChatOpenAI:
        """初始化第三方Chat模型
        
        Args:
            config: Chat配置
            **kwargs: 额外参数
        
        Returns:
            ChatOpenAI: OpenAI兼容的Chat模型实例
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
        
        # 第三方服务可能不需要API key，使用占位符或提供的key
        if config.api_key:
            model_kwargs["api_key"] = config.api_key
        else:
            model_kwargs["api_key"] = "placeholder"
        
        if config.max_tokens:
            model_kwargs["max_tokens"] = config.max_tokens
        
        self.logger.info(f"初始化第三方Chat模型: {config.model_name} @ {config.base_url}")
        return ChatOpenAI(**model_kwargs)
    
    def initialize_embedding(self, config: EmbeddingConfig, **kwargs: Any) -> OpenAIEmbeddings:
        """初始化第三方Embedding模型
        
        Args:
            config: Embedding配置
            **kwargs: 额外参数
        
        Returns:
            OpenAIEmbeddings: OpenAI兼容的Embedding模型实例
        """
        if not config.base_url:
            raise ValueError("第三方服务必须提供base_url")
        
        model_kwargs = {
            "model": config.model_name,
            "base_url": config.base_url,
            **kwargs
        }
        
        # 第三方服务可能不需要API key
        if config.api_key:
            model_kwargs["api_key"] = config.api_key
        else:
            model_kwargs["api_key"] = "placeholder"
        
        if config.dimensions:
            model_kwargs["dimensions"] = config.dimensions
        
        self.logger.info(f"初始化第三方Embedding模型: {config.model_name} @ {config.base_url}")
        return OpenAIEmbeddings(**model_kwargs)
    
    def _validate_specific_config(self, config: Union[ChatConfig, EmbeddingConfig]) -> bool:
        """第三方服务特定的配置验证"""
        if not config.base_url:
            self.logger.error("第三方服务必须提供base_url")
            return False
        
        # 验证URL格式
        if not config.base_url.startswith(('http://', 'https://')):
            self.logger.error(f"无效的base_url格式: {config.base_url}")
            return False
        
        return True
    
    def validate_compatibility(self, config: Union[ChatConfig, EmbeddingConfig]) -> bool:
        """验证第三方服务的OpenAI兼容性
        
        Args:
            config: 服务配置
        
        Returns:
            bool: 是否兼容OpenAI API
        """
        if not config.base_url:
            return False
        
        try:
            # 检查models端点
            models_url = urljoin(config.base_url.rstrip('/') + '/', 'models')
            headers = self._get_compatibility_headers(config)
            
            response = requests.get(
                models_url,
                headers=headers,
                timeout=config.timeout or 10,
            )
            
            if response.status_code == 200:
                data = response.json()
                # 检查响应格式是否符合OpenAI API
                if isinstance(data, dict) and 'data' in data:
                    self.logger.info(f"第三方服务兼容性验证通过: {config.base_url}")
                    return True
            
            self.logger.warning(f"第三方服务可能不完全兼容OpenAI API: {config.base_url}")
            return False
            
        except Exception as e:
            self.logger.error(f"兼容性验证失败 {config.base_url}: {e}")
            return False
    
    def _get_compatibility_headers(self, config: Union[ChatConfig, EmbeddingConfig]) -> Dict[str, str]:
        """获取兼容性检查的请求头"""
        headers = {
            "User-Agent": "GFM-RAG/1.0",
            "Content-Type": "application/json",
        }
        
        if config.api_key and config.api_key != "placeholder":
            headers["Authorization"] = f"Bearer {config.api_key}"
        
        if config.custom_headers:
            headers.update(config.custom_headers)
        
        return headers
    
    def _get_health_check_url(self, base_url: str) -> str:
        """获取第三方服务的健康检查URL"""
        # 尝试常见的健康检查端点
        possible_endpoints = [
            'health',
            'v1/models',
            'models',
            '',  # 根路径
        ]
        
        for endpoint in possible_endpoints:
            url = urljoin(base_url.rstrip('/') + '/', endpoint)
            try:
                response = requests.head(url, timeout=5)
                if response.status_code < 500:
                    return url
            except:
                continue
        
        # 默认返回基础URL
        return base_url.rstrip('/')
    
    def _test_chat_connection(self, config: ChatConfig) -> bool:
        """测试第三方Chat连接"""
        try:
            # 首先验证兼容性
            if not self.validate_compatibility(config):
                return False
            
            model = self.initialize_chat(config)
            # 发送简单测试消息
            response = model.invoke("Hello")
            self.logger.debug(f"第三方Chat连接测试成功: {config.model_name}")
            return True
        except Exception as e:
            self.logger.warning(f"第三方Chat连接测试失败: {e}")
            return False
    
    def _test_embedding_connection(self, config: EmbeddingConfig) -> bool:
        """测试第三方Embedding连接"""
        try:
            # 首先验证兼容性
            if not self.validate_compatibility(config):
                return False
            
            model = self.initialize_embedding(config)
            # 发送简单测试文本
            embeddings = model.embed_query("test")
            self.logger.debug(f"第三方Embedding连接测试成功: {config.model_name}")
            return True
        except Exception as e:
            self.logger.warning(f"第三方Embedding连接测试失败: {e}")
            return False
    
    def get_available_models(self, config: Union[ChatConfig, EmbeddingConfig]) -> Dict[str, Any]:
        """获取第三方服务可用的模型列表
        
        Args:
            config: 服务配置
        
        Returns:
            Dict[str, Any]: 可用模型信息
        """
        if not config.base_url:
            return {"chat_models": [], "embedding_models": []}
        
        try:
            models_url = urljoin(config.base_url.rstrip('/') + '/', 'v1/models')
            headers = self._get_compatibility_headers(config)
            
            response = requests.get(
                models_url,
                headers=headers,
                timeout=config.timeout or 10,
            )
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'data' in data:
                    models = data['data']
                    model_names = [model.get('id', '') for model in models]
                    
                    # 简单分类（实际分类可能需要更复杂的逻辑）
                    chat_models = [name for name in model_names if any(
                        keyword in name.lower() for keyword in ['chat', 'instruct', 'gpt', 'llama']
                    )]
                    embedding_models = [name for name in model_names if any(
                        keyword in name.lower() for keyword in ['embed', 'e5', 'bge']
                    )]
                    
                    return {
                        "chat_models": chat_models,
                        "embedding_models": embedding_models,
                    }
            
        except Exception as e:
            self.logger.error(f"获取模型列表失败 {config.base_url}: {e}")
        
        return {"chat_models": [], "embedding_models": []}