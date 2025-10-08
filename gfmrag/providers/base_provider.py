"""
服务提供商基类模块

定义了服务提供商的抽象接口和基础功能。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import requests
from urllib.parse import urljoin

from gfmrag.config_manager import ChatConfig, EmbeddingConfig


logger = logging.getLogger(__name__)


@dataclass
class ServiceProvider:
    """服务提供商配置数据类"""
    provider_name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    custom_headers: Optional[Dict[str, str]] = None
    proxy_settings: Optional[Dict[str, str]] = None


class BaseProvider(ABC):
    """服务提供商基类
    
    定义了所有服务提供商必须实现的接口方法。
    """
    
    def __init__(self, provider_name: str):
        """初始化服务提供商
        
        Args:
            provider_name: 提供商名称
        """
        self.provider_name = provider_name
        self.logger = logging.getLogger(f"{__name__}.{provider_name}")
    
    @abstractmethod
    def initialize_chat(self, config: ChatConfig, **kwargs: Any) -> Any:
        """初始化Chat模型
        
        Args:
            config: Chat配置
            **kwargs: 额外参数
        
        Returns:
            Chat模型实例
        """
        pass
    
    @abstractmethod
    def initialize_embedding(self, config: EmbeddingConfig, **kwargs: Any) -> Any:
        """初始化Embedding模型
        
        Args:
            config: Embedding配置
            **kwargs: 额外参数
        
        Returns:
            Embedding模型实例
        """
        pass
    
    def validate_provider(self, config: Union[ChatConfig, EmbeddingConfig]) -> bool:
        """验证提供商配置
        
        Args:
            config: 服务配置
        
        Returns:
            bool: 验证是否通过
        """
        try:
            # 基础验证
            if not config.provider:
                self.logger.error("provider不能为空")
                return False
            
            if not config.model_name:
                self.logger.error("model_name不能为空")
                return False
            
            # 子类可以重写此方法添加特定验证
            return self._validate_specific_config(config)
            
        except Exception as e:
            self.logger.error(f"配置验证失败: {e}")
            return False
    
    def _validate_specific_config(self, config: Union[ChatConfig, EmbeddingConfig]) -> bool:
        """子类特定的配置验证
        
        Args:
            config: 服务配置
        
        Returns:
            bool: 验证是否通过
        """
        return True
    
    def health_check(self, config: Union[ChatConfig, EmbeddingConfig]) -> bool:
        """健康检查
        
        Args:
            config: 服务配置
        
        Returns:
            bool: 服务是否健康
        """
        if not config.base_url:
            return True  # 本地服务或无需健康检查
        
        try:
            # 构建健康检查URL
            health_url = self._get_health_check_url(config.base_url)
            
            # 发送健康检查请求
            response = requests.get(
                health_url,
                timeout=config.timeout or 10,
                headers=self._get_health_check_headers(config),
            )
            
            return response.status_code == 200
            
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"健康检查失败 {config.base_url}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"健康检查异常 {config.base_url}: {e}")
            return False
    
    def _get_health_check_url(self, base_url: str) -> str:
        """获取健康检查URL
        
        Args:
            base_url: 基础URL
        
        Returns:
            str: 健康检查URL
        """
        # 默认使用基础URL，子类可以重写
        return base_url.rstrip('/')
    
    def _get_health_check_headers(self, config: Union[ChatConfig, EmbeddingConfig]) -> Dict[str, str]:
        """获取健康检查请求头
        
        Args:
            config: 服务配置
        
        Returns:
            Dict[str, str]: 请求头
        """
        headers = {"User-Agent": "GFM-RAG/1.0"}
        
        if config.custom_headers:
            headers.update(config.custom_headers)
        
        return headers
    
    def test_connection(self, config: Union[ChatConfig, EmbeddingConfig]) -> bool:
        """测试连接
        
        Args:
            config: 服务配置
        
        Returns:
            bool: 连接是否成功
        """
        try:
            if isinstance(config, ChatConfig):
                return self._test_chat_connection(config)
            elif isinstance(config, EmbeddingConfig):
                return self._test_embedding_connection(config)
            else:
                return False
        except Exception as e:
            self.logger.error(f"连接测试失败: {e}")
            return False
    
    def _test_chat_connection(self, config: ChatConfig) -> bool:
        """测试Chat连接
        
        Args:
            config: Chat配置
        
        Returns:
            bool: 连接是否成功
        """
        # 子类实现具体的连接测试
        return True
    
    def _test_embedding_connection(self, config: EmbeddingConfig) -> bool:
        """测试Embedding连接
        
        Args:
            config: Embedding配置
        
        Returns:
            bool: 连接是否成功
        """
        # 子类实现具体的连接测试
        return True
    
    def get_model_info(self, config: Union[ChatConfig, EmbeddingConfig]) -> Dict[str, Any]:
        """获取模型信息
        
        Args:
            config: 服务配置
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            "provider": self.provider_name,
            "model_name": config.model_name,
            "base_url": config.base_url,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }
    
    def get_supported_models(self) -> Dict[str, Any]:
        """获取支持的模型列表
        
        Returns:
            Dict[str, Any]: 支持的模型信息
        """
        # 子类可以重写此方法返回具体的模型列表
        return {
            "chat_models": [],
            "embedding_models": [],
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider_name='{self.provider_name}')"