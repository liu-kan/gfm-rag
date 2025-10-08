"""
统一配置管理器模块

该模块实现了GFM-RAG项目的统一配置管理架构，支持多层配置、环境变量管理、
配置验证等功能，为LangChain优化提供配置支持。
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import yaml
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """服务配置基类"""
    provider: str
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.0
    custom_headers: Optional[Dict[str, str]] = None
    proxy_settings: Optional[Dict[str, str]] = None


@dataclass
class ChatConfig(ServiceConfig):
    """Chat服务配置"""
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


@dataclass
class EmbeddingConfig(ServiceConfig):
    """Embedding服务配置"""
    batch_size: int = 32
    normalize: bool = True
    dimensions: Optional[int] = None
    query_instruct: Optional[str] = None
    passage_instruct: Optional[str] = None


@dataclass
class GlobalConfig:
    """全局配置"""
    default_provider: str = "openai"
    timeout: int = 60
    max_retries: int = 3
    fallback_enabled: bool = True
    logging_level: str = "INFO"
    cache_enabled: bool = True
    performance_monitoring: bool = False


class ConfigurationManager:
    """统一配置管理器
    
    提供多层配置管理、环境变量处理、配置验证等功能。
    支持从环境变量、YAML文件、命令行参数等多种来源加载配置。
    """
    
    # 标准环境变量前缀
    ENV_PREFIX = "GFMRAG"
    
    # 配置项优先级（从高到低）
    CONFIG_PRIORITY = ["env", "cli", "yaml", "default"]
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径，可选
        """
        self.config_path = Path(config_path) if config_path else None
        self.global_config = GlobalConfig()
        self._config_cache: Dict[str, Any] = {}
        self._env_cache: Dict[str, str] = {}
        
        # 加载环境变量缓存
        self._load_env_cache()
        
        # 如果提供了配置文件路径，则加载配置文件
        if self.config_path and self.config_path.exists():
            self.load_config_file(self.config_path)
    
    def _load_env_cache(self) -> None:
        """加载环境变量到缓存"""
        for key, value in os.environ.items():
            if key.startswith(f"{self.ENV_PREFIX}_"):
                self._env_cache[key] = value
    
    def load_config_file(self, config_path: Union[str, Path]) -> None:
        """从YAML文件加载配置
        
        Args:
            config_path: 配置文件路径
        """
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_path}")
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 使用OmegaConf处理配置
            self._config_cache.update(config_data or {})
            logger.info(f"成功加载配置文件: {config_path}")
            
        except Exception as e:
            logger.error(f"加载配置文件失败 {config_path}: {e}")
            raise
    
    def get_env_var(self, key: str, default: Any = None) -> Any:
        """获取环境变量值
        
        Args:
            key: 环境变量名（不包含前缀）
            default: 默认值
        
        Returns:
            环境变量值或默认值
        """
        full_key = f"{self.ENV_PREFIX}_{key.upper()}"
        return self._env_cache.get(full_key, default)
    
    def get_config_value(self, key: str, config_type: str = "global", default: Any = None) -> Any:
        """获取配置值（按优先级）
        
        Args:
            key: 配置键名
            config_type: 配置类型 ("global", "chat", "embedding")
            default: 默认值
        
        Returns:
            配置值
        """
        # 1. 环境变量优先级最高
        env_key = f"{config_type.upper()}_{key.upper()}" if config_type != "global" else key.upper()
        env_value = self.get_env_var(env_key)
        if env_value is not None:
            return self._convert_value(env_value)
        
        # 2. YAML配置文件
        yaml_path = f"{config_type}.{key}" if config_type != "global" else key
        yaml_value = self._get_nested_value(self._config_cache, yaml_path)
        if yaml_value is not None:
            return yaml_value
        
        # 3. 返回默认值
        return default
    
    def _convert_value(self, value: str) -> Any:
        """转换字符串值为适当的类型"""
        # 布尔值转换
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # 数字转换
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # 返回字符串
        return value
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """获取嵌套字典中的值"""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def get_chat_config(self, provider: Optional[str] = None) -> ChatConfig:
        """获取Chat服务配置
        
        Args:
            provider: 服务提供商名称，如果为None则使用默认提供商
        
        Returns:
            ChatConfig实例
        """
        if provider is None:
            provider = self.get_config_value("default_provider", "global", "openai")
        
        return ChatConfig(
            provider=provider,
            model_name=self.get_config_value("model_name", "chat", self._get_default_chat_model(provider)),
            api_key=self.get_config_value("api_key", "chat") or self._get_provider_api_key(provider),
            base_url=self.get_config_value("base_url", "chat") or self._get_default_base_url(provider),
            timeout=self.get_config_value("timeout", "chat", self.global_config.timeout),
            max_retries=self.get_config_value("max_retries", "chat", self.global_config.max_retries),
            temperature=self.get_config_value("temperature", "chat", 0.0),
            max_tokens=self.get_config_value("max_tokens", "chat"),
            top_p=self.get_config_value("top_p", "chat"),
            frequency_penalty=self.get_config_value("frequency_penalty", "chat"),
            presence_penalty=self.get_config_value("presence_penalty", "chat"),
            custom_headers=self.get_config_value("custom_headers", "chat"),
            proxy_settings=self.get_config_value("proxy_settings", "chat"),
        )
    
    def get_embedding_config(self, provider: Optional[str] = None) -> EmbeddingConfig:
        """获取Embedding服务配置
        
        Args:
            provider: 服务提供商名称，如果为None则使用默认提供商
        
        Returns:
            EmbeddingConfig实例
        """
        if provider is None:
            provider = self.get_config_value("default_provider", "global", "openai")
        
        return EmbeddingConfig(
            provider=provider,
            model_name=self.get_config_value("model_name", "embedding", self._get_default_embedding_model(provider)),
            api_key=self.get_config_value("api_key", "embedding") or self._get_provider_api_key(provider),
            base_url=self.get_config_value("base_url", "embedding") or self._get_default_base_url(provider),
            timeout=self.get_config_value("timeout", "embedding", self.global_config.timeout),
            max_retries=self.get_config_value("max_retries", "embedding", self.global_config.max_retries),
            temperature=self.get_config_value("temperature", "embedding", 0.0),
            batch_size=self.get_config_value("batch_size", "embedding", 32),
            normalize=self.get_config_value("normalize", "embedding", True),
            dimensions=self.get_config_value("dimensions", "embedding"),
            query_instruct=self.get_config_value("query_instruct", "embedding"),
            passage_instruct=self.get_config_value("passage_instruct", "embedding"),
            custom_headers=self.get_config_value("custom_headers", "embedding"),
            proxy_settings=self.get_config_value("proxy_settings", "embedding"),
        )
    
    def _get_provider_api_key(self, provider: str) -> Optional[str]:
        """获取服务提供商的API密钥"""
        # 标准API密钥环境变量
        api_key_map = {
            "openai": "OPENAI_API_KEY",
            "together": "TOGETHER_API_KEY", 
            "nvidia": "NVIDIA_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        
        standard_key = api_key_map.get(provider)
        if standard_key:
            return os.environ.get(standard_key)
        
        # 自定义提供商API密钥
        custom_key = f"{provider.upper()}_API_KEY"
        return os.environ.get(custom_key)
    
    def _get_default_base_url(self, provider: str) -> Optional[str]:
        """获取服务提供商的默认基础URL"""
        default_urls = {
            "openai": "https://api.openai.com/v1",
            "nvidia": "https://integrate.api.nvidia.com/v1",
            "together": "https://api.together.xyz/v1",
            "anthropic": "https://api.anthropic.com",
        }
        return default_urls.get(provider)
    
    def _get_default_chat_model(self, provider: str) -> str:
        """获取服务提供商的默认Chat模型"""
        default_models = {
            "openai": "gpt-3.5-turbo",
            "nvidia": "meta/llama3-70b-instruct",
            "together": "meta-llama/Llama-2-70b-chat-hf",
            "anthropic": "claude-3-sonnet-20240229",
            "ollama": "llama3",
            "llama.cpp": "llama-2-7b-chat.gguf",
        }
        return default_models.get(provider, "gpt-3.5-turbo")
    
    def _get_default_embedding_model(self, provider: str) -> str:
        """获取服务提供商的默认Embedding模型"""
        default_models = {
            "openai": "text-embedding-ada-002",
            "nvidia": "nvidia/NV-Embed-v1",
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        }
        return default_models.get(provider, "text-embedding-ada-002")
    
    def validate_config(self) -> Dict[str, List[str]]:
        """验证配置
        
        Returns:
            Dict[str, List[str]]: 验证错误信息，按配置类型分组
        """
        errors = {"global": [], "chat": [], "embedding": []}
        
        # 验证全局配置
        try:
            global_config = self.global_config
            if global_config.timeout <= 0:
                errors["global"].append("timeout必须大于0")
            if global_config.max_retries < 0:
                errors["global"].append("max_retries必须大于等于0")
        except Exception as e:
            errors["global"].append(f"全局配置验证失败: {e}")
        
        # 验证Chat配置
        try:
            chat_config = self.get_chat_config()
            if not chat_config.provider:
                errors["chat"].append("provider不能为空")
            if not chat_config.model_name:
                errors["chat"].append("model_name不能为空")
        except Exception as e:
            errors["chat"].append(f"Chat配置验证失败: {e}")
        
        # 验证Embedding配置
        try:
            embedding_config = self.get_embedding_config()
            if not embedding_config.provider:
                errors["embedding"].append("provider不能为空")
            if not embedding_config.model_name:
                errors["embedding"].append("model_name不能为空")
            if embedding_config.batch_size <= 0:
                errors["embedding"].append("batch_size必须大于0")
        except Exception as e:
            errors["embedding"].append(f"Embedding配置验证失败: {e}")
        
        # 移除空错误列表
        return {k: v for k, v in errors.items() if v}
    
    def export_config(self, output_path: Union[str, Path]) -> None:
        """导出当前配置到YAML文件
        
        Args:
            output_path: 输出文件路径
        """
        config_dict = {
            "global": {
                "default_provider": self.global_config.default_provider,
                "timeout": self.global_config.timeout,
                "max_retries": self.global_config.max_retries,
                "fallback_enabled": self.global_config.fallback_enabled,
                "logging_level": self.global_config.logging_level,
                "cache_enabled": self.global_config.cache_enabled,
                "performance_monitoring": self.global_config.performance_monitoring,
            },
            "chat": self.get_chat_config().__dict__,
            "embedding": self.get_embedding_config().__dict__,
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"配置已导出到: {output_path}")


# 全局配置管理器实例
_global_config_manager: Optional[ConfigurationManager] = None


def get_config_manager() -> ConfigurationManager:
    """获取全局配置管理器实例"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigurationManager()
    return _global_config_manager


def set_config_manager(config_manager: ConfigurationManager) -> None:
    """设置全局配置管理器实例"""
    global _global_config_manager
    _global_config_manager = config_manager


def reset_config_manager() -> None:
    """重置全局配置管理器实例"""
    global _global_config_manager
    _global_config_manager = None