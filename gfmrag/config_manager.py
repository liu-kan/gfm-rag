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
    
    支持的环境变量：
    - GFMRAG_CHAT_PROVIDER: Chat服务提供商
    - GFMRAG_CHAT_MODEL_NAME: Chat模型名称
    - GFMRAG_CHAT_BASE_URL: Chat服务Base URL
    - GFMRAG_CHAT_KEY: Chat服务认证密钥
    - GFMRAG_EMBEDDING_PROVIDER: Embedding服务提供商
    - GFMRAG_EMBEDDING_MODEL_NAME: Embedding模型名称
    - GFMRAG_EMBEDDING_BASE_URL: Embedding服务Base URL
    - GFMRAG_EMBEDDING_KEY: Embedding服务认证密钥
    """
    
    # 标准环境变量前缀
    ENV_PREFIX = "GFMRAG"
    
    # 配置项优先级（从高到低）
    CONFIG_PRIORITY = ["env", "cli", "yaml", "default"]
    
    # 环境变量映射配置
    ENV_VAR_MAPPING = {
        # Chat 服务环境变量映射
        "CHAT_PROVIDER": ("chat", "provider"),
        "CHAT_MODEL_NAME": ("chat", "model_name"),
        "CHAT_BASE_URL": ("chat", "base_url"),
        "CHAT_KEY": ("chat", "api_key"),
        # Embedding 服务环境变量映射
        "EMBEDDING_PROVIDER": ("embedding", "provider"),
        "EMBEDDING_MODEL_NAME": ("embedding", "model_name"),
        "EMBEDDING_BASE_URL": ("embedding", "base_url"),
        "EMBEDDING_KEY": ("embedding", "api_key"),
    }
    
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
        env_value = self._get_env_config_value(config_type, key)
        if env_value is not None:
            return self._convert_value(env_value)
        
        # 2. YAML配置文件
        yaml_path = f"{config_type}.{key}" if config_type != "global" else key
        yaml_value = self._get_nested_value(self._config_cache, yaml_path)
        if yaml_value is not None:
            return yaml_value
        
        # 3. 返回默认值
        return default
    
    def _get_env_config_value(self, config_type: str, key: str) -> Any:
        """从环境变量获取配置值
        
        Args:
            config_type: 配置类型 ("global", "chat", "embedding")
            key: 配置键名
        
        Returns:
            环境变量值或None
        """
        # 构建环境变量键名
        if config_type == "global":
            env_key = key.upper()
        else:
            env_key = f"{config_type.upper()}_{key.upper()}"
        
        return self.get_env_var(env_key)
    
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
            provider: 服务提供商名称，如果为None则使用环境变量或默认提供商
        
        Returns:
            ChatConfig实例
        """
        # 优先从环境变量获取provider，然后是参数，最后是默认值
        if provider is None:
            provider = (
                self.get_config_value("provider", "chat") or 
                self.get_config_value("default_provider", "global", "openai")
            )
        
        # 获取API密钥，优先使用Chat专用密钥，然后是通用提供商密钥
        api_key = (
            self.get_config_value("api_key", "chat") or 
            self._get_provider_api_key(provider)
        )
        
        # 处理无认证模式：如果api_key为空字符串，则设为None表示无认证
        if api_key == "":
            api_key = None
        
        return ChatConfig(
            provider=provider,
            model_name=self.get_config_value("model_name", "chat", self._get_default_chat_model(provider)),
            api_key=api_key,
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
            provider: 服务提供商名称，如果为None则使用环境变量或默认提供商
        
        Returns:
            EmbeddingConfig实例
        """
        # 优先从环境变量获取provider，然后是参数，最后是默认值
        if provider is None:
            provider = (
                self.get_config_value("provider", "embedding") or 
                self.get_config_value("default_provider", "global", "openai")
            )
        
        # 获取API密钥，优先使用Embedding专用密钥，然后是通用提供商密钥
        api_key = (
            self.get_config_value("api_key", "embedding") or 
            self._get_provider_api_key(provider)
        )
        
        # 处理无认证模式：如果api_key为空字符串，则设为None表示无认证
        if api_key == "":
            api_key = None
        
        return EmbeddingConfig(
            provider=provider,
            model_name=self.get_config_value("model_name", "embedding", self._get_default_embedding_model(provider)),
            api_key=api_key,
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
            "third-party": "gpt-3.5-turbo",  # 第三方服务使用OpenAI兼容接口
            "vllm": "llama-2-7b-chat",
        }
        return default_models.get(provider, "gpt-3.5-turbo")
    
    def _get_default_embedding_model(self, provider: str) -> str:
        """获取服务提供商的默认Embedding模型"""
        default_models = {
            "openai": "text-embedding-ada-002",
            "nvidia": "nvidia/NV-Embed-v1",
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
            "third-party": "text-embedding-ada-002",  # 第三方服务使用OpenAI兼容接口
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
            
            # 验证base_url格式（如果设置）
            if chat_config.base_url:
                self._validate_url(chat_config.base_url, "chat.base_url", errors["chat"])
                
            # 验证provider和model_name兼容性
            self._validate_provider_model_compatibility(
                chat_config.provider, 
                chat_config.model_name, 
                "chat", 
                errors["chat"]
            )
            
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
                
            # 验证base_url格式（如果设置）
            if embedding_config.base_url:
                self._validate_url(embedding_config.base_url, "embedding.base_url", errors["embedding"])
                
            # 验证provider和model_name兼容性
            self._validate_provider_model_compatibility(
                embedding_config.provider, 
                embedding_config.model_name, 
                "embedding", 
                errors["embedding"]
            )
            
        except Exception as e:
            errors["embedding"].append(f"Embedding配置验证失败: {e}")
        
        # 移除空错误列表
        return {k: v for k, v in errors.items() if v}
    
    def _validate_url(self, url: str, field_name: str, errors: List[str]) -> None:
        """验证URL格式"""
        from urllib.parse import urlparse
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                errors.append(f"{field_name} URL格式无效: {url}")
        except Exception:
            errors.append(f"{field_name} URL格式无效: {url}")
    
    def _validate_provider_model_compatibility(
        self, 
        provider: str, 
        model_name: str, 
        config_type: str, 
        errors: List[str]
    ) -> None:
        """验证提供商和模型的兼容性"""
        # 这里可以添加特定的兼容性检查逻辑
        # 例如，检查OpenAI提供商是否使用了正确的模型名称格式
        known_providers = {
            "openai", "nvidia", "together", "anthropic", 
            "ollama", "llama.cpp", "third-party", "vllm", "huggingface"
        }
        
        if provider not in known_providers:
            logger.warning(f"未知的提供商: {provider} (在 {config_type} 配置中)")
    
    def get_environment_variables_summary(self) -> Dict[str, Any]:
        """获取当前设置的环境变量摘要
        
        Returns:
            Dict[str, Any]: 环境变量摘要，不包含敏感信息
        """
        summary = {}
        
        for env_key, (config_type, config_key) in self.ENV_VAR_MAPPING.items():
            full_env_key = f"{self.ENV_PREFIX}_{env_key}"
            value = self._env_cache.get(full_env_key)
            
            if value is not None:
                # 对于API密钥，只显示是否设置，不显示实际值
                if "key" in env_key.lower():
                    summary[full_env_key] = "<已设置>" if value else "<未设置>"
                else:
                    summary[full_env_key] = value
                    
        return summary
    
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


def create_config_from_env() -> ConfigurationManager:
    """从环境变量创建配置管理器
    
    这是一个便捷函数，用于快速从环境变量创建配置管理器实例。
    
    Returns:
        ConfigurationManager: 配置管理器实例
    """
    return ConfigurationManager()


def log_environment_config() -> None:
    """记录当前环境变量配置信息到日志"""
    config_manager = get_config_manager()
    env_summary = config_manager.get_environment_variables_summary()
    
    if env_summary:
        logger.info("检测到的GFMRAG环境变量:")
        for key, value in env_summary.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.info("未检测到GFMRAG环境变量，将使用配置文件或默认值")