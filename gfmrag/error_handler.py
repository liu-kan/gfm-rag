"""
错误处理和容错机制模块

该模块实现了完整的错误处理和容错策略，包括重试机制、备用方案、
优雅降级等功能，确保系统的健壮性和稳定性。
"""

import logging
import time
import random
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
import functools

from gfmrag.config_manager import ChatConfig, EmbeddingConfig, get_config_manager


logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """错误类型枚举"""
    CONFIG_ERROR = "config_error"
    CONNECTION_ERROR = "connection_error"  
    AUTHENTICATION_ERROR = "authentication_error"
    SERVICE_ERROR = "service_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retriable_errors: List[ErrorType] = None
    
    def __post_init__(self):
        if self.retriable_errors is None:
            self.retriable_errors = [
                ErrorType.CONNECTION_ERROR,
                ErrorType.TIMEOUT_ERROR,
                ErrorType.RATE_LIMIT_ERROR,
                ErrorType.SERVICE_ERROR,
            ]


@dataclass
class ErrorContext:
    """错误上下文信息"""
    error_type: ErrorType
    original_exception: Exception
    attempt: int
    timestamp: float
    config: Optional[Union[ChatConfig, EmbeddingConfig]] = None
    additional_info: Optional[Dict[str, Any]] = None


class LangChainError(Exception):
    """LangChain相关错误基类"""
    
    def __init__(self, message: str, error_type: ErrorType, original_exception: Exception = None):
        super().__init__(message)
        self.error_type = error_type
        self.original_exception = original_exception
        self.timestamp = time.time()


class ConfigurationError(LangChainError):
    """配置错误"""
    
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message, ErrorType.CONFIG_ERROR, original_exception)


class ConnectionError(LangChainError):
    """连接错误"""
    
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message, ErrorType.CONNECTION_ERROR, original_exception)


class AuthenticationError(LangChainError):
    """认证错误"""
    
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message, ErrorType.AUTHENTICATION_ERROR, original_exception)


class ServiceError(LangChainError):
    """服务错误"""
    
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message, ErrorType.SERVICE_ERROR, original_exception)


class TimeoutError(LangChainError):
    """超时错误"""
    
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message, ErrorType.TIMEOUT_ERROR, original_exception)


class RateLimitError(LangChainError):
    """速率限制错误"""
    
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message, ErrorType.RATE_LIMIT_ERROR, original_exception)


class ErrorClassifier:
    """错误分类器
    
    负责将原始异常分类为特定的错误类型。
    """
    
    def __init__(self):
        """初始化错误分类器"""
        self.classification_rules = {
            # 连接错误
            ErrorType.CONNECTION_ERROR: [
                "ConnectionError",
                "ConnectTimeout", 
                "ReadTimeout",
                "HTTPSConnectionPool",
                "connection refused",
                "network is unreachable",
                "name resolution failed",
            ],
            
            # 认证错误
            ErrorType.AUTHENTICATION_ERROR: [
                "401",
                "unauthorized",
                "invalid api key",
                "authentication failed",
                "access denied",
            ],
            
            # 服务错误
            ErrorType.SERVICE_ERROR: [
                "500",
                "502",
                "503", 
                "504",
                "internal server error",
                "bad gateway",
                "service unavailable",
                "gateway timeout",
            ],
            
            # 超时错误
            ErrorType.TIMEOUT_ERROR: [
                "timeout",
                "timed out",
                "request timeout",
                "read timeout",
            ],
            
            # 速率限制错误
            ErrorType.RATE_LIMIT_ERROR: [
                "429",
                "rate limit",
                "too many requests",
                "quota exceeded",
            ],
            
            # 配置错误
            ErrorType.CONFIG_ERROR: [
                "invalid model",
                "model not found",
                "invalid parameter",
                "missing required parameter",
                "configuration error",
            ],
        }
    
    def classify_error(self, exception: Exception) -> ErrorType:
        """分类错误
        
        Args:
            exception: 原始异常
        
        Returns:
            ErrorType: 错误类型
        """
        error_message = str(exception).lower()
        exception_type = type(exception).__name__
        
        # 检查异常类型和消息
        for error_type, patterns in self.classification_rules.items():
            for pattern in patterns:
                if pattern.lower() in error_message or pattern.lower() in exception_type.lower():
                    return error_type
        
        # 如果无法分类，返回未知错误
        return ErrorType.UNKNOWN_ERROR


class RetryHandler:
    """重试处理器
    
    实现指数退避重试策略。
    """
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        """初始化重试处理器
        
        Args:
            retry_config: 重试配置，如果为None则使用默认配置
        """
        self.retry_config = retry_config or RetryConfig()
        self.error_classifier = ErrorClassifier()
    
    def should_retry(self, error_type: ErrorType, attempt: int) -> bool:
        """判断是否应该重试
        
        Args:
            error_type: 错误类型
            attempt: 当前尝试次数
        
        Returns:
            bool: 是否应该重试
        """
        if attempt >= self.retry_config.max_retries:
            return False
        
        return error_type in self.retry_config.retriable_errors
    
    def calculate_delay(self, attempt: int) -> float:
        """计算重试延迟时间
        
        Args:
            attempt: 当前尝试次数
        
        Returns:
            float: 延迟时间（秒）
        """
        # 指数退避算法
        delay = self.retry_config.base_delay * (self.retry_config.backoff_factor ** attempt)
        delay = min(delay, self.retry_config.max_delay)
        
        # 添加抖动
        if self.retry_config.jitter:
            jitter = random.uniform(0.1, 0.3) * delay
            delay += jitter
        
        return delay
    
    def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """执行函数并在失败时重试
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
        
        Returns:
            Any: 函数执行结果
        
        Raises:
            LangChainError: 重试耗尽后抛出的错误
        """
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                error_type = self.error_classifier.classify_error(e)
                
                logger.warning(
                    f"函数执行失败 (尝试 {attempt + 1}/{self.retry_config.max_retries + 1}): "
                    f"{type(e).__name__}: {e}"
                )
                
                # 检查是否应该重试
                if not self.should_retry(error_type, attempt):
                    break
                
                # 计算延迟时间
                if attempt < self.retry_config.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.info(f"等待 {delay:.2f} 秒后重试...")
                    time.sleep(delay)
        
        # 重试耗尽，抛出最后的异常
        error_type = self.error_classifier.classify_error(last_exception)
        raise self._convert_to_langchain_error(last_exception, error_type)
    
    def _convert_to_langchain_error(self, exception: Exception, error_type: ErrorType) -> LangChainError:
        """将原始异常转换为LangChain错误"""
        error_classes = {
            ErrorType.CONFIG_ERROR: ConfigurationError,
            ErrorType.CONNECTION_ERROR: ConnectionError,
            ErrorType.AUTHENTICATION_ERROR: AuthenticationError,
            ErrorType.SERVICE_ERROR: ServiceError,
            ErrorType.TIMEOUT_ERROR: TimeoutError,
            ErrorType.RATE_LIMIT_ERROR: RateLimitError,
        }
        
        error_class = error_classes.get(error_type, LangChainError)
        return error_class(str(exception), original_exception=exception)


class FallbackManager:
    """备用方案管理器
    
    管理各种备用方案策略。
    """
    
    def __init__(self):
        """初始化备用方案管理器"""
        self.config_manager = get_config_manager()
        self.fallback_configs = {
            "chat": [
                ChatConfig(provider="openai", model_name="gpt-3.5-turbo"),
                ChatConfig(provider="together", model_name="meta-llama/Llama-2-7b-chat-hf"),
            ],
            "embedding": [
                EmbeddingConfig(provider="huggingface", model_name="sentence-transformers/all-MiniLM-L6-v2"),
                EmbeddingConfig(provider="huggingface", model_name="sentence-transformers/all-mpnet-base-v2"),
            ],
        }
    
    def get_fallback_chat_config(self, failed_config: ChatConfig) -> Optional[ChatConfig]:
        """获取Chat服务的备用配置
        
        Args:
            failed_config: 失败的配置
        
        Returns:
            Optional[ChatConfig]: 备用配置，如果没有合适的备用配置则返回None
        """
        for fallback_config in self.fallback_configs["chat"]:
            if fallback_config.provider != failed_config.provider:
                # 继承一些配置
                fallback_config.temperature = failed_config.temperature
                fallback_config.timeout = failed_config.timeout
                fallback_config.max_retries = failed_config.max_retries
                return fallback_config
        
        return None
    
    def get_fallback_embedding_config(self, failed_config: EmbeddingConfig) -> Optional[EmbeddingConfig]:
        """获取Embedding服务的备用配置
        
        Args:
            failed_config: 失败的配置
        
        Returns:
            Optional[EmbeddingConfig]: 备用配置，如果没有合适的备用配置则返回None
        """
        for fallback_config in self.fallback_configs["embedding"]:
            if fallback_config.provider != failed_config.provider:
                # 继承一些配置
                fallback_config.batch_size = failed_config.batch_size
                fallback_config.normalize = failed_config.normalize
                fallback_config.timeout = failed_config.timeout
                return fallback_config
        
        return None
    
    def register_fallback_config(
        self,
        service_type: str,
        config: Union[ChatConfig, EmbeddingConfig]
    ) -> None:
        """注册备用配置
        
        Args:
            service_type: 服务类型 ("chat" 或 "embedding")
            config: 备用配置
        """
        if service_type not in self.fallback_configs:
            self.fallback_configs[service_type] = []
        
        self.fallback_configs[service_type].append(config)
        logger.info(f"注册 {service_type} 备用配置: {config.provider}/{config.model_name}")


class GracefulDegradation:
    """优雅降级处理器
    
    在服务不可用时提供降级服务。
    """
    
    def __init__(self):
        """初始化优雅降级处理器"""
        pass
    
    def get_mock_chat_response(self, input_text: str) -> str:
        """获取模拟的Chat响应
        
        Args:
            input_text: 输入文本
        
        Returns:
            str: 模拟响应
        """
        logger.warning("使用模拟Chat响应（降级模式）")
        return f"[模拟响应] 收到您的消息: {input_text[:100]}..."
    
    def get_mock_embeddings(self, texts: List[str], dimensions: int = 768) -> List[List[float]]:
        """获取模拟的嵌入向量
        
        Args:
            texts: 文本列表
            dimensions: 嵌入维度
        
        Returns:
            List[List[float]]: 模拟嵌入向量列表
        """
        logger.warning("使用模拟嵌入向量（降级模式）")
        import hashlib
        
        embeddings = []
        for text in texts:
            # 使用文本哈希生成确定性的模拟嵌入
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            
            # 将哈希字节转换为浮点数
            embedding = []
            for i in range(dimensions):
                byte_index = i % len(hash_bytes)
                value = (hash_bytes[byte_index] / 255.0) * 2 - 1  # 标准化到[-1, 1]
                embedding.append(value)
            
            embeddings.append(embedding)
        
        return embeddings


def with_error_handling(
    retry_config: Optional[RetryConfig] = None,
    enable_fallback: bool = True,
    enable_degradation: bool = False
):
    """错误处理装饰器
    
    Args:
        retry_config: 重试配置
        enable_fallback: 是否启用备用方案
        enable_degradation: 是否启用优雅降级
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry_handler = RetryHandler(retry_config)
            fallback_manager = FallbackManager()
            degradation = GracefulDegradation()
            
            try:
                return retry_handler.execute_with_retry(func, *args, **kwargs)
            
            except LangChainError as e:
                logger.error(f"函数 {func.__name__} 执行失败: {e}")
                
                # 尝试备用方案
                if enable_fallback:
                    try:
                        return _execute_with_fallback(
                            func, fallback_manager, *args, **kwargs
                        )
                    except Exception as fallback_error:
                        logger.error(f"备用方案也失败: {fallback_error}")
                
                # 尝试优雅降级
                if enable_degradation:
                    return _execute_with_degradation(
                        func, degradation, *args, **kwargs
                    )
                
                # 所有方案都失败，重新抛出原始错误
                raise e
        
        return wrapper
    return decorator


def _execute_with_fallback(func: Callable, fallback_manager: FallbackManager, *args, **kwargs):
    """使用备用方案执行函数"""
    # 这里需要根据具体的函数类型来实现备用逻辑
    # 这是一个简化的示例实现
    logger.info("尝试使用备用方案...")
    raise NotImplementedError("备用方案执行逻辑需要根据具体函数实现")


def _execute_with_degradation(func: Callable, degradation: GracefulDegradation, *args, **kwargs):
    """使用优雅降级执行函数"""
    # 这里需要根据具体的函数类型来实现降级逻辑
    # 这是一个简化的示例实现
    logger.info("使用优雅降级模式...")
    
    func_name = func.__name__
    if "chat" in func_name.lower():
        # 假设是Chat相关函数
        input_text = str(args[0]) if args else "默认输入"
        return degradation.get_mock_chat_response(input_text)
    elif "embed" in func_name.lower():
        # 假设是Embedding相关函数
        texts = args[0] if args and isinstance(args[0], list) else ["默认文本"]
        return degradation.get_mock_embeddings(texts)
    else:
        raise NotImplementedError("不支持的函数类型降级")


# 全局错误处理器实例
_global_retry_handler: Optional[RetryHandler] = None
_global_fallback_manager: Optional[FallbackManager] = None
_global_degradation: Optional[GracefulDegradation] = None


def get_retry_handler() -> RetryHandler:
    """获取全局重试处理器"""
    global _global_retry_handler
    if _global_retry_handler is None:
        _global_retry_handler = RetryHandler()
    return _global_retry_handler


def get_fallback_manager() -> FallbackManager:
    """获取全局备用方案管理器"""
    global _global_fallback_manager
    if _global_fallback_manager is None:
        _global_fallback_manager = FallbackManager()
    return _global_fallback_manager


def get_degradation() -> GracefulDegradation:
    """获取全局优雅降级处理器"""
    global _global_degradation
    if _global_degradation is None:
        _global_degradation = GracefulDegradation()
    return _global_degradation