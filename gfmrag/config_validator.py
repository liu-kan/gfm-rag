"""
配置验证机制模块

该模块实现了全面的配置验证功能，包括语法验证、语义验证、
连接验证和权限验证等多级验证机制。
"""

import logging
import re
import requests
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from dataclasses import dataclass

from gfmrag.config_manager import ChatConfig, EmbeddingConfig, GlobalConfig


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """验证结果数据类"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    config_type: str
    
    def add_error(self, message: str) -> None:
        """添加错误信息"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """添加警告信息"""
        self.warnings.append(message)
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """合并验证结果"""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            config_type=f"{self.config_type}+{other.config_type}"
        )


class ConfigValidator:
    """配置验证器
    
    提供多级配置验证功能，确保配置的正确性和可用性。
    """
    
    def __init__(self):
        """初始化配置验证器"""
        self.logger = logging.getLogger(__name__)
        
        # 支持的提供商列表
        self.supported_providers = {
            "chat": ["openai", "nvidia", "together", "ollama", "llama.cpp", "third-party"],
            "embedding": ["openai", "third-party", "huggingface", "sentence-transformers", "nvidia"],
        }
        
        # 模型名称正则表达式
        self.model_patterns = {
            "openai_chat": r"^(gpt-|o1-)",
            "openai_embedding": r"^text-embedding-",
            "nvidia": r"^(nvidia/|meta/)",
        }
    
    def validate_global_config(self, config: GlobalConfig) -> ValidationResult:
        """验证全局配置
        
        Args:
            config: 全局配置
        
        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(True, [], [], "global")
        
        try:
            # 验证默认提供商
            if not config.default_provider:
                result.add_error("default_provider不能为空")
            elif config.default_provider not in self.supported_providers["chat"]:
                result.add_error(f"不支持的默认提供商: {config.default_provider}")
            
            # 验证超时时间
            if config.timeout <= 0:
                result.add_error("timeout必须大于0")
            elif config.timeout > 300:
                result.add_warning("timeout设置过大，可能导致长时间等待")
            
            # 验证重试次数
            if config.max_retries < 0:
                result.add_error("max_retries必须大于等于0")
            elif config.max_retries > 10:
                result.add_warning("max_retries设置过大，可能导致长时间重试")
            
            # 验证日志级别
            valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if config.logging_level not in valid_log_levels:
                result.add_error(f"无效的日志级别: {config.logging_level}")
            
        except Exception as e:
            result.add_error(f"全局配置验证异常: {e}")
        
        return result
    
    def validate_chat_config(self, config: ChatConfig) -> ValidationResult:
        """验证Chat配置
        
        Args:
            config: Chat配置
        
        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(True, [], [], "chat")
        
        try:
            # 语法验证
            syntax_result = self._validate_chat_syntax(config)
            result = result.merge(syntax_result)
            
            # 语义验证
            semantic_result = self._validate_chat_semantics(config)
            result = result.merge(semantic_result)
            
            # 安全验证
            security_result = self._validate_chat_security(config)
            result = result.merge(security_result)
            
        except Exception as e:
            result.add_error(f"Chat配置验证异常: {e}")
        
        return result
    
    def validate_embedding_config(self, config: EmbeddingConfig) -> ValidationResult:
        """验证Embedding配置
        
        Args:
            config: Embedding配置
        
        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(True, [], [], "embedding")
        
        try:
            # 语法验证
            syntax_result = self._validate_embedding_syntax(config)
            result = result.merge(syntax_result)
            
            # 语义验证
            semantic_result = self._validate_embedding_semantics(config)
            result = result.merge(semantic_result)
            
            # 安全验证
            security_result = self._validate_embedding_security(config)
            result = result.merge(security_result)
            
        except Exception as e:
            result.add_error(f"Embedding配置验证异常: {e}")
        
        return result
    
    def _validate_chat_syntax(self, config: ChatConfig) -> ValidationResult:
        """验证Chat配置语法"""
        result = ValidationResult(True, [], [], "chat_syntax")
        
        # 基础字段验证
        if not config.provider:
            result.add_error("provider不能为空")
        elif config.provider not in self.supported_providers["chat"]:
            result.add_error(f"不支持的Chat提供商: {config.provider}")
        
        if not config.model_name:
            result.add_error("model_name不能为空")
        
        # 数值范围验证
        if config.temperature < 0 or config.temperature > 2:
            result.add_error("temperature必须在0-2之间")
        
        if config.timeout <= 0:
            result.add_error("timeout必须大于0")
        
        if config.max_retries < 0:
            result.add_error("max_retries必须大于等于0")
        
        if config.max_tokens is not None and config.max_tokens <= 0:
            result.add_error("max_tokens必须大于0")
        
        if config.top_p is not None and (config.top_p < 0 or config.top_p > 1):
            result.add_error("top_p必须在0-1之间")
        
        # URL格式验证
        if config.base_url and not self._is_valid_url(config.base_url):
            result.add_error(f"无效的base_url格式: {config.base_url}")
        
        return result
    
    def _validate_chat_semantics(self, config: ChatConfig) -> ValidationResult:
        """验证Chat配置语义"""
        result = ValidationResult(True, [], [], "chat_semantics")
        
        # 提供商特定验证
        if config.provider == "openai":
            if not re.match(self.model_patterns["openai_chat"], config.model_name):
                result.add_warning(f"OpenAI Chat模型名称可能不正确: {config.model_name}")
        
        elif config.provider == "nvidia":
            if not re.match(self.model_patterns["nvidia"], config.model_name):
                result.add_warning(f"NVIDIA模型名称可能不正确: {config.model_name}")
        
        elif config.provider == "third-party":
            if not config.base_url:
                result.add_error("第三方服务必须提供base_url")
        
        elif config.provider == "llama.cpp":
            if not config.model_name.endswith(('.gguf', '.bin')):
                result.add_warning("llama.cpp通常使用.gguf或.bin格式的模型文件")
        
        # 参数组合验证
        if config.temperature == 0 and config.top_p is not None:
            result.add_warning("temperature为0时，top_p参数可能不会生效")
        
        return result
    
    def _validate_chat_security(self, config: ChatConfig) -> ValidationResult:
        """验证Chat配置安全性"""
        result = ValidationResult(True, [], [], "chat_security")
        
        # API密钥验证
        if config.provider in ["openai", "nvidia", "together"]:
            if not config.api_key:
                result.add_warning(f"{config.provider}服务通常需要API密钥")
            elif len(config.api_key) < 10:
                result.add_warning("API密钥长度可能不正确")
        
        # URL安全验证
        if config.base_url:
            parsed_url = urlparse(config.base_url)
            if parsed_url.scheme not in ['http', 'https']:
                result.add_error("base_url必须使用http或https协议")
            if parsed_url.scheme == 'http':
                result.add_warning("使用http协议可能存在安全风险，建议使用https")
        
        return result
    
    def _validate_embedding_syntax(self, config: EmbeddingConfig) -> ValidationResult:
        """验证Embedding配置语法"""
        result = ValidationResult(True, [], [], "embedding_syntax")
        
        # 基础字段验证
        if not config.provider:
            result.add_error("provider不能为空")
        elif config.provider not in self.supported_providers["embedding"]:
            result.add_error(f"不支持的Embedding提供商: {config.provider}")
        
        if not config.model_name:
            result.add_error("model_name不能为空")
        
        # 数值范围验证
        if config.batch_size <= 0:
            result.add_error("batch_size必须大于0")
        elif config.batch_size > 1000:
            result.add_warning("batch_size设置过大，可能导致内存不足")
        
        if config.timeout <= 0:
            result.add_error("timeout必须大于0")
        
        if config.dimensions is not None and config.dimensions <= 0:
            result.add_error("dimensions必须大于0")
        
        # URL格式验证
        if config.base_url and not self._is_valid_url(config.base_url):
            result.add_error(f"无效的base_url格式: {config.base_url}")
        
        return result
    
    def _validate_embedding_semantics(self, config: EmbeddingConfig) -> ValidationResult:
        """验证Embedding配置语义"""
        result = ValidationResult(True, [], [], "embedding_semantics")
        
        # 提供商特定验证
        if config.provider == "openai":
            if not re.match(self.model_patterns["openai_embedding"], config.model_name):
                result.add_warning(f"OpenAI Embedding模型名称可能不正确: {config.model_name}")
        
        elif config.provider == "nvidia":
            if "embed" not in config.model_name.lower():
                result.add_warning(f"NVIDIA Embedding模型名称可能不正确: {config.model_name}")
        
        elif config.provider == "third-party":
            if not config.base_url:
                result.add_error("第三方服务必须提供base_url")
        
        return result
    
    def _validate_embedding_security(self, config: EmbeddingConfig) -> ValidationResult:
        """验证Embedding配置安全性"""
        result = ValidationResult(True, [], [], "embedding_security")
        
        # API密钥验证
        if config.provider == "openai":
            if not config.api_key:
                result.add_warning("OpenAI服务通常需要API密钥")
        
        # URL安全验证
        if config.base_url:
            parsed_url = urlparse(config.base_url)
            if parsed_url.scheme not in ['http', 'https']:
                result.add_error("base_url必须使用http或https协议")
            if parsed_url.scheme == 'http':
                result.add_warning("使用http协议可能存在安全风险，建议使用https")
        
        return result
    
    def validate_connection(self, config: Union[ChatConfig, EmbeddingConfig]) -> ValidationResult:
        """验证服务连接
        
        Args:
            config: 服务配置
        
        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(True, [], [], "connection")
        
        if not config.base_url:
            result.add_warning("未提供base_url，跳过连接测试")
            return result
        
        try:
            # 发送健康检查请求
            response = requests.get(
                config.base_url.rstrip('/'),
                timeout=min(config.timeout or 10, 10),
                headers={"User-Agent": "GFM-RAG-Validator/1.0"}
            )
            
            if response.status_code < 500:
                result.add_warning("服务连接正常")
            else:
                result.add_error(f"服务返回错误状态码: {response.status_code}")
            
        except requests.exceptions.Timeout:
            result.add_error("连接超时")
        except requests.exceptions.ConnectionError:
            result.add_error("连接失败")
        except Exception as e:
            result.add_error(f"连接测试异常: {e}")
        
        return result
    
    def validate_permissions(self, config: Union[ChatConfig, EmbeddingConfig]) -> ValidationResult:
        """验证API权限
        
        Args:
            config: 服务配置
        
        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(True, [], [], "permissions")
        
        if not config.api_key:
            result.add_warning("未提供API密钥，跳过权限验证")
            return result
        
        try:
            # 这里可以添加具体的权限验证逻辑
            # 例如调用API的列表模型端点来验证权限
            if config.provider == "openai" and config.base_url:
                models_url = f"{config.base_url.rstrip('/')}/models"
                headers = {
                    "Authorization": f"Bearer {config.api_key}",
                    "User-Agent": "GFM-RAG-Validator/1.0"
                }
                
                response = requests.get(models_url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    result.add_warning("API权限验证通过")
                elif response.status_code == 401:
                    result.add_error("API密钥无效")
                elif response.status_code == 403:
                    result.add_error("API权限不足")
                else:
                    result.add_warning(f"权限验证返回状态码: {response.status_code}")
            
        except Exception as e:
            result.add_warning(f"权限验证异常: {e}")
        
        return result
    
    def _is_valid_url(self, url: str) -> bool:
        """检查URL格式是否有效"""
        try:
            parsed = urlparse(url)
            return parsed.scheme in ['http', 'https'] and parsed.netloc
        except:
            return False
    
    def comprehensive_validation(
        self,
        global_config: Optional[GlobalConfig] = None,
        chat_config: Optional[ChatConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        test_connections: bool = False,
        test_permissions: bool = False
    ) -> Dict[str, ValidationResult]:
        """全面配置验证
        
        Args:
            global_config: 全局配置
            chat_config: Chat配置
            embedding_config: Embedding配置
            test_connections: 是否测试连接
            test_permissions: 是否测试权限
        
        Returns:
            Dict[str, ValidationResult]: 各类型配置的验证结果
        """
        results = {}
        
        # 验证全局配置
        if global_config:
            results["global"] = self.validate_global_config(global_config)
        
        # 验证Chat配置
        if chat_config:
            results["chat"] = self.validate_chat_config(chat_config)
            
            if test_connections:
                conn_result = self.validate_connection(chat_config)
                results["chat_connection"] = conn_result
            
            if test_permissions:
                perm_result = self.validate_permissions(chat_config)
                results["chat_permissions"] = perm_result
        
        # 验证Embedding配置
        if embedding_config:
            results["embedding"] = self.validate_embedding_config(embedding_config)
            
            if test_connections:
                conn_result = self.validate_connection(embedding_config)
                results["embedding_connection"] = conn_result
            
            if test_permissions:
                perm_result = self.validate_permissions(embedding_config)
                results["embedding_permissions"] = perm_result
        
        return results
    
    def generate_validation_report(self, results: Dict[str, ValidationResult]) -> str:
        """生成验证报告
        
        Args:
            results: 验证结果字典
        
        Returns:
            str: 格式化的验证报告
        """
        report_lines = ["=" * 50, "配置验证报告", "=" * 50]
        
        total_errors = 0
        total_warnings = 0
        
        for config_type, result in results.items():
            report_lines.append(f"\n[{config_type.upper()}]")
            report_lines.append(f"状态: {'✓ 通过' if result.is_valid else '✗ 失败'}")
            
            if result.errors:
                report_lines.append("错误:")
                for error in result.errors:
                    report_lines.append(f"  - {error}")
                total_errors += len(result.errors)
            
            if result.warnings:
                report_lines.append("警告:")
                for warning in result.warnings:
                    report_lines.append(f"  - {warning}")
                total_warnings += len(result.warnings)
        
        report_lines.append(f"\n总结: {total_errors} 个错误, {total_warnings} 个警告")
        report_lines.append("=" * 50)
        
        return "\n".join(report_lines)