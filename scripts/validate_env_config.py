#!/usr/bin/env python3
"""
GFMRAG环境变量配置验证脚本

该脚本用于验证GFMRAG环境变量配置的正确性，
包括语法验证、连接测试和兼容性检查。

使用方法:
    python scripts/validate_env_config.py
    
或者:
    python -m scripts.validate_env_config
"""

import os
import sys
import logging
from typing import Dict, List, Optional
from urllib.parse import urlparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gfmrag.config_manager import (
    ConfigurationManager, 
    get_config_manager,
    log_environment_config
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnvironmentConfigValidator:
    """环境变量配置验证器"""
    
    def __init__(self):
        self.config_manager = get_config_manager()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.success_count = 0
        self.total_checks = 0
    
    def validate_all(self) -> bool:
        """执行所有验证检查
        
        Returns:
            bool: 是否通过所有验证
        """
        logger.info("开始验证GFMRAG环境变量配置...")
        
        # 显示环境变量设置
        log_environment_config()
        
        # 执行各项检查
        self._check_environment_variables()
        self._check_config_syntax()
        self._check_provider_compatibility()
        self._check_url_validity()
        self._check_authentication_setup()
        
        # 显示结果
        self._display_results()
        
        return len(self.errors) == 0
    
    def _check_environment_variables(self) -> None:
        """检查环境变量设置"""
        logger.info("检查环境变量设置...")
        self.total_checks += 1
        
        env_vars = self.config_manager.get_environment_variables_summary()
        
        if not env_vars:
            self.warnings.append("未设置任何GFMRAG环境变量，将使用配置文件或默认值")
        else:
            self.success_count += 1
            logger.info(f"检测到 {len(env_vars)} 个GFMRAG环境变量")
    
    def _check_config_syntax(self) -> None:
        """检查配置语法"""
        logger.info("验证配置语法...")
        self.total_checks += 1
        
        try:
            # 验证chat配置
            chat_config = self.config_manager.get_chat_config()
            logger.debug(f"Chat配置: {chat_config.provider}/{chat_config.model_name}")
            
            # 验证embedding配置
            embedding_config = self.config_manager.get_embedding_config()
            logger.debug(f"Embedding配置: {embedding_config.provider}/{embedding_config.model_name}")
            
            # 使用内置验证
            validation_errors = self.config_manager.validate_config()
            
            if validation_errors:
                for category, errors in validation_errors.items():
                    for error in errors:
                        self.errors.append(f"{category}配置错误: {error}")
            else:
                self.success_count += 1
                logger.info("配置语法验证通过")
                
        except Exception as e:
            self.errors.append(f"配置语法验证失败: {e}")
    
    def _check_provider_compatibility(self) -> None:
        """检查提供商兼容性"""
        logger.info("检查提供商兼容性...")
        self.total_checks += 1
        
        try:
            chat_config = self.config_manager.get_chat_config()
            embedding_config = self.config_manager.get_embedding_config()
            
            # 检查已知提供商
            known_chat_providers = {
                "openai", "nvidia", "together", "anthropic", 
                "ollama", "llama.cpp", "third-party", "vllm"
            }
            known_embedding_providers = {
                "openai", "nvidia", "huggingface", "third-party"
            }
            
            if chat_config.provider not in known_chat_providers:
                self.warnings.append(f"未知的Chat提供商: {chat_config.provider}")
            
            if embedding_config.provider not in known_embedding_providers:
                self.warnings.append(f"未知的Embedding提供商: {embedding_config.provider}")
            
            # 检查第三方服务配置
            if chat_config.provider in ["third-party", "vllm"]:
                if not chat_config.base_url:
                    self.errors.append(f"第三方Chat服务 '{chat_config.provider}' 需要设置 GFMRAG_CHAT_BASE_URL")
            
            if embedding_config.provider == "third-party":
                if not embedding_config.base_url:
                    self.errors.append("第三方Embedding服务需要设置 GFMRAG_EMBEDDING_BASE_URL")
            
            if not self.errors:
                self.success_count += 1
                logger.info("提供商兼容性检查通过")
                
        except Exception as e:
            self.errors.append(f"提供商兼容性检查失败: {e}")
    
    def _check_url_validity(self) -> None:
        """检查URL有效性"""
        logger.info("检查URL有效性...")
        self.total_checks += 1
        
        try:
            chat_config = self.config_manager.get_chat_config()
            embedding_config = self.config_manager.get_embedding_config()
            
            # 检查Chat base_url
            if chat_config.base_url:
                if not self._is_valid_url(chat_config.base_url):
                    self.errors.append(f"无效的Chat base_url: {chat_config.base_url}")
            
            # 检查Embedding base_url
            if embedding_config.base_url:
                if not self._is_valid_url(embedding_config.base_url):
                    self.errors.append(f"无效的Embedding base_url: {embedding_config.base_url}")
            
            if not self.errors:
                self.success_count += 1
                logger.info("URL有效性检查通过")
                
        except Exception as e:
            self.errors.append(f"URL有效性检查失败: {e}")
    
    def _check_authentication_setup(self) -> None:
        """检查认证设置"""
        logger.info("检查认证设置...")
        self.total_checks += 1
        
        try:
            chat_config = self.config_manager.get_chat_config()
            embedding_config = self.config_manager.get_embedding_config()
            
            # 检查需要认证的服务
            auth_required_providers = {"openai", "nvidia", "together", "anthropic"}
            
            # Chat认证检查
            if chat_config.provider in auth_required_providers:
                if not chat_config.api_key:
                    self.warnings.append(
                        f"Chat服务 '{chat_config.provider}' 通常需要API密钥，"
                        "但未设置 GFMRAG_CHAT_KEY"
                    )
            elif chat_config.provider in ["ollama", "llama.cpp"]:
                if chat_config.api_key:
                    self.warnings.append(
                        f"Chat服务 '{chat_config.provider}' 通常不需要API密钥，"
                        "但设置了 GFMRAG_CHAT_KEY"
                    )
            
            # Embedding认证检查
            if embedding_config.provider in auth_required_providers:
                if not embedding_config.api_key:
                    self.warnings.append(
                        f"Embedding服务 '{embedding_config.provider}' 通常需要API密钥，"
                        "但未设置 GFMRAG_EMBEDDING_KEY"
                    )
            
            self.success_count += 1
            logger.info("认证设置检查完成")
            
        except Exception as e:
            self.errors.append(f"认证设置检查失败: {e}")
    
    def _is_valid_url(self, url: str) -> bool:
        """验证URL格式"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _display_results(self) -> None:
        """显示验证结果"""
        print("\n" + "="*60)
        print("GFMRAG环境变量配置验证结果")
        print("="*60)
        
        print(f"总检查项: {self.total_checks}")
        print(f"通过检查: {self.success_count}")
        print(f"错误数量: {len(self.errors)}")
        print(f"警告数量: {len(self.warnings)}")
        
        if self.errors:
            print("\n❌ 错误:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print("\n⚠️  警告:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ 所有检查通过！配置验证成功。")
        elif not self.errors:
            print("\n✅ 配置验证通过，但有一些警告需要注意。")
        else:
            print("\n❌ 配置验证失败，请修正错误后重试。")
        
        print("="*60)


def main():
    """主函数"""
    validator = EnvironmentConfigValidator()
    success = validator.validate_all()
    
    # 返回适当的退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()