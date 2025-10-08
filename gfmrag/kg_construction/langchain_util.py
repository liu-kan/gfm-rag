"""
LangChain 模型初始化工具模块

该模块提供了初始化各种LangChain模型的工具函数，
支持环境变量配置和多种服务提供商。

支持的环境变量：
- GFMRAG_CHAT_PROVIDER: Chat服务提供商（如 'openai', 'third-party', 'ollama'）
- GFMRAG_CHAT_MODEL_NAME: Chat模型名称（如 'gpt-3.5-turbo', 'llama3'）
- GFMRAG_CHAT_BASE_URL: 第三方服务Base URL
- GFMRAG_CHAT_KEY: Chat服务API密钥（可选，空值表示无认证）

示例用法：

1. 使用环境变量：
   export GFMRAG_CHAT_PROVIDER="third-party"
   export GFMRAG_CHAT_MODEL_NAME="llama-2-7b-chat"
   export GFMRAG_CHAT_BASE_URL="http://localhost:8000/v1"
   # GFMRAG_CHAT_KEY 未设置，使用无认证模式
   
   model = init_langchain_model_from_env()

2. 混合使用：
   model = init_langchain_model("openai", "gpt-4", temperature=0.5)

3. 完全从参数：
   model = init_langchain_model(
       llm="third-party",
       model_name="custom-model", 
       base_url="http://localhost:8000/v1"
   )
"""

import os
import logging
from typing import Any, Optional

from langchain_community.chat_models import ChatLlamaCpp, ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether

from gfmrag.config_manager import get_config_manager, ChatConfig
from gfmrag.langchain_factory import LangChainModelFactory

logger = logging.getLogger(__name__)

# 创建全局模型工厂实例
_langchain_factory = LangChainModelFactory()


def init_langchain_model(
    llm: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 5,
    timeout: int = 60,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ChatOpenAI | ChatTogether | ChatOllama | ChatLlamaCpp:
    """
    Initialize a language model with enhanced environment variable support.
    
    This function now prioritizes function parameters over environment variables,
    enabling seamless integration with GFMRAG_* environment variables while
    allowing explicit parameter overrides.
    
    Parameter priority order:
    1. Explicit function parameters (highest priority)
    2. Environment variables via GFMRAG_* 
    3. Configuration defaults (lowest priority)
    
    Environment variables:
    - GFMRAG_CHAT_PROVIDER: Chat service provider 
    - GFMRAG_CHAT_MODEL_NAME: Chat model name
    - GFMRAG_CHAT_BASE_URL: Chat service base URL
    - GFMRAG_CHAT_KEY: Chat service API key
    
    :param llm: The LLM provider, if None will use environment variable or default
    :param model_name: The model name, if None will use environment variable or default
    :param temperature: Temperature for generation
    :param max_retries: Maximum number of retries
    :param timeout: Timeout in seconds
    :param base_url: Base URL for API (supports third-party services)
    :param api_key: API key (optional, will use environment variables if not provided)
    :param kwargs: Additional arguments passed to the model
    """
    try:
        # Get configuration manager to leverage environment variables
        config_manager = get_config_manager()
        
        # If no provider specified, use environment variable or get from config
        if llm is None:
            base_config = config_manager.get_chat_config()
            llm = base_config.provider
            
        # If no model_name specified, use environment variable or get from config  
        if model_name is None:
            base_config = config_manager.get_chat_config(llm)
            model_name = base_config.model_name
            
        # Get environment configuration
        env_config = config_manager.get_chat_config(llm)
        
        # Create final configuration with parameter priority
        # Parameters override environment variables
        final_config = ChatConfig(
            provider=llm,
            model_name=model_name,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
            base_url=base_url if base_url is not None else env_config.base_url,
            api_key=api_key if api_key is not None else env_config.api_key,
        )
        
        model = _langchain_factory.create_chat_model(final_config, **kwargs)
        logger.info(f"Successfully initialized {llm} model: {model_name}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to initialize {llm} model {model_name}: {e}")
        raise


def init_langchain_model_from_env(**kwargs: Any) -> ChatOpenAI | ChatTogether | ChatOllama | ChatLlamaCpp:
    """
    Initialize a language model entirely from environment variables.
    
    This function creates a chat model using only environment variables,
    making it ideal for containerized environments and CI/CD pipelines.
    
    Required environment variables:
    - GFMRAG_CHAT_PROVIDER: e.g., 'openai', 'third-party', 'ollama'
    - GFMRAG_CHAT_MODEL_NAME: e.g., 'gpt-3.5-turbo', 'llama3'
    
    Optional environment variables:
    - GFMRAG_CHAT_BASE_URL: for third-party services
    - GFMRAG_CHAT_KEY: API key (if required by the service)
    
    :param kwargs: Additional arguments passed to the model
    """
    logger.info("Creating language model from environment variables")
    return init_langchain_model(**kwargs)


# 保持向后兼容的原始实现（已弃用）
def init_langchain_model_legacy(
    llm: str,
    model_name: str,
    temperature: float = 0.0,
    max_retries: int = 5,
    timeout: int = 60,
    **kwargs: Any,
) -> ChatOpenAI | ChatTogether | ChatOllama | ChatLlamaCpp:
    """
    Legacy implementation of init_langchain_model.
    
    DEPRECATED: This function is deprecated and will be removed in future versions.
    Please use init_langchain_model() instead.
    """
    logger.warning(
        "init_langchain_model_legacy is deprecated. "
        "Please use init_langchain_model() for enhanced features."
    )
    if llm == "openai":
        # https://python.langchain.com/v0.1/docs/integrations/chat/openai/

        assert model_name.startswith("gpt-")
        return ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=model_name,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
            **kwargs,
        )
    elif llm == "nvidia":
        # https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints/

        return ChatNVIDIA(
            nvidia_api_key=os.environ.get("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1",
            model=model_name,
            temperature=temperature,
            **kwargs,
        )
    elif llm == "together":
        # https://python.langchain.com/v0.1/docs/integrations/chat/together/

        return ChatTogether(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            model=model_name,
            temperature=temperature,
            **kwargs,
        )
    elif llm == "ollama":
        # https://python.langchain.com/v0.1/docs/integrations/chat/ollama/

        return ChatOllama(model=model_name)  # e.g., 'llama3'
    elif llm == "llama.cpp":
        # https://python.langchain.com/v0.2/docs/integrations/chat/llamacpp/

        return ChatLlamaCpp(
            model_path=model_name, verbose=True
        )  # model_name is the model path (gguf file)
    else:
        # add any LLMs you want to use here using LangChain
        raise NotImplementedError(f"LLM '{llm}' not implemented yet.")
