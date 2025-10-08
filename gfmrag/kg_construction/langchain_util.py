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
    llm: str,
    model_name: str,
    temperature: float = 0.0,
    max_retries: int = 5,
    timeout: int = 60,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ChatOpenAI | ChatTogether | ChatOllama | ChatLlamaCpp:
    """
    Initialize a language model from the langchain library with enhanced configuration support.
    
    This function now supports third-party OpenAI-compatible services through base_url parameter
    and integrates with the unified configuration management system.
    
    :param llm: The LLM to use, e.g., 'openai', 'together', 'third-party'
    :param model_name: The model name to use, e.g., 'gpt-3.5-turbo'
    :param temperature: Temperature for generation
    :param max_retries: Maximum number of retries
    :param timeout: Timeout in seconds
    :param base_url: Base URL for API (supports third-party services)
    :param api_key: API key (optional, will use config manager if not provided)
    :param kwargs: Additional arguments passed to the model
    """
    try:
        # 使用新的工厂方法创建模型
        config = ChatConfig(
            provider=llm,
            model_name=model_name,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
            base_url=base_url,
            api_key=api_key,
        )
        
        model = _langchain_factory.create_chat_model(config, **kwargs)
        logger.info(f"Successfully initialized {llm} model: {model_name}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to initialize {llm} model {model_name}: {e}")
        raise


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
