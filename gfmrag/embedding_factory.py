"""
Embedding模型工厂模块

该模块专门处理文本嵌入模型的创建和管理，支持多种Embedding服务提供商。
"""

import logging
from typing import Any, Dict, Optional, Union

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from gfmrag.config_manager import EmbeddingConfig, get_config_manager
from gfmrag.providers import OpenAIProvider, ThirdPartyProvider


logger = logging.getLogger(__name__)


class EmbeddingModelFactory:
    """Embedding模型工厂
    
    专门负责创建和管理各种文本嵌入模型。
    """
    
    def __init__(self):
        """初始化Embedding模型工厂"""
        self.config_manager = get_config_manager()
        self._model_cache: Dict[str, Any] = {}
        self._providers = {
            "openai": OpenAIProvider(),
            "third-party": ThirdPartyProvider(),
        }
    
    def create_embedding_model(
        self,
        config: Optional[EmbeddingConfig] = None,
        **kwargs: Any
    ) -> Union[OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformer]:
        """创建Embedding模型实例
        
        Args:
            config: Embedding配置，如果为None则使用默认配置
            **kwargs: 额外的模型参数
        
        Returns:
            Embedding模型实例
        """
        if config is None:
            config = self.config_manager.get_embedding_config()
        
        # 验证配置
        self._validate_config(config)
        
        # 生成缓存键
        cache_key = self._generate_cache_key(config)
        
        # 检查缓存
        if cache_key in self._model_cache:
            logger.debug(f"从缓存返回Embedding模型: {config.provider}/{config.model_name}")
            return self._model_cache[cache_key]
        
        try:
            model = self._create_model_impl(config, **kwargs)
            
            # 测试模型
            self._test_model(model, config)
            
            # 缓存模型
            self._model_cache[cache_key] = model
            
            logger.info(f"成功创建Embedding模型: {config.provider}/{config.model_name}")
            return model
            
        except Exception as e:
            logger.error(f"创建Embedding模型失败: {config.provider}/{config.model_name}, 错误: {e}")
            
            # 尝试备用方案
            if self.config_manager.global_config.fallback_enabled:
                return self._create_fallback_model(config, **kwargs)
            
            raise
    
    def _create_model_impl(
        self,
        config: EmbeddingConfig,
        **kwargs: Any
    ) -> Union[OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformer]:
        """Embedding模型创建的具体实现"""
        
        if config.provider in ["openai", "third-party"]:
            return self._create_openai_compatible_model(config, **kwargs)
        elif config.provider in ["huggingface", "sentence-transformers"]:
            return self._create_huggingface_model(config, **kwargs)
        elif config.provider == "nvidia":
            return self._create_nvidia_model(config, **kwargs)
        else:
            # 默认使用HuggingFace/SentenceTransformers
            logger.warning(f"未知的Embedding提供商 {config.provider}，使用HuggingFace模型")
            return self._create_huggingface_model(config, **kwargs)
    
    def _create_openai_compatible_model(
        self,
        config: EmbeddingConfig,
        **kwargs: Any
    ) -> OpenAIEmbeddings:
        """创建OpenAI兼容的Embedding模型"""
        provider = self._providers.get(config.provider)
        if not provider:
            provider = self._providers["openai" if config.provider == "openai" else "third-party"]
        
        return provider.initialize_embedding(config, **kwargs)
    
    def _create_huggingface_model(
        self,
        config: EmbeddingConfig,
        **kwargs: Any
    ) -> HuggingFaceEmbeddings:
        """创建HuggingFace Embedding模型"""
        model_kwargs = {
            "model_name": config.model_name,
            "encode_kwargs": {
                "normalize_embeddings": config.normalize,
                "batch_size": config.batch_size,
            },
            **kwargs
        }
        
        logger.info(f"初始化HuggingFace Embedding模型: {config.model_name}")
        return HuggingFaceEmbeddings(**model_kwargs)
    
    def _create_nvidia_model(
        self,
        config: EmbeddingConfig,
        **kwargs: Any
    ) -> SentenceTransformer:
        """创建NVIDIA Embedding模型（使用SentenceTransformer）"""
        from gfmrag.text_emb_models import NVEmbedV2
        
        model_kwargs = {
            "text_emb_model_name": config.model_name,
            "normalize": config.normalize,
            "batch_size": config.batch_size,
            "query_instruct": config.query_instruct,
            "passage_instruct": config.passage_instruct,
            **kwargs
        }
        
        logger.info(f"初始化NVIDIA Embedding模型: {config.model_name}")
        return NVEmbedV2(**model_kwargs)
    
    def create_sentence_transformer_model(
        self,
        config: Optional[EmbeddingConfig] = None,
        **kwargs: Any
    ) -> SentenceTransformer:
        """直接创建SentenceTransformer模型
        
        Args:
            config: Embedding配置
            **kwargs: 额外参数
        
        Returns:
            SentenceTransformer模型实例
        """
        if config is None:
            config = self.config_manager.get_embedding_config()
        
        model_kwargs = {
            "trust_remote_code": True,
            **kwargs
        }
        
        if hasattr(config, 'model_kwargs') and config.model_kwargs:
            model_kwargs.update(config.model_kwargs)
        
        logger.info(f"初始化SentenceTransformer模型: {config.model_name}")
        return SentenceTransformer(config.model_name, **model_kwargs)
    
    def create_custom_embedding_model(
        self,
        model_class: type,
        config: Optional[EmbeddingConfig] = None,
        **kwargs: Any
    ) -> Any:
        """创建自定义Embedding模型
        
        Args:
            model_class: 模型类
            config: Embedding配置
            **kwargs: 额外参数
        
        Returns:
            自定义模型实例
        """
        if config is None:
            config = self.config_manager.get_embedding_config()
        
        model_kwargs = {
            "text_emb_model_name": config.model_name,
            "normalize": config.normalize,
            "batch_size": config.batch_size,
            **kwargs
        }
        
        logger.info(f"初始化自定义Embedding模型: {model_class.__name__}")
        return model_class(**model_kwargs)
    
    def _validate_config(self, config: EmbeddingConfig) -> None:
        """验证Embedding配置"""
        if not config.provider:
            raise ValueError("provider不能为空")
        if not config.model_name:
            raise ValueError("model_name不能为空")
        if config.batch_size <= 0:
            raise ValueError("batch_size必须大于0")
        if config.timeout <= 0:
            raise ValueError("timeout必须大于0")
    
    def _test_model(self, model: Any, config: EmbeddingConfig) -> None:
        """测试Embedding模型"""
        try:
            # 测试embedding功能
            test_texts = ["Hello, world!", "This is a test."]
            
            if hasattr(model, 'embed_documents'):
                # LangChain Embedding模型
                embeddings = model.embed_documents(test_texts)
            elif hasattr(model, 'encode'):
                # SentenceTransformer模型
                embeddings = model.encode(test_texts)
            else:
                logger.warning(f"无法测试模型 {type(model).__name__}：不支持的接口")
                return
            
            if embeddings and len(embeddings) == len(test_texts):
                logger.debug(f"Embedding模型测试成功: {config.provider}/{config.model_name}")
            else:
                logger.warning(f"Embedding模型测试失败：输出不匹配")
                
        except Exception as e:
            logger.warning(f"Embedding模型测试失败: {e}")
    
    def _generate_cache_key(self, config: EmbeddingConfig) -> str:
        """生成缓存键"""
        return f"embedding:{config.provider}:{config.model_name}:{hash(str(config))}"
    
    def _create_fallback_model(
        self,
        config: EmbeddingConfig,
        **kwargs: Any
    ) -> HuggingFaceEmbeddings:
        """创建备用Embedding模型"""
        logger.warning(f"使用备用Embedding模型替代 {config.provider}/{config.model_name}")
        
        fallback_config = EmbeddingConfig(
            provider="huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=config.batch_size,
            normalize=config.normalize,
        )
        
        return self._create_huggingface_model(fallback_config, **kwargs)
    
    def clear_cache(self) -> None:
        """清空模型缓存"""
        self._model_cache.clear()
        logger.info("已清空Embedding模型缓存")
    
    def get_cached_models(self) -> Dict[str, str]:
        """获取缓存的模型信息"""
        return {k: type(v).__name__ for k, v in self._model_cache.items()}
    
    def get_supported_providers(self) -> list[str]:
        """获取支持的提供商列表"""
        return [
            "openai",
            "third-party", 
            "huggingface",
            "sentence-transformers",
            "nvidia",
        ]
    
    def benchmark_model(
        self,
        config: Optional[EmbeddingConfig] = None,
        test_texts: Optional[list[str]] = None,
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """性能基准测试
        
        Args:
            config: Embedding配置
            test_texts: 测试文本，如果为None则使用默认测试文本
            num_runs: 测试运行次数
        
        Returns:
            性能测试结果
        """
        if config is None:
            config = self.config_manager.get_embedding_config()
        
        if test_texts is None:
            test_texts = [
                "This is a test sentence for benchmarking.",
                "Another sentence to test the embedding model performance.",
                "Embedding models are used to convert text into vector representations.",
            ]
        
        import time
        
        try:
            model = self.create_embedding_model(config)
            
            # 预热
            if hasattr(model, 'embed_documents'):
                model.embed_documents(test_texts[:1])
            elif hasattr(model, 'encode'):
                model.encode(test_texts[:1])
            
            # 性能测试
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                
                if hasattr(model, 'embed_documents'):
                    embeddings = model.embed_documents(test_texts)
                elif hasattr(model, 'encode'):
                    embeddings = model.encode(test_texts)
                else:
                    raise ValueError("不支持的模型接口")
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            throughput = len(test_texts) / avg_time
            
            return {
                "provider": config.provider,
                "model_name": config.model_name,
                "avg_time": avg_time,
                "throughput": throughput,
                "embedding_dim": len(embeddings[0]) if embeddings else 0,
                "test_texts_count": len(test_texts),
                "runs": num_runs,
            }
            
        except Exception as e:
            logger.error(f"性能测试失败: {e}")
            return {"error": str(e)}


# 工具函数
def create_embedding_model(
    provider: str,
    model_name: str,
    **kwargs: Any
) -> Union[OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformer]:
    """便捷函数：创建Embedding模型
    
    Args:
        provider: 服务提供商
        model_name: 模型名称
        **kwargs: 额外参数
    
    Returns:
        Embedding模型实例
    """
    factory = EmbeddingModelFactory()
    config = EmbeddingConfig(provider=provider, model_name=model_name, **kwargs)
    return factory.create_embedding_model(config)