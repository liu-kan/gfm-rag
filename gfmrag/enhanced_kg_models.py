"""
增强的知识图谱构建错误处理集成模块

该模块集成了Token管理、JSON解析容错、进程管理等功能，
提供统一的错误处理接口，用于优化OpenIE和NER模型的稳定性。
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from gfmrag.token_manager import get_token_manager, DocumentSize, TokenAllocation
from gfmrag.json_parser import get_json_extractor, ParseResult
from gfmrag.process_manager import batch_process_with_recovery
from gfmrag.error_handler import with_error_handling, RetryConfig

logger = logging.getLogger(__name__)


@dataclass
class KGProcessingConfig:
    """KG处理配置"""
    max_workers: int = 10
    checkpoint_dir: str = "./kg_checkpoints"
    max_retries: int = 3
    retry_delay: float = 2.0
    enable_adaptive_tokens: bool = True
    enable_json_repair: bool = True
    enable_recovery: bool = True
    batch_size: int = 20


class EnhancedNERModel:
    """增强的NER模型 - 集成错误处理功能"""
    
    def __init__(self, base_ner_model, config: Optional[KGProcessingConfig] = None):
        """初始化增强NER模型
        
        Args:
            base_ner_model: 原始NER模型实例
            config: 处理配置
        """
        self.base_model = base_ner_model
        self.config = config or KGProcessingConfig()
        self.token_manager = get_token_manager()
        self.json_extractor = get_json_extractor()
        
        # 性能统计
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'token_optimizations': 0,
            'json_repairs': 0,
            'average_processing_time': 0.0
        }
        
        logger.info("增强NER模型初始化完成")
    
    @with_error_handling(
        retry_config=RetryConfig(max_retries=3, base_delay=1.0),
        enable_fallback=True
    )
    def __call__(self, text: str) -> List[str]:
        """处理文本并提取命名实体
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 提取的命名实体列表
        """
        start_time = time.time()
        self.stats['total_calls'] += 1
        
        try:
            # 获取最优Token分配
            if self.config.enable_adaptive_tokens:
                allocation = self.token_manager.get_optimal_allocation(text, "ner")
                # 更新模型的max_tokens
                original_max_tokens = getattr(self.base_model, 'max_tokens', None)
                if hasattr(self.base_model, 'max_tokens'):
                    self.base_model.max_tokens = allocation.ner_tokens
                    self.stats['token_optimizations'] += 1
                    logger.debug(f"Token优化: {original_max_tokens} -> {allocation.ner_tokens}")
            
            # 检查是否需要分段处理
            max_tokens = getattr(self.base_model, 'max_tokens', 1024)
            needs_segmentation, num_segments = self.token_manager.suggest_segmentation(text, max_tokens)
            
            if needs_segmentation and num_segments > 1:
                logger.info(f"文本过长，分为 {num_segments} 段处理")
                return self._process_segmented_text(text, num_segments)
            else:
                return self._process_single_text(text)
                
        except Exception as e:
            self.stats['failed_calls'] += 1
            logger.error(f"NER处理失败: {e}")
            # 返回空列表作为降级处理
            return []
        finally:
            processing_time = time.time() - start_time
            self._update_processing_time(processing_time)
    
    def _process_single_text(self, text: str) -> List[str]:
        """处理单个文本"""
        try:
            # 调用原始模型
            result = self.base_model(text)
            self.stats['successful_calls'] += 1
            return result
            
        except Exception as e:
            # 如果是JSON解析错误，尝试使用增强解析器
            if "json" in str(e).lower() and self.config.enable_json_repair:
                logger.warning(f"检测到JSON解析错误，尝试修复: {e}")
                return self._repair_and_extract(str(e), text)
            else:
                raise
    
    def _process_segmented_text(self, text: str, num_segments: int) -> List[str]:
        """处理分段文本"""
        segments = self.token_manager.split_text_semantically(text, num_segments)
        all_entities = []
        
        for i, segment in enumerate(segments):
            logger.debug(f"处理第 {i+1}/{num_segments} 段")
            try:
                entities = self._process_single_text(segment)
                all_entities.extend(entities)
            except Exception as e:
                logger.warning(f"处理第 {i+1} 段失败: {e}")
                continue
        
        # 去重
        return list(set(all_entities))
    
    def _repair_and_extract(self, error_message: str, original_text: str) -> List[str]:
        """修复JSON并提取实体"""
        # 这里需要根据实际的错误获取响应内容
        # 简化实现，实际应用中需要更复杂的错误分析
        self.stats['json_repairs'] += 1
        logger.warning("使用降级处理，返回空实体列表")
        return []
    
    def _update_processing_time(self, processing_time: float):
        """更新平均处理时间"""
        total_calls = self.stats['total_calls']
        current_avg = self.stats['average_processing_time']
        self.stats['average_processing_time'] = (
            (current_avg * (total_calls - 1) + processing_time) / total_calls
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self.stats.copy()
        if stats['total_calls'] > 0:
            stats['success_rate'] = stats['successful_calls'] / stats['total_calls']
        else:
            stats['success_rate'] = 0.0
        return stats


class EnhancedOpenIEModel:
    """增强的OpenIE模型 - 集成错误处理功能"""
    
    def __init__(self, base_openie_model, config: Optional[KGProcessingConfig] = None):
        """初始化增强OpenIE模型
        
        Args:
            base_openie_model: 原始OpenIE模型实例
            config: 处理配置
        """
        self.base_model = base_openie_model
        self.config = config or KGProcessingConfig()
        self.token_manager = get_token_manager()
        self.json_extractor = get_json_extractor()
        
        # 性能统计
        self.stats = {
            'total_calls': 0,  
            'successful_calls': 0,
            'failed_calls': 0,
            'token_optimizations': 0,
            'json_repairs': 0,
            'segmentation_count': 0,
            'average_processing_time': 0.0
        }
        
        logger.info("增强OpenIE模型初始化完成")
    
    @with_error_handling(
        retry_config=RetryConfig(max_retries=3, base_delay=2.0),
        enable_fallback=True
    )
    def __call__(self, text: str) -> Dict[str, Any]:
        """处理文本并提取开放信息
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, Any]: 包含实体和三元组的字典
        """
        start_time = time.time()
        self.stats['total_calls'] += 1
        
        try:
            # 获取最优Token分配
            if self.config.enable_adaptive_tokens:
                allocation = self.token_manager.get_optimal_allocation(text, "openie")
                self._apply_token_allocation(allocation)
            
            # 检查是否需要分段处理
            max_ner_tokens = getattr(self.base_model, 'max_ner_tokens', 1024)
            max_triples_tokens = getattr(self.base_model, 'max_triples_tokens', 4096)
            
            # 使用较小的Token限制来判断是否分段
            max_tokens = min(max_ner_tokens, max_triples_tokens // 2)
            needs_segmentation, num_segments = self.token_manager.suggest_segmentation(text, max_tokens)
            
            if needs_segmentation and num_segments > 1:
                logger.info(f"文本过长，分为 {num_segments} 段处理")
                self.stats['segmentation_count'] += 1
                return self._process_segmented_text(text, num_segments)
            else:
                return self._process_single_text(text)
                
        except Exception as e:
            self.stats['failed_calls'] += 1
            logger.error(f"OpenIE处理失败: {e}")
            # 返回空结果作为降级处理
            return {
                "passage": text,
                "extracted_entities": [],
                "extracted_triples": []
            }
        finally:
            processing_time = time.time() - start_time
            self._update_processing_time(processing_time)
    
    def _apply_token_allocation(self, allocation: TokenAllocation):
        """应用Token分配"""
        original_ner = getattr(self.base_model, 'max_ner_tokens', None)
        original_triples = getattr(self.base_model, 'max_triples_tokens', None)
        
        if hasattr(self.base_model, 'max_ner_tokens'):
            self.base_model.max_ner_tokens = allocation.ner_tokens
        if hasattr(self.base_model, 'max_triples_tokens'):
            self.base_model.max_triples_tokens = allocation.triples_tokens
        
        self.stats['token_optimizations'] += 1
        logger.debug(f"Token优化 - NER: {original_ner} -> {allocation.ner_tokens}, "
                    f"Triples: {original_triples} -> {allocation.triples_tokens}")
    
    def _process_single_text(self, text: str) -> Dict[str, Any]:
        """处理单个文本"""
        try:
            result = self.base_model(text)
            
            # 验证和修复结果
            if self.config.enable_json_repair:
                result = self._validate_and_repair_result(result, text)
            
            self.stats['successful_calls'] += 1
            return result
            
        except Exception as e:
            logger.error(f"单文本处理失败: {e}")
            raise
    
    def _process_segmented_text(self, text: str, num_segments: int) -> Dict[str, Any]:
        """处理分段文本"""
        segments = self.token_manager.split_text_semantically(text, num_segments)
        
        all_entities = []
        all_triples = []
        
        for i, segment in enumerate(segments):
            logger.debug(f"处理第 {i+1}/{num_segments} 段")
            try:
                result = self._process_single_text(segment)
                all_entities.extend(result.get('extracted_entities', []))
                all_triples.extend(result.get('extracted_triples', []))
            except Exception as e:
                logger.warning(f"处理第 {i+1} 段失败: {e}")
                continue
        
        # 去重和合并
        unique_entities = list(set(all_entities))
        unique_triples = self._deduplicate_triples(all_triples)
        
        return {
            "passage": text,
            "extracted_entities": unique_entities,
            "extracted_triples": unique_triples
        }
    
    def _validate_and_repair_result(self, result: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """验证并修复结果"""
        if not isinstance(result, dict):
            logger.warning("结果不是字典格式，使用默认结构")
            return {
                "passage": original_text,
                "extracted_entities": [],
                "extracted_triples": []
            }
        
        # 确保必需字段存在
        if "extracted_entities" not in result:
            result["extracted_entities"] = []
        if "extracted_triples" not in result:
            result["extracted_triples"] = []
        if "passage" not in result:
            result["passage"] = original_text
        
        # 验证数据类型
        if not isinstance(result["extracted_entities"], list):
            logger.warning("实体字段不是列表，尝试转换")
            try:
                result["extracted_entities"] = list(result["extracted_entities"])
            except:
                result["extracted_entities"] = []
        
        if not isinstance(result["extracted_triples"], list):
            logger.warning("三元组字段不是列表，尝试转换")
            try:
                result["extracted_triples"] = list(result["extracted_triples"])
            except:
                result["extracted_triples"] = []
        
        return result
    
    def _deduplicate_triples(self, triples: List[List[str]]) -> List[List[str]]:
        """去重三元组"""
        seen = set()
        unique_triples = []
        
        for triple in triples:
            if isinstance(triple, list) and len(triple) >= 3:
                # 标准化三元组
                normalized = tuple(str(item).strip().lower() for item in triple[:3])
                if normalized not in seen:
                    seen.add(normalized)
                    unique_triples.append(triple[:3])  # 只保留前三个元素
        
        return unique_triples
    
    def _update_processing_time(self, processing_time: float):
        """更新平均处理时间"""
        total_calls = self.stats['total_calls']
        current_avg = self.stats['average_processing_time']
        self.stats['average_processing_time'] = (
            (current_avg * (total_calls - 1) + processing_time) / total_calls
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self.stats.copy()
        if stats['total_calls'] > 0:
            stats['success_rate'] = stats['successful_calls'] / stats['total_calls']
        else:
            stats['success_rate'] = 0.0
        return stats


def batch_kg_construction(
    documents: List[str],
    openie_model,
    config: Optional[KGProcessingConfig] = None
) -> Dict[str, Any]:
    """批量知识图谱构建 - 带容错和恢复机制
    
    Args:
        documents: 文档列表
        openie_model: OpenIE模型实例
        config: 处理配置
        
    Returns:
        Dict[str, Any]: 构建结果和统计信息
    """
    config = config or KGProcessingConfig()
    enhanced_model = EnhancedOpenIEModel(openie_model, config)
    
    def process_document(doc: str) -> Dict[str, Any]:
        """处理单个文档"""
        try:
            return enhanced_model(doc)
        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            return {
                "passage": doc,
                "extracted_entities": [],
                "extracted_triples": [],
                "error": str(e)
            }
    
    # 使用带恢复机制的批量处理
    result = batch_process_with_recovery(
        input_data=documents,
        worker_func=process_document,
        max_workers=config.max_workers,
        checkpoint_dir=config.checkpoint_dir,
        job_id=f"kg_construction_{int(time.time())}",
        max_retries=config.max_retries,
        retry_delay=config.retry_delay
    )
    
    # 汇总统计信息
    model_stats = enhanced_model.get_statistics()
    result['model_stats'] = model_stats
    
    logger.info(f"批量KG构建完成 - 模型统计: {model_stats}")
    
    return result


def create_enhanced_models(base_ner_model, base_openie_model, config: Optional[KGProcessingConfig] = None):
    """创建增强模型实例的便捷函数
    
    Args:
        base_ner_model: 原始NER模型
        base_openie_model: 原始OpenIE模型
        config: 处理配置
        
    Returns:
        Tuple[EnhancedNERModel, EnhancedOpenIEModel]: 增强模型实例
    """
    config = config or KGProcessingConfig()
    
    enhanced_ner = EnhancedNERModel(base_ner_model, config)
    enhanced_openie = EnhancedOpenIEModel(base_openie_model, config)
    
    return enhanced_ner, enhanced_openie