"""
自适应Token管理器

该模块实现了智能Token分配和管理策略，支持：
- 基于文本长度的动态Token分配
- 环境变量配置覆盖
- 多场景Token优化
- 自适应调整机制
"""

import os
import logging
import math
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentSize(Enum):
    """文档规模分类"""
    SMALL = "small"      # <500字符
    MEDIUM = "medium"    # 500-2000字符
    LARGE = "large"      # 2000-5000字符
    XLARGE = "xlarge"    # >5000字符


@dataclass
class TokenAllocation:
    """Token分配配置"""
    ner_tokens: int
    triples_tokens: int
    chat_tokens: int
    buffer_ratio: float = 0.1
    
    def with_buffer(self) -> 'TokenAllocation':
        """返回包含缓冲区的Token分配"""
        buffer_ner = int(self.ner_tokens * (1 + self.buffer_ratio))
        buffer_triples = int(self.triples_tokens * (1 + self.buffer_ratio))
        buffer_chat = int(self.chat_tokens * (1 + self.buffer_ratio))
        
        return TokenAllocation(
            ner_tokens=buffer_ner,
            triples_tokens=buffer_triples,
            chat_tokens=buffer_chat,
            buffer_ratio=self.buffer_ratio
        )


class AdaptiveTokenManager:
    """自适应Token管理器
    
    负责根据文档规模、处理类型和环境配置动态分配Token，
    提供智能的Token管理策略。
    """
    
    def __init__(self):
        """初始化Token管理器"""
        self.default_allocations = self._load_default_allocations()
        self.environment_overrides = self._load_environment_overrides()
        self.adaptive_enabled = self._get_env_bool("GFMRAG_ADAPTIVE_TOKENS", True)
        
        # 性能统计
        self.allocation_stats = {
            "total_allocations": 0,
            "size_distribution": {size.value: 0 for size in DocumentSize},
            "average_efficiency": 0.0
        }
        
        logger.info(f"Token管理器初始化完成 - 自适应模式: {self.adaptive_enabled}")
    
    def _load_default_allocations(self) -> Dict[DocumentSize, TokenAllocation]:
        """加载默认Token分配策略"""
        return {
            DocumentSize.SMALL: TokenAllocation(
                ner_tokens=512,
                triples_tokens=4096,
                chat_tokens=4096
            ),
            DocumentSize.MEDIUM: TokenAllocation(
                ner_tokens=800,
                triples_tokens=6144,
                chat_tokens=6144
            ),
            DocumentSize.LARGE: TokenAllocation(
                ner_tokens=1200,
                triples_tokens=8192,
                chat_tokens=8192
            ),
            DocumentSize.XLARGE: TokenAllocation(
                ner_tokens=1600,
                triples_tokens=12288,
                chat_tokens=12288
            )
        }
    
    def _load_environment_overrides(self) -> Dict[str, int]:
        """加载环境变量覆盖配置"""
        env_vars = {
            "GFMRAG_NER_MAX_TOKENS": 800,
            "GFMRAG_OPENIE_NER_TOKENS": 800,
            "GFMRAG_OPENIE_TRIPLES_TOKENS": 8192,
            "GFMRAG_CHAT_MAX_TOKENS": 8192,
            "GFMRAG_NER_TOKENS_SMALL": 512,
            "GFMRAG_NER_TOKENS_MEDIUM": 800,
            "GFMRAG_NER_TOKENS_LARGE": 1200,
            "GFMRAG_NER_TOKENS_XLARGE": 1600,
            "GFMRAG_TRIPLES_TOKENS_SMALL": 4096,
            "GFMRAG_TRIPLES_TOKENS_MEDIUM": 6144,
            "GFMRAG_TRIPLES_TOKENS_LARGE": 8192,
            "GFMRAG_TRIPLES_TOKENS_XLARGE": 12288,
        }
        
        overrides = {}
        for env_var, default_value in env_vars.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    overrides[env_var] = int(value)
                    logger.info(f"环境变量覆盖: {env_var}={value}")
                except ValueError:
                    logger.warning(f"无效的环境变量值: {env_var}={value}, 使用默认值: {default_value}")
                    overrides[env_var] = default_value
            else:
                overrides[env_var] = default_value
        
        return overrides
    
    def _get_env_bool(self, env_var: str, default: bool) -> bool:
        """获取布尔型环境变量"""
        value = os.getenv(env_var)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def classify_document_size(self, text: str) -> DocumentSize:
        """根据文本长度分类文档规模
        
        Args:
            text: 输入文本
            
        Returns:
            DocumentSize: 文档规模分类
        """
        length = len(text)
        
        if length < 500:
            return DocumentSize.SMALL
        elif length < 2000:
            return DocumentSize.MEDIUM
        elif length < 5000:
            return DocumentSize.LARGE
        else:
            return DocumentSize.XLARGE
    
    def estimate_token_usage(self, text: str) -> int:
        """估算文本的Token使用量
        
        Args:
            text: 输入文本
            
        Returns:
            int: 估算的Token数量
        """
        # 简单的Token估算：英文约4字符/token，中文约1.5字符/token
        # 这里使用保守估算，约3字符/token
        return max(1, len(text) // 3)
    
    def get_optimal_allocation(
        self, 
        text: str, 
        task_type: str = "openie",
        custom_requirements: Optional[Dict[str, int]] = None
    ) -> TokenAllocation:
        """获取最优Token分配策略
        
        Args:
            text: 输入文本
            task_type: 任务类型 ('ner', 'openie', 'chat')
            custom_requirements: 自定义Token需求
            
        Returns:
            TokenAllocation: 优化的Token分配
        """
        doc_size = self.classify_document_size(text)
        base_allocation = self.default_allocations[doc_size].copy()
        
        # 应用环境变量覆盖
        if self.adaptive_enabled:
            base_allocation = self._apply_environment_overrides(base_allocation, doc_size)
        
        # 应用自定义需求
        if custom_requirements:
            base_allocation = self._apply_custom_requirements(base_allocation, custom_requirements)
        
        # 根据任务类型优化
        optimized_allocation = self._optimize_for_task(base_allocation, task_type, text)
        
        # 更新统计信息
        self._update_stats(doc_size)
        
        logger.debug(f"文档规模: {doc_size.value}, 任务: {task_type}, "
                    f"分配: NER={optimized_allocation.ner_tokens}, "
                    f"Triples={optimized_allocation.triples_tokens}")
        
        return optimized_allocation.with_buffer()
    
    def _apply_environment_overrides(
        self, 
        allocation: TokenAllocation, 
        doc_size: DocumentSize
    ) -> TokenAllocation:
        """应用环境变量覆盖"""
        size_suffix = doc_size.value.upper()
        
        # 尝试获取特定规模的配置
        ner_key = f"GFMRAG_NER_TOKENS_{size_suffix}"
        triples_key = f"GFMRAG_TRIPLES_TOKENS_{size_suffix}"
        
        if ner_key in self.environment_overrides:
            allocation.ner_tokens = self.environment_overrides[ner_key]
        elif "GFMRAG_NER_MAX_TOKENS" in self.environment_overrides:
            allocation.ner_tokens = self.environment_overrides["GFMRAG_NER_MAX_TOKENS"]
        
        if triples_key in self.environment_overrides:
            allocation.triples_tokens = self.environment_overrides[triples_key]
        elif "GFMRAG_OPENIE_TRIPLES_TOKENS" in self.environment_overrides:
            allocation.triples_tokens = self.environment_overrides["GFMRAG_OPENIE_TRIPLES_TOKENS"]
        
        if "GFMRAG_CHAT_MAX_TOKENS" in self.environment_overrides:
            allocation.chat_tokens = self.environment_overrides["GFMRAG_CHAT_MAX_TOKENS"]
        
        return allocation
    
    def _apply_custom_requirements(
        self, 
        allocation: TokenAllocation, 
        requirements: Dict[str, int]
    ) -> TokenAllocation:
        """应用自定义Token需求"""
        if "ner_tokens" in requirements:
            allocation.ner_tokens = requirements["ner_tokens"]
        if "triples_tokens" in requirements:
            allocation.triples_tokens = requirements["triples_tokens"]
        if "chat_tokens" in requirements:
            allocation.chat_tokens = requirements["chat_tokens"]
        
        return allocation
    
    def _optimize_for_task(
        self, 
        allocation: TokenAllocation, 
        task_type: str, 
        text: str
    ) -> TokenAllocation:
        """根据任务类型优化Token分配"""
        estimated_tokens = self.estimate_token_usage(text)
        
        if task_type == "ner":
            # NER任务优先保证实体识别Token充足
            min_ner_tokens = min(estimated_tokens * 2, allocation.ner_tokens)
            allocation.ner_tokens = max(allocation.ner_tokens, min_ner_tokens)
            
        elif task_type == "openie":
            # OpenIE任务需要平衡NER和关系抽取
            min_ner_tokens = min(estimated_tokens, allocation.ner_tokens)
            min_triples_tokens = min(estimated_tokens * 3, allocation.triples_tokens)
            
            allocation.ner_tokens = max(allocation.ner_tokens, min_ner_tokens)
            allocation.triples_tokens = max(allocation.triples_tokens, min_triples_tokens)
            
        elif task_type == "chat":
            # 对话任务优先保证足够的生成Token
            min_chat_tokens = min(estimated_tokens * 2, allocation.chat_tokens)
            allocation.chat_tokens = max(allocation.chat_tokens, min_chat_tokens)
        
        return allocation
    
    def _update_stats(self, doc_size: DocumentSize):
        """更新统计信息"""
        self.allocation_stats["total_allocations"] += 1
        self.allocation_stats["size_distribution"][doc_size.value] += 1
    
    def suggest_segmentation(self, text: str, max_tokens: int) -> Tuple[bool, int]:
        """建议是否需要分段处理
        
        Args:
            text: 输入文本
            max_tokens: 最大Token限制
            
        Returns:
            Tuple[bool, int]: (是否需要分段, 建议分段数量)
        """
        estimated_tokens = self.estimate_token_usage(text)
        
        if estimated_tokens <= max_tokens:
            return False, 1
        
        # 计算需要的分段数量
        segments = math.ceil(estimated_tokens / max_tokens)
        
        logger.info(f"文本预计使用 {estimated_tokens} tokens，超过限制 {max_tokens}，建议分为 {segments} 段")
        
        return True, segments
    
    def split_text_semantically(self, text: str, num_segments: int) -> list[str]:
        """语义化分割文本
        
        Args:
            text: 输入文本
            num_segments: 分段数量
            
        Returns:
            list[str]: 分割后的文本段落
        """
        if num_segments <= 1:
            return [text]
        
        # 简单实现：按句子分割，后续可以改进为更智能的语义分割
        sentences = self._split_into_sentences(text)
        if len(sentences) <= num_segments:
            return sentences
        
        # 平均分配句子到各个段落
        segment_size = len(sentences) // num_segments
        segments = []
        
        for i in range(num_segments):
            start_idx = i * segment_size
            if i == num_segments - 1:  # 最后一段包含剩余所有句子
                end_idx = len(sentences)
            else:
                end_idx = (i + 1) * segment_size
            
            segment_sentences = sentences[start_idx:end_idx]
            segments.append(" ".join(segment_sentences))
        
        return segments
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """将文本分割为句子"""
        # 简单的句子分割实现
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """获取分配统计信息"""
        return self.allocation_stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.allocation_stats = {
            "total_allocations": 0,
            "size_distribution": {size.value: 0 for size in DocumentSize},
            "average_efficiency": 0.0
        }


# 全局Token管理器实例
_token_manager = None


def get_token_manager() -> AdaptiveTokenManager:
    """获取全局Token管理器实例"""
    global _token_manager
    if _token_manager is None:
        _token_manager = AdaptiveTokenManager()
    return _token_manager


def calculate_optimal_tokens(
    text: str, 
    task_type: str = "openie",
    custom_requirements: Optional[Dict[str, int]] = None
) -> TokenAllocation:
    """计算最优Token分配的便捷函数
    
    Args:
        text: 输入文本
        task_type: 任务类型
        custom_requirements: 自定义需求
        
    Returns:
        TokenAllocation: 优化的Token分配
    """
    manager = get_token_manager()
    return manager.get_optimal_allocation(text, task_type, custom_requirements)