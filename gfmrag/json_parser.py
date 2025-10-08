"""
JSON解析容错系统

该模块提供多层JSON解析策略和智能格式修复功能，
能够处理各种格式异常的LLM响应，确保数据解析的稳定性。
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ParseStrategy(Enum):
    """解析策略枚举"""
    STRICT = "strict"          # 严格JSON解析
    LENIENT = "lenient"        # 宽松解析
    REGEX = "regex"            # 正则表达式提取
    REPAIR = "repair"          # 智能修复
    FALLBACK = "fallback"      # 降级处理


@dataclass
class ParseResult:
    """解析结果"""
    success: bool
    data: Any
    strategy: ParseStrategy
    confidence: float
    error_message: Optional[str] = None
    raw_content: Optional[str] = None
    
    def is_valid(self) -> bool:
        """检查解析结果是否有效"""
        return self.success and self.data is not None


class JSONParseError(Exception):
    """JSON解析异常"""
    
    def __init__(self, message: str, original_content: str, parse_strategy: ParseStrategy):
        super().__init__(message)
        self.original_content = original_content
        self.parse_strategy = parse_strategy


class RobustJSONParser:
    """鲁棒的JSON解析器
    
    提供多层解析策略，能够处理各种格式异常的JSON响应。
    """
    
    def __init__(self):
        """初始化JSON解析器"""
        self.parse_attempts = 0
        self.success_stats = {strategy.value: 0 for strategy in ParseStrategy}
        self.failure_patterns = []
        
        # 常见的格式修复规则
        self.repair_rules = [
            (r'(?<!\\)"([^"]*?)(?<!\\)"(\s*:\s*)([^",}\]]+)(?=[,}\]])', r'"\1"\2"\3"'),  # 值加引号
            (r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)', r'\1"\2"\3'),  # 键加引号
            (r',\s*([}\]])', r'\1'),  # 移除尾随逗号
            (r'([}\]])\s*,', r'\1'),  # 移除多余逗号
            (r':\s*([^",{\[\]}\s]+)(?=\s*[,}\]])', r': "\1"'),  # 简单值加引号
        ]
        
        logger.info("JSON解析器初始化完成")
    
    def parse(self, content: str, expected_keys: Optional[List[str]] = None) -> ParseResult:
        """解析JSON内容
        
        Args:
            content: 待解析的JSON内容
            expected_keys: 期望的JSON键列表
            
        Returns:
            ParseResult: 解析结果
        """
        self.parse_attempts += 1
        content = content.strip() if content else ""
        
        if not content:
            return ParseResult(False, None, ParseStrategy.STRICT, 0.0, "空内容")
        
        # 依次尝试各种解析策略
        strategies = [
            self._parse_strict,
            self._parse_lenient, 
            self._parse_regex,
            self._parse_repair,
            self._parse_fallback
        ]
        
        for strategy_func in strategies:
            try:
                result = strategy_func(content, expected_keys)
                if result.success:
                    self.success_stats[result.strategy.value] += 1
                    logger.debug(f"解析成功 - 策略: {result.strategy.value}, 置信度: {result.confidence}")
                    return result
            except Exception as e:
                logger.debug(f"策略 {strategy_func.__name__} 失败: {e}")
                continue
        
        # 所有策略都失败
        return ParseResult(
            False, 
            None, 
            ParseStrategy.FALLBACK, 
            0.0, 
            "所有解析策略都失败",
            content
        )
    
    def _parse_strict(self, content: str, expected_keys: Optional[List[str]] = None) -> ParseResult:
        """严格JSON解析"""
        try:
            data = json.loads(content)
            confidence = self._calculate_confidence(data, expected_keys, 1.0)
            return ParseResult(True, data, ParseStrategy.STRICT, confidence)
        except json.JSONDecodeError as e:
            raise JSONParseError(f"严格解析失败: {e}", content, ParseStrategy.STRICT)
    
    def _parse_lenient(self, content: str, expected_keys: Optional[List[str]] = None) -> ParseResult:
        """宽松解析 - 提取JSON部分"""
        # 尝试从响应中提取JSON部分
        json_patterns = [
            r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}',  # 完整JSON对象
            r'\[(?:[^\[\]]|(?:\[(?:[^\[\]]|(?:\[[^\[\]]*\])*)*\])*)*\]',  # JSON数组
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    confidence = self._calculate_confidence(data, expected_keys, 0.8)
                    return ParseResult(True, data, ParseStrategy.LENIENT, confidence, raw_content=match)
                except json.JSONDecodeError:
                    continue
        
        raise JSONParseError("宽松解析未找到有效JSON", content, ParseStrategy.LENIENT)
    
    def _parse_regex(self, content: str, expected_keys: Optional[List[str]] = None) -> ParseResult:
        """正则表达式解析 - 提取键值对"""
        if not expected_keys:
            raise JSONParseError("正则解析需要期望键列表", content, ParseStrategy.REGEX)
        
        data = {}
        patterns = {
            # 各种键值对格式
            'quoted_key_quoted_value': r'"([^"]+)"\s*:\s*"([^"]*)"',
            'quoted_key_unquoted_value': r'"([^"]+)"\s*:\s*([^,}\]]+)',
            'unquoted_key_quoted_value': r'([a-zA-Z_]\w*)\s*:\s*"([^"]*)"',
            'unquoted_key_unquoted_value': r'([a-zA-Z_]\w*)\s*:\s*([^,}\]]+)',
            'array_values': r'"([^"]+)"\s*:\s*\[([^\]]*)\]',
        }
        
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, content)
            for key, value in matches:
                key = key.strip()
                if key in expected_keys:
                    # 处理数组值
                    if pattern_name == 'array_values':
                        # 解析数组内容
                        array_items = [item.strip(' "') for item in value.split(',') if item.strip()]
                        data[key] = array_items
                    else:
                        # 清理值
                        value = value.strip(' "')
                        # 尝试转换为合适的类型
                        data[key] = self._convert_value(value)
        
        if data:
            confidence = self._calculate_confidence(data, expected_keys, 0.6)
            return ParseResult(True, data, ParseStrategy.REGEX, confidence)
        
        raise JSONParseError("正则解析未提取到有效数据", content, ParseStrategy.REGEX)
    
    def _parse_repair(self, content: str, expected_keys: Optional[List[str]] = None) -> ParseResult:
        """智能修复解析"""
        repaired_content = content
        
        # 应用修复规则
        for pattern, replacement in self.repair_rules:
            repaired_content = re.sub(pattern, replacement, repaired_content)
        
        # 尝试修复常见问题
        repaired_content = self._fix_common_issues(repaired_content)
        
        try:
            data = json.loads(repaired_content)
            confidence = self._calculate_confidence(data, expected_keys, 0.7)
            return ParseResult(True, data, ParseStrategy.REPAIR, confidence, raw_content=repaired_content)
        except json.JSONDecodeError:
            # 进一步修复
            further_repaired = self._aggressive_repair(repaired_content)
            try:
                data = json.loads(further_repaired)
                confidence = self._calculate_confidence(data, expected_keys, 0.5)
                return ParseResult(True, data, ParseStrategy.REPAIR, confidence, raw_content=further_repaired)
            except json.JSONDecodeError as e:
                raise JSONParseError(f"修复解析失败: {e}", content, ParseStrategy.REPAIR)
    
    def _parse_fallback(self, content: str, expected_keys: Optional[List[str]] = None) -> ParseResult:
        """降级处理 - 返回安全默认值"""
        if expected_keys:
            # 根据期望键创建默认结构
            if "named_entities" in expected_keys:
                data = {"named_entities": []}
            elif "triples" in expected_keys:
                data = {"triples": []}
            elif "entities" in expected_keys:
                data = {"entities": []}
            else:
                data = {key: [] for key in expected_keys}
        else:
            data = {}
        
        return ParseResult(True, data, ParseStrategy.FALLBACK, 0.1, "降级处理", content)
    
    def _fix_common_issues(self, content: str) -> str:
        """修复常见JSON问题"""
        # 移除注释
        content = re.sub(r'//.*?\n', '', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # 修复单引号
        content = re.sub(r"'([^']*)'", r'"\1"', content)
        
        # 修复尾随逗号
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        # 确保字符串转义
        content = re.sub(r'(?<!\\)"', r'"', content)
        
        return content
    
    def _aggressive_repair(self, content: str) -> str:
        """激进修复策略"""
        # 尝试补全缺失的括号
        open_braces = content.count('{')
        close_braces = content.count('}')
        if open_braces > close_braces:
            content += '}' * (open_braces - close_braces)
        
        open_brackets = content.count('[')
        close_brackets = content.count(']')
        if open_brackets > close_brackets:
            content += ']' * (open_brackets - close_brackets)
        
        # 移除无效字符
        content = re.sub(r'[^\w\s\[\]{}":,.-]', '', content)
        
        return content
    
    def _convert_value(self, value: str) -> Any:
        """将字符串值转换为合适的类型"""
        value = value.strip()
        
        # 布尔值
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # 数字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # null值
        if value.lower() in ('null', 'none', ''):
            return None
        
        # 字符串
        return value
    
    def _calculate_confidence(self, data: Any, expected_keys: Optional[List[str]], base_confidence: float) -> float:
        """计算解析置信度"""
        if not expected_keys:
            return base_confidence
        
        if not isinstance(data, dict):
            return base_confidence * 0.5
        
        # 检查期望键的存在
        found_keys = sum(1 for key in expected_keys if key in data)
        key_ratio = found_keys / len(expected_keys) if expected_keys else 0
        
        # 检查数据质量
        quality_score = self._assess_data_quality(data)
        
        return base_confidence * (0.5 * key_ratio + 0.5 * quality_score)
    
    def _assess_data_quality(self, data: Dict[str, Any]) -> float:
        """评估数据质量"""
        if not data:
            return 0.0
        
        quality_factors = []
        
        # 检查是否有空值
        non_empty_values = sum(1 for v in data.values() if v)
        quality_factors.append(non_empty_values / len(data))
        
        # 检查数据类型合理性
        type_score = 0
        for key, value in data.items():
            if "entities" in key.lower() and isinstance(value, list):
                type_score += 1
            elif "triples" in key.lower() and isinstance(value, list):
                type_score += 1
            elif isinstance(value, (str, int, float, bool)):
                type_score += 0.8
        
        if data:
            quality_factors.append(type_score / len(data))
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取解析统计信息"""
        total_attempts = sum(self.success_stats.values())
        stats = {
            "total_parse_attempts": self.parse_attempts,
            "successful_parses": total_attempts,
            "success_rate": total_attempts / self.parse_attempts if self.parse_attempts > 0 else 0,
            "strategy_distribution": self.success_stats.copy()
        }
        
        if total_attempts > 0:
            stats["strategy_percentages"] = {
                strategy: (count / total_attempts) * 100
                for strategy, count in self.success_stats.items()
            }
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.parse_attempts = 0
        self.success_stats = {strategy.value: 0 for strategy in ParseStrategy}


class EnhancedJSONExtractor:
    """增强的JSON提取器
    
    专门用于从LLM响应中提取JSON数据，支持多种响应格式。
    """
    
    def __init__(self):
        self.parser = RobustJSONParser()
    
    def extract_named_entities(self, response: str) -> ParseResult:
        """从响应中提取命名实体"""
        return self.parser.parse(response, expected_keys=["named_entities", "entities"])
    
    def extract_triples(self, response: str) -> ParseResult:
        """从响应中提取关系三元组"""
        return self.parser.parse(response, expected_keys=["triples", "relations"])
    
    def extract_structured_data(self, response: str, expected_schema: Dict[str, type]) -> ParseResult:
        """根据期望模式提取结构化数据"""
        expected_keys = list(expected_schema.keys())
        result = self.parser.parse(response, expected_keys)
        
        if result.success and isinstance(result.data, dict):
            # 验证数据类型
            validated_data = {}
            for key, expected_type in expected_schema.items():
                if key in result.data:
                    value = result.data[key]
                    if self._validate_type(value, expected_type):
                        validated_data[key] = value
                    else:
                        logger.warning(f"字段 {key} 类型不匹配，期望 {expected_type}, 实际 {type(value)}")
            
            result.data = validated_data
        
        return result
    
    def _validate_type(self, value: Any, expected_type: type) -> bool:
        """验证值类型"""
        if expected_type == list:
            return isinstance(value, list)
        elif expected_type == dict:
            return isinstance(value, dict)
        elif expected_type == str:
            return isinstance(value, str)
        elif expected_type == int:
            return isinstance(value, int)
        elif expected_type == float:
            return isinstance(value, (int, float))
        elif expected_type == bool:
            return isinstance(value, bool)
        else:
            return isinstance(value, expected_type)


# 全局解析器实例
_json_parser = None
_json_extractor = None


def get_json_parser() -> RobustJSONParser:
    """获取全局JSON解析器实例"""
    global _json_parser
    if _json_parser is None:
        _json_parser = RobustJSONParser()
    return _json_parser


def get_json_extractor() -> EnhancedJSONExtractor:
    """获取全局JSON提取器实例"""
    global _json_extractor
    if _json_extractor is None:
        _json_extractor = EnhancedJSONExtractor()
    return _json_extractor


def safe_parse_json(content: str, expected_keys: Optional[List[str]] = None) -> ParseResult:
    """安全解析JSON的便捷函数"""
    parser = get_json_parser()
    return parser.parse(content, expected_keys)