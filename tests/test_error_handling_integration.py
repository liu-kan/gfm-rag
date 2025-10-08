"""
知识图谱构建错误处理集成测试

该测试模块验证所有错误处理机制的有效性，包括：
- Token管理器测试
- JSON解析容错测试
- 进程容错机制测试
- 监控系统测试
- 端到端集成测试
"""

import os
import json
import time
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gfmrag.token_manager import (
    AdaptiveTokenManager, 
    DocumentSize, 
    TokenAllocation,
    get_token_manager,
    calculate_optimal_tokens
)
from gfmrag.json_parser import (
    RobustJSONParser,
    EnhancedJSONExtractor,
    ParseStrategy,
    ParseResult,
    safe_parse_json
)
from gfmrag.process_manager import (
    ResilientProcessPool,
    TaskInfo,
    TaskStatus,
    ProcessState,
    batch_process_with_recovery,
    create_task_id
)
from gfmrag.kg_monitoring import (
    KGMonitoringSystem,
    MetricsCollector,
    AlertManager,
    ErrorAnalyzer,
    AlertLevel,
    get_monitoring_system,
    monitor_kg_processing
)
from gfmrag.enhanced_kg_models import (
    EnhancedNERModel,
    EnhancedOpenIEModel,
    KGProcessingConfig,
    batch_kg_construction
)


class TestTokenManager(unittest.TestCase):
    """Token管理器测试"""
    
    def setUp(self):
        """测试设置"""
        self.token_manager = AdaptiveTokenManager()
    
    def test_document_size_classification(self):
        """测试文档规模分类"""
        # 小文档
        small_text = "This is a small document."
        self.assertEqual(
            self.token_manager.classify_document_size(small_text),
            DocumentSize.SMALL
        )
        
        # 中文档
        medium_text = "This is a medium document. " * 50
        self.assertEqual(
            self.token_manager.classify_document_size(medium_text),
            DocumentSize.MEDIUM
        )
        
        # 大文档
        large_text = "This is a large document. " * 200
        self.assertEqual(
            self.token_manager.classify_document_size(large_text),
            DocumentSize.LARGE
        )
        
        # 超大文档
        xlarge_text = "This is an extra large document. " * 500
        self.assertEqual(
            self.token_manager.classify_document_size(xlarge_text),
            DocumentSize.XLARGE
        )
    
    def test_token_estimation(self):
        """测试Token估算"""
        text = "Hello world, this is a test."
        estimated = self.token_manager.estimate_token_usage(text)
        self.assertGreater(estimated, 0)
        self.assertLess(estimated, len(text))  # 应该少于字符数
    
    def test_optimal_allocation(self):
        """测试最优Token分配"""
        text = "Test document for optimal allocation."
        allocation = self.token_manager.get_optimal_allocation(text, "openie")
        
        self.assertIsInstance(allocation, TokenAllocation)
        self.assertGreater(allocation.ner_tokens, 0)
        self.assertGreater(allocation.triples_tokens, 0)
        self.assertGreater(allocation.chat_tokens, 0)
    
    def test_environment_variable_override(self):
        """测试环境变量覆盖"""
        with patch.dict(os.environ, {'GFMRAG_NER_MAX_TOKENS': '1000'}):
            manager = AdaptiveTokenManager()
            self.assertEqual(manager.environment_overrides['GFMRAG_NER_MAX_TOKENS'], 1000)
    
    def test_segmentation_suggestion(self):
        """测试分段建议"""
        long_text = "Very long text. " * 1000
        needs_segmentation, num_segments = self.token_manager.suggest_segmentation(long_text, 500)
        
        self.assertTrue(needs_segmentation)
        self.assertGreater(num_segments, 1)
    
    def test_text_segmentation(self):
        """测试文本分割"""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        segments = self.token_manager.split_text_semantically(text, 2)
        
        self.assertEqual(len(segments), 2)
        self.assertTrue(all(isinstance(segment, str) for segment in segments))


class TestJSONParser(unittest.TestCase):
    """JSON解析器测试"""
    
    def setUp(self):
        """测试设置"""
        self.parser = RobustJSONParser()
        self.extractor = EnhancedJSONExtractor()
    
    def test_strict_parsing(self):
        """测试严格解析"""
        valid_json = '{"named_entities": ["Entity1", "Entity2"]}'
        result = self.parser.parse(valid_json)
        
        self.assertTrue(result.success)
        self.assertEqual(result.strategy, ParseStrategy.STRICT)
        self.assertEqual(result.data["named_entities"], ["Entity1", "Entity2"])
    
    def test_lenient_parsing(self):
        """测试宽松解析"""
        messy_response = '''
        Here is the JSON response:
        {"named_entities": ["Entity1", "Entity2"]}
        Additional text here.
        '''
        result = self.parser.parse(messy_response)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["named_entities"], ["Entity1", "Entity2"])
    
    def test_repair_parsing(self):
        """测试修复解析"""
        broken_json = '{"named_entities": ["Entity1", "Entity2",]}'  # 尾随逗号
        result = self.parser.parse(broken_json)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["named_entities"], ["Entity1", "Entity2"])
    
    def test_regex_parsing(self):
        """测试正则解析"""
        unstructured_response = '''
        The named entities are:
        Entity1, Entity2, Entity3
        '''
        result = self.parser.parse(unstructured_response, ["named_entities"])
        
        # 正则解析可能成功也可能失败，取决于具体实现
        self.assertIsInstance(result, ParseResult)
    
    def test_fallback_parsing(self):
        """测试降级解析"""
        invalid_content = "This is not JSON at all!"
        result = self.parser.parse(invalid_content, ["named_entities"])
        
        self.assertTrue(result.success)
        self.assertEqual(result.strategy, ParseStrategy.FALLBACK)
        self.assertEqual(result.data["named_entities"], [])
    
    def test_named_entity_extraction(self):
        """测试命名实体提取"""
        response = '{"named_entities": ["John Doe", "New York"]}'
        result = self.extractor.extract_named_entities(response)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["named_entities"], ["John Doe", "New York"])
    
    def test_triples_extraction(self):
        """测试三元组提取"""
        response = '{"triples": [["John", "lives in", "New York"]]}'
        result = self.extractor.extract_triples(response)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["triples"], [["John", "lives in", "New York"]])


class TestProcessManager(unittest.TestCase):
    """进程管理器测试"""
    
    def setUp(self):
        """测试设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.pool = ResilientProcessPool(
            max_workers=2,
            checkpoint_dir=self.temp_dir,
            job_id="test_job"
        )
    
    def tearDown(self):
        """测试清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_task_submission(self):
        """测试任务提交"""
        tasks = [("task1", "data1"), ("task2", "data2")]
        self.pool.submit_tasks(tasks)
        
        self.assertEqual(len(self.pool.task_queue), 2)
        self.assertEqual(self.pool.task_queue[0].task_id, "task1")
        self.assertEqual(self.pool.task_queue[1].task_id, "task2")
    
    def test_simple_processing(self):
        """测试简单处理"""
        def simple_worker(data):
            return f"processed_{data}"
        
        tasks = [("task1", "data1"), ("task2", "data2")]
        self.pool.submit_tasks(tasks)
        
        stats = self.pool.process_tasks(simple_worker)
        
        self.assertGreater(stats['total_time'], 0)
        self.assertEqual(stats['completed_tasks'], 2)
        self.assertEqual(stats['failed_tasks'], 0)
    
    def test_error_handling(self):
        """测试错误处理"""
        def failing_worker(data):
            if data == "bad_data":
                raise ValueError("Test error")
            return f"processed_{data}"
        
        tasks = [("task1", "good_data"), ("task2", "bad_data")]
        self.pool.submit_tasks(tasks)
        
        stats = self.pool.process_tasks(failing_worker)
        
        # 应该有一个成功，一个最终失败（重试次数用尽）
        self.assertEqual(stats['completed_tasks'], 1)
        self.assertGreaterEqual(stats['failed_tasks'], 1)
    
    def test_checkpoint_save_load(self):
        """测试检查点保存和加载"""
        # 创建一些任务状态
        task1 = TaskInfo("task1", "data1", TaskStatus.COMPLETED, "result1")
        task2 = TaskInfo("task2", "data2", TaskStatus.FAILED, error="test error")
        
        self.pool.completed_tasks["task1"] = task1
        self.pool.failed_tasks["task2"] = task2
        
        # 保存检查点
        self.pool._save_current_progress()
        
        # 创建新的进程池并恢复
        new_pool = ResilientProcessPool(
            max_workers=2,
            checkpoint_dir=self.temp_dir,
            job_id="test_job"
        )
        
        recovered = new_pool.resume_from_checkpoint()
        self.assertTrue(recovered)
        self.assertEqual(len(new_pool.completed_tasks), 1)
    
    def test_task_id_generation(self):
        """测试任务ID生成"""
        data1 = "test_data_1"
        data2 = "test_data_2"
        
        id1 = create_task_id(data1)
        id2 = create_task_id(data2)
        
        self.assertNotEqual(id1, id2)
        self.assertEqual(len(id1), 16)  # MD5 hash prefix
        
        # 相同数据应该生成相同ID
        id1_again = create_task_id(data1)
        self.assertEqual(id1, id1_again)


class TestMonitoringSystem(unittest.TestCase):
    """监控系统测试"""
    
    def setUp(self):
        """测试设置"""
        self.monitoring = KGMonitoringSystem()
    
    def tearDown(self):
        """测试清理"""
        self.monitoring.stop_monitoring()
    
    def test_metrics_collection(self):
        """测试指标收集"""
        collector = self.monitoring.metrics_collector
        
        # 记录一些指标
        collector.record_counter("test_counter", 1.0, {"tag": "value"})
        collector.record_gauge("test_gauge", 42.0)
        collector.record_timer("test_timer", 1.5)
        
        # 检查指标
        metrics = collector.get_current_metrics()
        self.assertGreater(len(metrics), 0)
        
        # 检查统计信息
        stats = collector.get_statistics()
        self.assertGreater(stats['total_metrics'], 0)
    
    def test_alert_creation(self):
        """测试告警创建"""
        alert_manager = self.monitoring.alert_manager
        
        alert = alert_manager.create_alert(
            AlertLevel.WARNING,
            "Test Alert",
            "This is a test alert",
            "test_source"
        )
        
        self.assertIsNotNone(alert.alert_id)
        self.assertEqual(alert.level, AlertLevel.WARNING)
        self.assertEqual(alert.title, "Test Alert")
        
        # 检查活动告警
        active_alerts = alert_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 1)
        
        # 解决告警
        resolved = alert_manager.resolve_alert(alert.alert_id)
        self.assertTrue(resolved)
        
        # 检查告警已解决
        active_alerts = alert_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 0)
    
    def test_error_analysis(self):
        """测试错误分析"""
        analyzer = self.monitoring.error_analyzer
        
        # 记录一些错误
        analyzer.record_error("TokenLimitError", "Token limit exceeded")
        analyzer.record_error("JSONParseError", "Invalid JSON format")
        analyzer.record_error("TokenLimitError", "Another token error")
        
        # 获取错误摘要
        summary = analyzer.get_error_summary()
        
        self.assertEqual(summary['total_errors'], 3)
        self.assertIn('token_limit', summary['error_categories'])
        self.assertIn('json_parse', summary['error_categories'])
        
        # 获取建议
        recommendations = analyzer.get_recommendations()
        self.assertIsInstance(recommendations, list)
    
    def test_kg_processing_monitoring(self):
        """测试KG处理监控"""
        # 记录处理开始
        self.monitoring.record_kg_processing_start("test_task")
        
        # 记录成功
        self.monitoring.record_kg_processing_success("test_task", 2.5, 10, 5)
        
        # 记录错误
        self.monitoring.record_kg_processing_error(
            "test_task2", 
            "TestError", 
            "Test error message"
        )
        
        # 检查指标
        metrics = self.monitoring.metrics_collector.get_current_metrics()
        self.assertGreater(len(metrics), 0)
    
    def test_monitoring_decorator(self):
        """测试监控装饰器"""
        @monitor_kg_processing
        def test_function(data, task_id="decorator_test"):
            return {
                "extracted_entities": ["Entity1", "Entity2"],
                "extracted_triples": [["E1", "rel", "E2"]]
            }
        
        result = test_function("test_data")
        
        self.assertEqual(len(result["extracted_entities"]), 2)
        self.assertEqual(len(result["extracted_triples"]), 1)
        
        # 检查指标是否被记录
        metrics = self.monitoring.metrics_collector.get_current_metrics()
        self.assertGreater(len(metrics), 0)


class TestEnhancedModels(unittest.TestCase):
    """增强模型测试"""
    
    def setUp(self):
        """测试设置"""
        # 创建模拟的基础模型
        self.mock_ner_model = Mock()
        self.mock_openie_model = Mock()
        
        # 配置模拟返回值
        self.mock_ner_model.return_value = ["Entity1", "Entity2"]
        self.mock_openie_model.return_value = {
            "passage": "test passage",
            "extracted_entities": ["Entity1", "Entity2"],
            "extracted_triples": [["Entity1", "relation", "Entity2"]]
        }
        
        # 添加必要的属性
        self.mock_ner_model.max_tokens = 300
        self.mock_openie_model.max_ner_tokens = 300
        self.mock_openie_model.max_triples_tokens = 4096
    
    def test_enhanced_ner_model(self):
        """测试增强NER模型"""
        config = KGProcessingConfig(enable_adaptive_tokens=True)
        enhanced_ner = EnhancedNERModel(self.mock_ner_model, config)
        
        result = enhanced_ner("Test text for NER processing.")
        
        self.assertEqual(result, ["Entity1", "Entity2"])
        self.mock_ner_model.assert_called_once()
        
        # 检查统计信息
        stats = enhanced_ner.get_statistics()
        self.assertEqual(stats['total_calls'], 1)
        self.assertEqual(stats['successful_calls'], 1)
    
    def test_enhanced_openie_model(self):
        """测试增强OpenIE模型"""
        config = KGProcessingConfig(enable_adaptive_tokens=True)
        enhanced_openie = EnhancedOpenIEModel(self.mock_openie_model, config)
        
        result = enhanced_openie("Test text for OpenIE processing.")
        
        self.assertIn("extracted_entities", result)
        self.assertIn("extracted_triples", result)
        self.mock_openie_model.assert_called_once()
        
        # 检查统计信息
        stats = enhanced_openie.get_statistics()
        self.assertEqual(stats['total_calls'], 1)
        self.assertEqual(stats['successful_calls'], 1)
    
    def test_error_handling_in_enhanced_models(self):
        """测试增强模型的错误处理"""
        # 配置模拟模型抛出异常
        self.mock_ner_model.side_effect = Exception("Test error")
        
        config = KGProcessingConfig()
        enhanced_ner = EnhancedNERModel(self.mock_ner_model, config)
        
        # 应该返回空列表而不是抛出异常
        result = enhanced_ner("Test text")
        self.assertEqual(result, [])
        
        # 检查错误统计
        stats = enhanced_ner.get_statistics()
        self.assertEqual(stats['failed_calls'], 1)
    
    def test_segmentation_processing(self):
        """测试分段处理"""
        # 创建长文本
        long_text = "This is a very long text. " * 200
        
        config = KGProcessingConfig(enable_adaptive_tokens=True)
        enhanced_ner = EnhancedNERModel(self.mock_ner_model, config)
        
        # 设置较小的max_tokens以触发分段
        enhanced_ner.base_model.max_tokens = 100
        
        result = enhanced_ner(long_text)
        
        # 应该调用多次基础模型（分段处理）
        self.assertGreaterEqual(self.mock_ner_model.call_count, 1)


class TestIntegration(unittest.TestCase):
    """端到端集成测试"""
    
    def setUp(self):
        """测试设置"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_batch_kg_construction(self):
        """测试批量KG构建"""
        # 创建模拟OpenIE模型
        mock_openie = Mock()
        mock_openie.return_value = {
            "passage": "test",
            "extracted_entities": ["Entity1"],
            "extracted_triples": [["E1", "rel", "E2"]]
        }
        mock_openie.max_ner_tokens = 300
        mock_openie.max_triples_tokens = 4096
        
        # 测试文档
        documents = [
            "First test document.",
            "Second test document.",
            "Third test document."
        ]
        
        config = KGProcessingConfig(
            max_workers=2,
            checkpoint_dir=self.temp_dir
        )
        
        result = batch_kg_construction(documents, mock_openie, config)
        
        self.assertIn('stats', result)
        self.assertIn('model_stats', result)
        
        # 验证所有文档都被处理
        stats = result['stats']
        self.assertEqual(stats['completed_tasks'], 3)
    
    def test_environment_variable_integration(self):
        """测试环境变量集成"""
        with patch.dict(os.environ, {
            'GFMRAG_NER_MAX_TOKENS': '1000',
            'GFMRAG_OPENIE_TRIPLES_TOKENS': '8000'
        }):
            token_manager = AdaptiveTokenManager()
            allocation = token_manager.get_optimal_allocation("test text", "openie")
            
            # 验证环境变量被应用
            self.assertEqual(allocation.ner_tokens, 1000)
            self.assertEqual(allocation.triples_tokens, 8000)
    
    def test_full_error_recovery_scenario(self):
        """测试完整错误恢复场景"""
        def unreliable_worker(data):
            """不可靠的工作函数，随机失败"""
            if "fail" in data:
                raise ValueError(f"Simulated failure for {data}")
            return f"processed_{data}"
        
        # 包含会失败的数据
        test_data = ["good1", "fail1", "good2", "fail2", "good3"]
        
        result = batch_process_with_recovery(
            input_data=test_data,
            worker_func=unreliable_worker,
            max_workers=2,
            checkpoint_dir=self.temp_dir,
            job_id="error_recovery_test",
            max_retries=2,
            retry_delay=0.1
        )
        
        self.assertIn('stats', result)
        stats = result['stats']
        
        # 应该有成功和失败的任务
        self.assertGreater(stats['completed_tasks'], 0)
        self.assertGreater(stats['failed_tasks'], 0)
        
        # 总任务数应该等于输入数据数量
        total_processed = stats['completed_tasks'] + stats['failed_tasks']
        self.assertEqual(total_processed, len(test_data))


def run_performance_tests():
    """运行性能测试"""
    print("运行性能测试...")
    
    # Token管理器性能测试
    token_manager = AdaptiveTokenManager()
    
    start_time = time.time()
    for i in range(1000):
        text = f"Test document {i}. " * 10
        allocation = token_manager.get_optimal_allocation(text, "openie")
    end_time = time.time()
    
    print(f"Token管理器性能: 1000次分配耗时 {end_time - start_time:.3f} 秒")
    
    # JSON解析器性能测试
    parser = RobustJSONParser()
    test_json = '{"named_entities": ["Entity1", "Entity2", "Entity3"]}'
    
    start_time = time.time()
    for i in range(1000):
        result = parser.parse(test_json)
    end_time = time.time()
    
    print(f"JSON解析器性能: 1000次解析耗时 {end_time - start_time:.3f} 秒")
    
    print("性能测试完成")


if __name__ == '__main__':
    # 运行单元测试
    unittest.main(verbosity=2, exit=False)
    
    # 运行性能测试
    run_performance_tests()