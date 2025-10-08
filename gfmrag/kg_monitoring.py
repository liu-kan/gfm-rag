"""
知识图谱构建错误监控和日志系统

该模块提供全面的监控和日志功能，包括：
- 实时错误监控和告警
- 详细的性能指标收集
- 可视化监控面板数据
- 智能错误分析和建议
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import queue
import hashlib

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """告警信息"""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: float
    source: str
    metadata: Dict[str, Any]
    resolved: bool = False
    resolved_timestamp: Optional[float] = None


@dataclass
class Metric:
    """监控指标"""
    name: str
    type: MetricType
    value: float
    timestamp: float
    tags: Dict[str, str]
    help_text: Optional[str] = None


@dataclass
class ErrorEvent:
    """错误事件"""
    event_id: str
    timestamp: float
    error_type: str
    error_message: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    resolved: bool = False


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_history: int = 1000):
        """初始化指标收集器
        
        Args:
            max_history: 最大历史记录数
        """
        self.max_history = max_history
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.current_metrics: Dict[str, Metric] = {}
        self.lock = threading.Lock()
        
        logger.info("指标收集器初始化完成")
    
    def record_counter(self, name: str, increment: float = 1.0, tags: Dict[str, str] = None):
        """记录计数器指标"""
        tags = tags or {}
        with self.lock:
            key = self._make_key(name, tags)
            current_value = self.current_metrics.get(key, Metric(name, MetricType.COUNTER, 0, 0, tags)).value
            new_value = current_value + increment
            
            metric = Metric(name, MetricType.COUNTER, new_value, time.time(), tags)
            self.current_metrics[key] = metric
            self.metrics_history[key].append(metric)
    
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """记录仪表盘指标"""
        tags = tags or {}
        with self.lock:
            key = self._make_key(name, tags)
            metric = Metric(name, MetricType.GAUGE, value, time.time(), tags)
            self.current_metrics[key] = metric
            self.metrics_history[key].append(metric)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """记录计时器指标"""
        tags = tags or {}
        with self.lock:
            key = self._make_key(name, tags)
            metric = Metric(name, MetricType.TIMER, duration, time.time(), tags)
            self.current_metrics[key] = metric
            self.metrics_history[key].append(metric)
    
    def _make_key(self, name: str, tags: Dict[str, str]) -> str:
        """生成指标键"""
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}#{tag_str}" if tag_str else name
    
    def get_current_metrics(self) -> Dict[str, Metric]:
        """获取当前指标"""
        with self.lock:
            return self.current_metrics.copy()
    
    def get_metric_history(self, name: str, tags: Dict[str, str] = None) -> List[Metric]:
        """获取指标历史"""
        tags = tags or {}
        key = self._make_key(name, tags)
        with self.lock:
            return list(self.metrics_history.get(key, []))
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                'total_metrics': len(self.current_metrics),
                'history_size': sum(len(history) for history in self.metrics_history.values()),
                'metric_names': list(set(metric.name for metric in self.current_metrics.values()))
            }


class AlertManager:
    """告警管理器"""
    
    def __init__(self, max_alerts: int = 1000):
        """初始化告警管理器
        
        Args:
            max_alerts: 最大告警数量
        """
        self.max_alerts = max_alerts
        self.alerts: deque = deque(maxlen=max_alerts)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.lock = threading.Lock()
        
        # 告警规则
        self.alert_rules = {
            'high_error_rate': {
                'threshold': 0.05,  # 5%错误率
                'window': 300,      # 5分钟窗口
                'level': AlertLevel.WARNING
            },
            'critical_error_rate': {
                'threshold': 0.1,   # 10%错误率
                'window': 300,
                'level': AlertLevel.CRITICAL
            },
            'low_success_rate': {
                'threshold': 0.9,   # 90%成功率以下
                'window': 600,      # 10分钟窗口
                'level': AlertLevel.WARNING
            },
            'high_processing_time': {
                'threshold': 60.0,  # 60秒处理时间
                'window': 300,
                'level': AlertLevel.WARNING
            }
        }
        
        logger.info("告警管理器初始化完成")
    
    def create_alert(
        self, 
        level: AlertLevel, 
        title: str, 
        message: str, 
        source: str = "system",
        metadata: Dict[str, Any] = None
    ) -> Alert:
        """创建告警"""
        metadata = metadata or {}
        alert_id = self._generate_alert_id(title, source)
        
        alert = Alert(
            alert_id=alert_id,
            level=level,
            title=title,
            message=message,
            timestamp=time.time(),
            source=source,
            metadata=metadata
        )
        
        with self.lock:
            self.alerts.append(alert)
            self.active_alerts[alert_id] = alert
        
        # 通知处理器
        self._notify_handlers(alert)
        
        logger.warning(f"告警创建: [{level.value}] {title} - {message}")
        return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_timestamp = time.time()
                del self.active_alerts[alert_id]
                logger.info(f"告警已解决: {alert.title}")
                return True
        return False
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """添加告警处理器"""
        self.alert_handlers.append(handler)
    
    def _generate_alert_id(self, title: str, source: str) -> str:
        """生成告警ID"""
        content = f"{title}_{source}_{int(time.time())}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _notify_handlers(self, alert: Alert):
        """通知告警处理器"""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"告警处理器执行失败: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活动告警"""
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        with self.lock:
            return list(self.alerts)[-limit:]


class ErrorAnalyzer:
    """错误分析器"""
    
    def __init__(self):
        """初始化错误分析器"""
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.error_events: deque = deque(maxlen=1000)
        self.lock = threading.Lock()
        
        # 错误分类规则
        self.error_categories = {
            'token_limit': ['token', 'length', 'limit', 'truncate'],
            'json_parse': ['json', 'parse', 'decode', 'format'],
            'network': ['connection', 'timeout', 'network', 'http'],
            'authentication': ['auth', 'key', 'unauthorized', '401'],
            'rate_limit': ['rate', 'limit', 'quota', '429'],
            'service': ['service', 'server', '500', '502', '503']
        }
        
        logger.info("错误分析器初始化完成")
    
    def record_error(
        self, 
        error_type: str, 
        error_message: str, 
        context: Dict[str, Any] = None,
        stack_trace: str = None
    ):
        """记录错误事件"""
        context = context or {}
        event_id = hashlib.md5(f"{error_type}_{error_message}_{time.time()}".encode()).hexdigest()[:16]
        
        error_event = ErrorEvent(
            event_id=event_id,
            timestamp=time.time(),
            error_type=error_type,
            error_message=error_message,
            context=context,
            stack_trace=stack_trace
        )
        
        with self.lock:
            self.error_events.append(error_event)
            
            # 分析错误模式
            category = self._categorize_error(error_message)
            self.error_patterns[category] += 1
        
        logger.error(f"错误记录: [{error_type}] {error_message}")
    
    def _categorize_error(self, error_message: str) -> str:
        """分类错误"""
        error_lower = error_message.lower()
        
        for category, keywords in self.error_categories.items():
            if any(keyword in error_lower for keyword in keywords):
                return category
        
        return 'unknown'
    
    def get_error_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        """获取错误摘要"""
        current_time = time.time()
        window_start = current_time - time_window
        
        with self.lock:
            recent_errors = [
                event for event in self.error_events 
                if event.timestamp >= window_start
            ]
            
            error_counts = defaultdict(int)
            for event in recent_errors:
                category = self._categorize_error(event.error_message)
                error_counts[category] += 1
            
            return {
                'total_errors': len(recent_errors),
                'error_categories': dict(error_counts),
                'error_rate': len(recent_errors) / (time_window / 60),  # errors per minute
                'top_errors': self._get_top_errors(recent_errors)
            }
    
    def _get_top_errors(self, errors: List[ErrorEvent], limit: int = 5) -> List[Dict[str, Any]]:
        """获取最常见错误"""
        error_counts = defaultdict(int)
        error_examples = {}
        
        for event in errors:
            error_counts[event.error_type] += 1
            if event.error_type not in error_examples:
                error_examples[event.error_type] = event.error_message
        
        top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        return [
            {
                'error_type': error_type,
                'count': count,
                'example': error_examples[error_type]
            }
            for error_type, count in top_errors
        ]
    
    def get_recommendations(self) -> List[str]:
        """获取优化建议"""
        recommendations = []
        error_summary = self.get_error_summary()
        
        if error_summary['error_categories'].get('token_limit', 0) > 5:
            recommendations.append("检测到频繁的Token限制错误，建议启用自适应Token管理或增加Token配额")
        
        if error_summary['error_categories'].get('json_parse', 0) > 3:
            recommendations.append("检测到JSON解析错误，建议启用JSON容错解析器")
        
        if error_summary['error_categories'].get('network', 0) > 10:
            recommendations.append("检测到网络错误较多，建议检查网络连接或增加重试机制")
        
        if error_summary['error_rate'] > 10:  # 10 errors per minute
            recommendations.append("错误率过高，建议启用断路器模式或降低并发度")
        
        return recommendations


class KGMonitoringSystem:
    """知识图谱监控系统"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化监控系统
        
        Args:
            config: 监控配置
        """
        self.config = config or {}
        
        # 初始化组件
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.error_analyzer = ErrorAnalyzer()
        
        # 监控状态
        self.is_running = False
        self.monitor_thread = None
        
        # 设置默认告警处理器
        self.alert_manager.add_handler(self._default_alert_handler)
        
        logger.info("KG监控系统初始化完成")
    
    def start_monitoring(self):
        """启动监控"""
        if self.is_running:
            logger.warning("监控系统已在运行")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("监控系统已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("监控系统已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                self._check_alerts()
                time.sleep(30)  # 每30秒检查一次
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(60)  # 出错时延长检查间隔
    
    def _check_alerts(self):
        """检查告警条件"""
        # 检查错误率
        error_summary = self.error_analyzer.get_error_summary(300)  # 5分钟窗口
        
        if error_summary['error_rate'] > 5:  # 每分钟超过5个错误
            self.alert_manager.create_alert(
                AlertLevel.WARNING,
                "高错误率",
                f"5分钟内错误率: {error_summary['error_rate']:.2f} 错误/分钟",
                "error_analyzer",
                {'error_rate': error_summary['error_rate']}
            )
        
        # 检查特定错误类型
        if error_summary['error_categories'].get('token_limit', 0) > 10:
            self.alert_manager.create_alert(
                AlertLevel.ERROR,
                "Token限制错误频发",
                f"5分钟内Token限制错误: {error_summary['error_categories']['token_limit']} 次",
                "error_analyzer"
            )
    
    def _default_alert_handler(self, alert: Alert):
        """默认告警处理器"""
        # 这里可以实现发送邮件、推送通知等
        logger.warning(f"[ALERT] {alert.level.value.upper()}: {alert.title} - {alert.message}")
    
    def record_kg_processing_start(self, task_id: str, context: Dict[str, Any] = None):
        """记录KG处理开始"""
        self.metrics_collector.record_counter("kg_processing_started", tags={'task_id': task_id})
        logger.info(f"KG处理开始: {task_id}")
    
    def record_kg_processing_success(self, task_id: str, duration: float, entities: int, triples: int):
        """记录KG处理成功"""
        tags = {'task_id': task_id}
        self.metrics_collector.record_counter("kg_processing_success", tags=tags)
        self.metrics_collector.record_timer("kg_processing_duration", duration, tags=tags)
        self.metrics_collector.record_gauge("entities_extracted", entities, tags=tags)
        self.metrics_collector.record_gauge("triples_extracted", triples, tags=tags)
    
    def record_kg_processing_error(self, task_id: str, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """记录KG处理错误"""
        tags = {'task_id': task_id, 'error_type': error_type}
        self.metrics_collector.record_counter("kg_processing_error", tags=tags)
        self.error_analyzer.record_error(error_type, error_message, context)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取监控面板数据"""
        return {
            'metrics': self.metrics_collector.get_current_metrics(),
            'active_alerts': self.alert_manager.get_active_alerts(),
            'error_summary': self.error_analyzer.get_error_summary(),
            'recommendations': self.error_analyzer.get_recommendations(),
            'system_stats': {
                'metrics_stats': self.metrics_collector.get_statistics(),
                'monitoring_active': self.is_running
            }
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """导出指标数据"""
        if format == "json":
            data = {
                'timestamp': time.time(),
                'metrics': {k: asdict(v) for k, v in self.metrics_collector.get_current_metrics().items()},
                'alerts': [asdict(alert) for alert in self.alert_manager.get_active_alerts()],
                'errors': self.error_analyzer.get_error_summary()
            }
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的格式: {format}")


# 全局监控实例
_monitoring_system = None


def get_monitoring_system() -> KGMonitoringSystem:
    """获取全局监控系统实例"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = KGMonitoringSystem()
        _monitoring_system.start_monitoring()
    return _monitoring_system


def monitor_kg_processing(func):
    """KG处理监控装饰器"""
    def wrapper(*args, **kwargs):
        monitor = get_monitoring_system()
        task_id = kwargs.get('task_id', f"task_{int(time.time())}")
        
        monitor.record_kg_processing_start(task_id)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # 尝试从结果中提取实体和三元组数量
            entities = 0
            triples = 0
            if isinstance(result, dict):
                entities = len(result.get('extracted_entities', []))
                triples = len(result.get('extracted_triples', []))
            
            monitor.record_kg_processing_success(task_id, duration, entities, triples)
            return result
            
        except Exception as e:
            error_type = type(e).__name__
            monitor.record_kg_processing_error(task_id, error_type, str(e))
            raise
    
    return wrapper