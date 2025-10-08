# 知识图谱构建错误处理系统部署和运维指南

## 概述

本指南详细介绍了如何部署和运维知识图谱构建错误处理系统，包括系统配置、监控设置、故障排除和性能优化。

## 快速开始

### 1. 环境变量配置

在系统部署前，需要配置以下环境变量来启用错误处理功能：

```bash
# 基础配置
export GFMRAG_CHAT_PROVIDER="third-party"  # 或 "openai", "nvidia", "together"
export GFMRAG_CHAT_MODEL_NAME="Qwen3-VL-30B-A3B-Instruct-FP8"
export GFMRAG_CHAT_BASE_URL="http://your-model-server:8000/v1"
export GFMRAG_CHAT_KEY=""  # 如果需要认证

# Token管理配置
export GFMRAG_NER_MAX_TOKENS=800
export GFMRAG_OPENIE_NER_TOKENS=800
export GFMRAG_OPENIE_TRIPLES_TOKENS=8192
export GFMRAG_CHAT_MAX_TOKENS=8192

# 自适应Token配置
export GFMRAG_NER_TOKENS_SMALL=512
export GFMRAG_NER_TOKENS_MEDIUM=800
export GFMRAG_NER_TOKENS_LARGE=1200
export GFMRAG_NER_TOKENS_XLARGE=1600

export GFMRAG_TRIPLES_TOKENS_SMALL=4096
export GFMRAG_TRIPLES_TOKENS_MEDIUM=6144
export GFMRAG_TRIPLES_TOKENS_LARGE=8192
export GFMRAG_TRIPLES_TOKENS_XLARGE=12288

# 容错配置
export GFMRAG_ADAPTIVE_TOKENS=true
export GFMRAG_TOKEN_BUFFER_RATIO=0.1
export GFMRAG_MAX_RETRIES=3
export GFMRAG_NUM_PROCESSES=10
export GFMRAG_BATCH_SIZE=20
```

### 2. 系统集成

#### 2.1 现有代码集成

要在现有的知识图谱构建代码中启用错误处理，进行以下修改：

**NER模型集成：**

```python
# 原代码
from gfmrag.kg_construction.ner_model import LLMNERModel

# 新代码
from gfmrag.kg_construction.ner_model import LLMNERModel
from gfmrag.enhanced_kg_models import EnhancedNERModel, KGProcessingConfig

# 创建增强模型
base_ner_model = LLMNERModel(...)
config = KGProcessingConfig(
    enable_adaptive_tokens=True,
    enable_json_repair=True,
    max_retries=3
)
enhanced_ner_model = EnhancedNERModel(base_ner_model, config)

# 使用增强模型
entities = enhanced_ner_model(text)
```

**OpenIE模型集成：**

```python
# 原代码
from gfmrag.kg_construction.openie_model import LLMOPENIEModel

# 新代码
from gfmrag.kg_construction.openie_model import LLMOPENIEModel
from gfmrag.enhanced_kg_models import EnhancedOpenIEModel, KGProcessingConfig

# 创建增强模型
base_openie_model = LLMOPENIEModel(...)
config = KGProcessingConfig(
    enable_adaptive_tokens=True,
    enable_json_repair=True,
    enable_recovery=True
)
enhanced_openie_model = EnhancedOpenIEModel(base_openie_model, config)

# 使用增强模型
result = enhanced_openie_model(text)
```

#### 2.2 批量处理集成

对于大规模文档处理，使用带错误恢复的批量处理：

```python
from gfmrag.enhanced_kg_models import batch_kg_construction

documents = ["doc1", "doc2", ...]
config = KGProcessingConfig(
    max_workers=10,
    checkpoint_dir="./kg_checkpoints",
    max_retries=3
)

results = batch_kg_construction(documents, openie_model, config)
```

### 3. 监控系统启动

启用监控系统以实时追踪处理状态：

```python
from gfmrag.kg_monitoring import get_monitoring_system

# 启动监控
monitoring = get_monitoring_system()

# 监控将自动启动，收集指标和检测异常
```

## 详细部署指南

### 1. 系统要求

#### 1.1 硬件要求

| 文档规模 | CPU | 内存 | 存储 | 推荐配置 |
|---------|-----|------|------|----------|
| 小规模（<1K文档） | 4核 | 8GB | 50GB | 开发环境 |
| 中规模（1K-10K文档） | 8核 | 16GB | 100GB | 测试环境 |
| 大规模（10K-100K文档） | 16核 | 32GB | 200GB | 生产环境 |
| 超大规模（>100K文档） | 32核 | 64GB | 500GB | 企业环境 |

#### 1.2 软件依赖

```bash
# Python版本
Python >= 3.8

# 核心依赖
langchain >= 0.1.0
openai >= 1.0.0
hydra-core >= 1.3.0
omegaconf >= 2.3.0

# 可选依赖（用于监控）
psutil >= 5.9.0
prometheus-client >= 0.15.0  # 如果需要Prometheus集成
```

### 2. 配置管理

#### 2.1 配置文件层级

系统支持多层级配置，优先级从高到低：

1. 环境变量（最高优先级）
2. 命令行参数
3. 配置文件
4. 代码默认值（最低优先级）

#### 2.2 配置验证

部署前验证配置正确性：

```bash
python -c "
from gfmrag.token_manager import get_token_manager
from gfmrag.json_parser import get_json_parser
from gfmrag.kg_monitoring import get_monitoring_system

# 验证Token管理器
tm = get_token_manager()
print(f'Token管理器初始化成功: {tm.adaptive_enabled}')

# 验证JSON解析器
jp = get_json_parser()
result = jp.parse('{\"test\": \"value\"}')
print(f'JSON解析器测试: {result.success}')

# 验证监控系统
ms = get_monitoring_system()
print(f'监控系统状态: {ms.is_running}')
"
```

### 3. 性能调优

#### 3.1 Token分配优化

根据实际使用情况调整Token分配：

```python
# 监控Token使用效率
from gfmrag.token_manager import get_token_manager

tm = get_token_manager()
stats = tm.get_allocation_stats()

print(f"Token分配统计: {stats}")

# 根据统计信息调整环境变量
if stats['size_distribution']['large'] > 50:  # 大文档比例高
    os.environ['GFMRAG_NER_MAX_TOKENS'] = '1200'
    os.environ['GFMRAG_OPENIE_TRIPLES_TOKENS'] = '10240'
```

#### 3.2 并发度调优

```python
import psutil

# 根据系统资源动态调整并发度
cpu_count = psutil.cpu_count()
memory_gb = psutil.virtual_memory().total // (1024**3)

# 保守估算：每个进程使用2GB内存
max_workers = min(cpu_count, memory_gb // 2, 20)

config = KGProcessingConfig(max_workers=max_workers)
```

## 运维监控

### 1. 关键指标监控

#### 1.1 系统指标

```python
# 获取系统状态
from gfmrag.kg_monitoring import get_monitoring_system

monitoring = get_monitoring_system()
dashboard_data = monitoring.get_dashboard_data()

# 关键指标
print("系统指标:")
print(f"- 处理成功率: {dashboard_data.get('success_rate', 0):.2%}")
print(f"- 平均处理时间: {dashboard_data.get('avg_processing_time', 0):.2f}s")
print(f"- 错误率: {dashboard_data.get('error_rate', 0):.2f}/分钟")
print(f"- 活动告警数: {len(dashboard_data.get('active_alerts', []))}")
```

#### 1.2 业务指标

| 指标名称 | 描述 | 正常范围 | 告警阈值 |
|---------|------|---------|----------|
| 成功处理率 | 成功处理的文档比例 | >95% | <90% |
| 平均处理时间 | 每个文档的平均处理时间 | <30s | >60s |
| Token利用率 | 实际使用Token占分配Token比例 | 80-95% | <70% 或 >98% |
| 错误率 | 每分钟错误数量 | <1/分钟 | >5/分钟 |
| 进程存活率 | 正常运行的进程比例 | >90% | <80% |

### 2. 日志管理

#### 2.1 日志配置

```python
import logging

# 配置结构化日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/gfmrag/kg_processing.log'),
        logging.StreamHandler()
    ]
)

# 为不同组件设置不同日志级别
logging.getLogger('gfmrag.token_manager').setLevel(logging.DEBUG)
logging.getLogger('gfmrag.json_parser').setLevel(logging.WARNING)
logging.getLogger('gfmrag.process_manager').setLevel(logging.INFO)
```

#### 2.2 日志分析

```bash
# 分析错误模式
grep "ERROR" /var/log/gfmrag/kg_processing.log | \
  awk '{print $5}' | sort | uniq -c | sort -nr

# 分析处理性能
grep "处理完成" /var/log/gfmrag/kg_processing.log | \
  awk '{print $7}' | awk -F: '{sum+=$1; count++} END {print "平均处理时间:", sum/count, "秒"}'

# 分析Token使用
grep "Token优化" /var/log/gfmrag/kg_processing.log | tail -100
```

### 3. 告警配置

#### 3.1 内置告警规则

系统内置以下告警规则：

- 高错误率（>5%）：WARNING级别
- 严重错误率（>10%）：CRITICAL级别
- 低成功率（<90%）：WARNING级别
- 高处理时间（>60s）：WARNING级别

#### 3.2 自定义告警

```python
from gfmrag.kg_monitoring import get_monitoring_system, AlertLevel

monitoring = get_monitoring_system()

# 添加自定义告警处理器
def custom_alert_handler(alert):
    if alert.level == AlertLevel.CRITICAL:
        # 发送紧急通知
        send_emergency_notification(alert)
    elif alert.level == AlertLevel.ERROR:
        # 发送邮件通知
        send_email_notification(alert)

monitoring.alert_manager.add_handler(custom_alert_handler)
```

## 故障排除

### 1. 常见问题诊断

#### 1.1 Token限制错误

**症状：** 频繁出现"Token limit exceeded"错误

**诊断：**
```python
from gfmrag.token_manager import get_token_manager

tm = get_token_manager()
stats = tm.get_allocation_stats()
print(f"当前Token分配统计: {stats}")

# 检查环境变量
import os
print(f"NER Token限制: {os.getenv('GFMRAG_NER_MAX_TOKENS')}")
print(f"OpenIE Token限制: {os.getenv('GFMRAG_OPENIE_TRIPLES_TOKENS')}")
```

**解决方案：**
1. 增加Token配额：`export GFMRAG_NER_MAX_TOKENS=1200`
2. 启用自适应Token：`export GFMRAG_ADAPTIVE_TOKENS=true`
3. 启用文本分段处理

#### 1.2 JSON解析错误

**症状：** "JSON parse error"或"Invalid JSON format"

**诊断：**
```python
from gfmrag.json_parser import get_json_parser

parser = get_json_parser()
stats = parser.get_statistics()
print(f"JSON解析统计: {stats}")
```

**解决方案：**
1. 启用JSON容错解析：配置中设置`enable_json_repair=True`
2. 检查模型响应格式
3. 更新JSON修复规则

#### 1.3 进程中断错误

**症状：** "Process interrupted"或"KeyboardInterrupt"

**诊断：**
```bash
# 检查检查点文件
ls -la ./kg_checkpoints/

# 查看进程状态
ps aux | grep python
```

**解决方案：**
1. 从检查点恢复：系统会自动恢复未完成任务
2. 调整进程数量：`export GFMRAG_NUM_PROCESSES=5`
3. 启用更频繁的检查点保存

### 2. 性能问题诊断

#### 2.1 处理速度慢

**诊断步骤：**
1. 检查系统资源使用率
2. 分析Token分配效率
3. 检查网络延迟

```python
import psutil
import time

# 系统资源监控
def monitor_resources():
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent
    
    print(f"CPU: {cpu}%, 内存: {memory}%, 磁盘: {disk}%")

# 网络延迟测试
def test_model_latency():
    from gfmrag.kg_construction.langchain_util import init_langchain_model
    
    model = init_langchain_model()
    start_time = time.time()
    
    # 发送测试请求
    response = model.invoke([{"role": "user", "content": "test"}])
    
    latency = time.time() - start_time
    print(f"模型响应延迟: {latency:.2f}秒")
```

#### 2.2 内存使用过高

**解决方案：**
1. 减少并发进程数
2. 启用批处理优化
3. 增加检查点频率以释放内存

```python
# 内存优化配置
config = KGProcessingConfig(
    max_workers=5,  # 减少并发度
    batch_size=10,  # 减少批量大小
    checkpoint_interval=5  # 增加检查点频率
)
```

## 扩展和定制

### 1. 自定义Token分配策略

```python
from gfmrag.token_manager import AdaptiveTokenManager

class CustomTokenManager(AdaptiveTokenManager):
    def get_optimal_allocation(self, text, task_type, custom_requirements=None):
        # 自定义分配逻辑
        allocation = super().get_optimal_allocation(text, task_type, custom_requirements)
        
        # 针对特定业务场景调整
        if "technical_document" in text.lower():
            allocation.ner_tokens *= 1.5
            allocation.triples_tokens *= 1.2
        
        return allocation
```

### 2. 自定义错误处理

```python
from gfmrag.enhanced_kg_models import EnhancedOpenIEModel

class CustomEnhancedOpenIE(EnhancedOpenIEModel):
    def _handle_specific_error(self, error, text):
        """处理特定类型的错误"""
        if "特定错误模式" in str(error):
            # 自定义错误处理逻辑
            return self._fallback_processing(text)
        else:
            return super()._handle_error(error, text)
```

### 3. 集成外部监控系统

#### 3.1 Prometheus集成

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from gfmrag.kg_monitoring import get_monitoring_system

# 创建Prometheus指标
processing_counter = Counter('kg_processing_total', 'Total KG processing requests')
processing_duration = Histogram('kg_processing_duration_seconds', 'KG processing duration')
active_tasks = Gauge('kg_active_tasks', 'Number of active KG tasks')

# 启动Prometheus HTTP服务器
start_http_server(8000)

# 集成到监控系统
monitoring = get_monitoring_system()

def prometheus_handler(alert):
    """将告警转发到Prometheus"""
    if alert.level.value == 'critical':
        processing_counter.inc()

monitoring.alert_manager.add_handler(prometheus_handler)
```

#### 3.2 ELK集成

```python
import json
import logging
from pythonjsonlogger import jsonlogger

# 配置JSON格式日志输出
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)

# 结构化日志记录
def log_kg_event(event_type, data):
    logger.info("KG事件", extra={
        "event_type": event_type,
        "timestamp": time.time(),
        "data": data
    })
```

## 最佳实践

### 1. 部署最佳实践

1. **分阶段部署**
   - 首先在测试环境验证
   - 小规模生产环境试运行
   - 逐步扩展到全量数据

2. **配置管理**
   - 使用版本控制管理配置文件
   - 环境变量集中管理
   - 配置变更记录和回滚机制

3. **监控覆盖**
   - 业务指标监控
   - 技术指标监控
   - 告警响应流程

### 2. 运维最佳实践

1. **定期检查**
   - 每日检查系统状态
   - 每周分析性能趋势
   - 每月优化配置参数

2. **故障预防**
   - 建立告警响应机制
   - 定期备份检查点
   - 准备故障恢复方案

3. **性能优化**
   - 根据实际负载调整配置
   - 定期清理历史数据
   - 优化资源使用效率

### 3. 开发最佳实践

1. **错误处理**
   - 优雅降级而非硬失败
   - 详细的错误日志记录
   - 合理的重试策略

2. **代码质量**
   - 完善的单元测试
   - 集成测试覆盖
   - 代码审查流程

3. **文档维护**
   - 及时更新部署文档
   - 记录配置变更原因
   - 维护故障处理手册

## 总结

通过本部署和运维指南，您可以：

1. **快速部署**：使用环境变量和配置文件快速启用错误处理功能
2. **实时监控**：通过监控系统实时追踪处理状态和性能指标
3. **故障排除**：使用诊断工具快速定位和解决问题
4. **性能优化**：根据监控数据持续优化系统配置
5. **扩展定制**：根据具体需求扩展和定制功能

系统设计遵循可观测性、可恢复性和可扩展性原则，确保知识图谱构建过程的稳定性和可靠性。

如需更多技术支持，请参考代码注释和测试用例，或通过监控系统获取实时运行状态。