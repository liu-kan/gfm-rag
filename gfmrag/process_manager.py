"""
进程容错与恢复机制

该模块实现了全面的多进程容错策略，包括：
- 进程状态监控和管理
- 断点续传机制
- 异常恢复和重试
- 进度持久化
"""

import os
import json
import time
import signal
import logging
import threading
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
import pickle
import hashlib

logger = logging.getLogger(__name__)


class ProcessState(Enum):
    """进程状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    RETRYING = "retrying"


class TaskStatus(Enum):
    """任务状态枚举"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    input_data: Any
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    attempts: int = 0
    last_attempt_time: Optional[float] = None
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'attempts': self.attempts,
            'last_attempt_time': self.last_attempt_time,
            'processing_time': self.processing_time
        }


@dataclass
class ProcessStats:
    """进程统计信息"""
    process_id: int
    start_time: float
    end_time: Optional[float] = None
    tasks_processed: int = 0
    tasks_failed: int = 0
    state: ProcessState = ProcessState.PENDING
    error_message: Optional[str] = None


@dataclass
class CheckpointData:
    """检查点数据"""
    checkpoint_id: str
    timestamp: float
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    task_states: Dict[str, TaskInfo]
    process_stats: Dict[int, ProcessStats]
    metadata: Dict[str, Any]


class ProgressManager:
    """进度管理器 - 负责进度持久化和恢复"""
    
    def __init__(self, checkpoint_dir: str, job_id: str):
        """初始化进度管理器
        
        Args:
            checkpoint_dir: 检查点目录
            job_id: 任务ID
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.job_id = job_id
        self.checkpoint_file = self.checkpoint_dir / f"{job_id}_checkpoint.json"
        self.lock = threading.Lock()
        
        # 确保目录存在
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"进度管理器初始化 - 检查点文件: {self.checkpoint_file}")
    
    def save_checkpoint(self, checkpoint_data: CheckpointData):
        """保存检查点"""
        with self.lock:
            try:
                # 序列化任务状态
                serializable_data = {
                    'checkpoint_id': checkpoint_data.checkpoint_id,
                    'timestamp': checkpoint_data.timestamp,
                    'total_tasks': checkpoint_data.total_tasks,
                    'completed_tasks': checkpoint_data.completed_tasks,
                    'failed_tasks': checkpoint_data.failed_tasks,
                    'task_states': {
                        task_id: {
                            'task_id': task.task_id,
                            'status': task.status.value,
                            'result': self._serialize_result(task.result),
                            'error': task.error,
                            'attempts': task.attempts,
                            'last_attempt_time': task.last_attempt_time,
                            'processing_time': task.processing_time
                        }
                        for task_id, task in checkpoint_data.task_states.items()
                    },
                    'process_stats': {
                        str(pid): {
                            'process_id': stats.process_id,
                            'start_time': stats.start_time,
                            'end_time': stats.end_time,
                            'tasks_processed': stats.tasks_processed,
                            'tasks_failed': stats.tasks_failed,
                            'state': stats.state.value,
                            'error_message': stats.error_message
                        }
                        for pid, stats in checkpoint_data.process_stats.items()
                    },
                    'metadata': checkpoint_data.metadata
                }
                
                # 写入临时文件，然后原子性重命名
                temp_file = self.checkpoint_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_data, f, indent=2, ensure_ascii=False)
                
                temp_file.replace(self.checkpoint_file)
                logger.debug(f"检查点已保存: {checkpoint_data.checkpoint_id}")
                
            except Exception as e:
                logger.error(f"保存检查点失败: {e}")
                raise
    
    def load_checkpoint(self) -> Optional[CheckpointData]:
        """加载检查点"""
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 反序列化任务状态
            task_states = {}
            for task_id, task_data in data['task_states'].items():
                task_states[task_id] = TaskInfo(
                    task_id=task_data['task_id'],
                    input_data=None,  # 输入数据需要从其他地方重新获取
                    status=TaskStatus(task_data['status']),
                    result=self._deserialize_result(task_data['result']),
                    error=task_data['error'],
                    attempts=task_data['attempts'],
                    last_attempt_time=task_data['last_attempt_time'],
                    processing_time=task_data['processing_time']
                )
            
            # 反序列化进程统计
            process_stats = {}
            for pid_str, stats_data in data['process_stats'].items():
                process_stats[int(pid_str)] = ProcessStats(
                    process_id=stats_data['process_id'],
                    start_time=stats_data['start_time'],
                    end_time=stats_data['end_time'],
                    tasks_processed=stats_data['tasks_processed'],
                    tasks_failed=stats_data['tasks_failed'],
                    state=ProcessState(stats_data['state']),
                    error_message=stats_data['error_message']
                )
            
            checkpoint_data = CheckpointData(
                checkpoint_id=data['checkpoint_id'],
                timestamp=data['timestamp'],
                total_tasks=data['total_tasks'],
                completed_tasks=data['completed_tasks'],
                failed_tasks=data['failed_tasks'],
                task_states=task_states,
                process_stats=process_stats,
                metadata=data['metadata']
            )
            
            logger.info(f"检查点已加载: {checkpoint_data.checkpoint_id}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return None
    
    def _serialize_result(self, result: Any) -> Any:
        """序列化结果"""
        if result is None:
            return None
        
        try:
            # 尝试JSON序列化
            json.dumps(result)
            return result
        except (TypeError, ValueError):
            # 使用pickle序列化复杂对象
            try:
                return {'__pickled__': pickle.dumps(result).hex()}
            except Exception:
                return {'__error__': str(result)}
    
    def _deserialize_result(self, result: Any) -> Any:
        """反序列化结果"""
        if result is None:
            return None
        
        if isinstance(result, dict):
            if '__pickled__' in result:
                try:
                    return pickle.loads(bytes.fromhex(result['__pickled__']))
                except Exception:
                    return None
            elif '__error__' in result:
                return result['__error__']
        
        return result


class ResilientProcessPool:
    """弹性进程池 - 具备容错和恢复能力的进程池"""
    
    def __init__(
        self,
        max_workers: int = None,
        checkpoint_dir: str = "./checkpoints",
        job_id: Optional[str] = None,
        checkpoint_interval: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """初始化弹性进程池
        
        Args:
            max_workers: 最大工作进程数
            checkpoint_dir: 检查点目录
            job_id: 任务ID
            checkpoint_interval: 检查点保存间隔
            max_retries: 最大重试次数
            retry_delay: 重试延迟
        """
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.job_id = job_id or f"job_{int(time.time())}"
        self.checkpoint_interval = checkpoint_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 初始化组件
        self.progress_manager = ProgressManager(checkpoint_dir, self.job_id)
        self.task_queue: List[TaskInfo] = []
        self.completed_tasks: Dict[str, TaskInfo] = {}
        self.failed_tasks: Dict[str, TaskInfo] = {}
        self.process_stats: Dict[int, ProcessStats] = {}
        
        # 状态管理
        self.is_running = False
        self.should_stop = False
        self.last_checkpoint_time = time.time()
        
        # 信号处理
        self._setup_signal_handlers()
        
        logger.info(f"弹性进程池初始化 - 工作进程: {self.max_workers}, 任务ID: {self.job_id}")
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.warning(f"接收到信号 {signum}, 开始优雅关闭...")
            self.should_stop = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def submit_tasks(self, tasks: List[Tuple[str, Any]]) -> None:
        """提交任务列表
        
        Args:
            tasks: 任务列表，每个元素为 (task_id, input_data)
        """
        for task_id, input_data in tasks:
            task_info = TaskInfo(
                task_id=task_id,
                input_data=input_data,
                status=TaskStatus.QUEUED
            )
            self.task_queue.append(task_info)
        
        logger.info(f"已提交 {len(tasks)} 个任务")
    
    def resume_from_checkpoint(self) -> bool:
        """从检查点恢复
        
        Returns:
            bool: 是否成功恢复
        """
        checkpoint_data = self.progress_manager.load_checkpoint()
        if not checkpoint_data:
            logger.info("未找到检查点，从头开始")
            return False
        
        # 恢复任务状态
        for task_id, task_info in checkpoint_data.task_states.items():
            if task_info.status == TaskStatus.COMPLETED:
                self.completed_tasks[task_id] = task_info
            elif task_info.status == TaskStatus.FAILED:
                if task_info.attempts < self.max_retries:
                    # 重新排队重试
                    task_info.status = TaskStatus.QUEUED
                    self.task_queue.append(task_info)
                else:
                    self.failed_tasks[task_id] = task_info
            else:
                # 未完成的任务重新排队
                task_info.status = TaskStatus.QUEUED
                self.task_queue.append(task_info)
        
        # 恢复进程统计
        self.process_stats = checkpoint_data.process_stats
        
        logger.info(f"从检查点恢复: 完成 {len(self.completed_tasks)}, "
                   f"失败 {len(self.failed_tasks)}, "
                   f"队列 {len(self.task_queue)}")
        
        return True
    
    def process_tasks(self, worker_func: Callable[[Any], Any]) -> Dict[str, Any]:
        """处理任务
        
        Args:
            worker_func: 工作函数
            
        Returns:
            Dict[str, Any]: 处理结果统计
        """
        self.is_running = True
        start_time = time.time()
        
        try:
            # 尝试从检查点恢复
            self.resume_from_checkpoint()
            
            # 处理任务
            self._process_with_executor(worker_func)
            
        except KeyboardInterrupt:
            logger.warning("收到中断信号，正在保存进度...")
            self._save_current_progress()
            raise
        except Exception as e:
            logger.error(f"处理任务时发生错误: {e}")
            self._save_current_progress()
            raise
        finally:
            self.is_running = False
        
        end_time = time.time()
        
        # 最终统计
        stats = {
            'total_tasks': len(self.task_queue) + len(self.completed_tasks) + len(self.failed_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'success_rate': len(self.completed_tasks) / (len(self.completed_tasks) + len(self.failed_tasks)) if (len(self.completed_tasks) + len(self.failed_tasks)) > 0 else 0,
            'total_time': end_time - start_time,
            'throughput': len(self.completed_tasks) / (end_time - start_time) if (end_time - start_time) > 0 else 0
        }
        
        logger.info(f"任务处理完成 - 统计: {stats}")
        return stats
    
    def _process_with_executor(self, worker_func: Callable[[Any], Any]):
        """使用执行器处理任务"""
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_task = {}
            active_tasks = {}
            
            while self.task_queue or future_to_task:
                if self.should_stop:
                    logger.info("收到停止信号，取消剩余任务...")
                    break
                
                # 提交新任务
                while len(future_to_task) < self.max_workers and self.task_queue:
                    task = self.task_queue.pop(0)
                    task.status = TaskStatus.PROCESSING
                    task.last_attempt_time = time.time()
                    
                    future = executor.submit(self._safe_worker, worker_func, task.input_data)
                    future_to_task[future] = task
                    active_tasks[task.task_id] = task
                
                # 等待任务完成
                if future_to_task:
                    for future in as_completed(future_to_task, timeout=1.0):
                        task = future_to_task.pop(future)
                        active_tasks.pop(task.task_id, None)
                        
                        try:
                            result = future.result()
                            if result.get('success', False):
                                task.status = TaskStatus.COMPLETED
                                task.result = result.get('data')
                                task.processing_time = time.time() - task.last_attempt_time
                                self.completed_tasks[task.task_id] = task
                            else:
                                self._handle_task_failure(task, result.get('error', 'Unknown error'))
                        
                        except Exception as e:
                            self._handle_task_failure(task, str(e))
                
                # 定期保存检查点
                if time.time() - self.last_checkpoint_time > self.checkpoint_interval:
                    self._save_current_progress()
                    self.last_checkpoint_time = time.time()
    
    def _safe_worker(self, worker_func: Callable[[Any], Any], input_data: Any) -> Dict[str, Any]:
        """安全的工作函数包装器"""
        try:
            result = worker_func(input_data)
            return {'success': True, 'data': result}
        except Exception as e:
            logger.error(f"工作函数执行失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_task_failure(self, task: TaskInfo, error_message: str):
        """处理任务失败"""
        task.attempts += 1
        task.error = error_message
        
        if task.attempts < self.max_retries:
            logger.warning(f"任务 {task.task_id} 失败，重试 {task.attempts}/{self.max_retries}: {error_message}")
            task.status = TaskStatus.QUEUED
            self.task_queue.append(task)
            time.sleep(self.retry_delay)
        else:
            logger.error(f"任务 {task.task_id} 最终失败: {error_message}")
            task.status = TaskStatus.FAILED
            self.failed_tasks[task.task_id] = task
    
    def _save_current_progress(self):
        """保存当前进度"""
        checkpoint_id = f"{self.job_id}_{int(time.time())}"
        
        # 合并所有任务状态
        all_tasks = {}
        all_tasks.update(self.completed_tasks)
        all_tasks.update(self.failed_tasks)
        
        # 添加队列中的任务
        for task in self.task_queue:
            all_tasks[task.task_id] = task
        
        checkpoint_data = CheckpointData(
            checkpoint_id=checkpoint_id,
            timestamp=time.time(),
            total_tasks=len(all_tasks),
            completed_tasks=len(self.completed_tasks),
            failed_tasks=len(self.failed_tasks),
            task_states=all_tasks,
            process_stats=self.process_stats,
            metadata={
                'max_workers': self.max_workers,
                'max_retries': self.max_retries,
                'retry_delay': self.retry_delay
            }
        )
        
        self.progress_manager.save_checkpoint(checkpoint_data)


@contextmanager
def resilient_processing(
    max_workers: int = None,
    checkpoint_dir: str = "./checkpoints",
    job_id: Optional[str] = None,
    **kwargs
):
    """弹性处理上下文管理器"""
    pool = ResilientProcessPool(
        max_workers=max_workers,
        checkpoint_dir=checkpoint_dir,
        job_id=job_id,
        **kwargs
    )
    
    try:
        yield pool
    finally:
        # 确保保存最终进度
        if pool.is_running:
            pool._save_current_progress()


def create_task_id(input_data: Any) -> str:
    """根据输入数据创建任务ID"""
    # 使用输入数据的哈希值作为任务ID
    data_str = str(input_data)
    return hashlib.md5(data_str.encode()).hexdigest()[:16]


def batch_process_with_recovery(
    input_data: List[Any],
    worker_func: Callable[[Any], Any],
    max_workers: int = None,
    checkpoint_dir: str = "./checkpoints",
    job_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """带恢复机制的批量处理便捷函数
    
    Args:
        input_data: 输入数据列表
        worker_func: 工作函数
        max_workers: 最大工作进程数  
        checkpoint_dir: 检查点目录
        job_id: 任务ID
        **kwargs: 其他参数
        
    Returns:
        Dict[str, Any]: 处理结果和统计信息
    """
    with resilient_processing(
        max_workers=max_workers,
        checkpoint_dir=checkpoint_dir,
        job_id=job_id,
        **kwargs
    ) as pool:
        
        # 创建任务列表
        tasks = [(create_task_id(data), data) for data in input_data]
        pool.submit_tasks(tasks)
        
        # 处理任务
        stats = pool.process_tasks(worker_func)
        
        # 返回结果
        return {
            'stats': stats,
            'completed_tasks': pool.completed_tasks,
            'failed_tasks': pool.failed_tasks
        }