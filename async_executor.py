#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异步执行器模块
支持并行工具调用、流式响应和异步任务调度
"""

import asyncio
import concurrent.futures
import time
import threading
from typing import Dict, List, Any, Callable, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import json
import sys

# ========== 数据类定义 ==========

class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class AsyncTask:
    """异步任务定义"""
    task_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None  # 超时时间（秒）
    result: Any = None
    error: Optional[Exception] = None
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """流式响应块"""
    task_id: str
    chunk_type: str  # "progress", "result", "error", "complete"
    content: Any
    progress: Optional[float] = None  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)


# ========== 异步工具执行器 ==========

class AsyncToolExecutor:
    """异步工具执行器"""

    def __init__(self, max_workers: int = 4, use_threadpool: bool = True):
        """
        初始化异步工具执行器

        Args:
            max_workers: 最大工作线程/进程数
            use_threadpool: 是否使用线程池（True）或进程池（False）
        """
        self.max_workers = max_workers
        self.use_threadpool = use_threadpool
        self.executor = None
        self.running_tasks: Dict[str, AsyncTask] = {}
        self.task_counter = 0
        self.lock = threading.Lock()

        # 性能统计
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0,
            "concurrent_max": 0,
            "timeout_tasks": 0
        }

    def _get_executor(self):
        """获取或创建执行器"""
        if self.executor is None:
            if self.use_threadpool:
                self.executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="async_tool_worker"
                )
            else:
                self.executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.max_workers
                )
        return self.executor

    def submit_task(self, task: AsyncTask) -> str:
        """
        提交任务到执行器

        Args:
            task: 异步任务

        Returns:
            任务ID
        """
        with self.lock:
            # 更新任务状态
            task.status = TaskStatus.PENDING
            task.start_time = None
            task.end_time = None
            task.result = None
            task.error = None

            # 记录任务
            self.running_tasks[task.task_id] = task
            self.stats["total_tasks"] += 1

            # 更新并发计数
            concurrent_count = len([t for t in self.running_tasks.values()
                                   if t.status in [TaskStatus.PENDING, TaskStatus.RUNNING]])
            self.stats["concurrent_max"] = max(self.stats["concurrent_max"], concurrent_count)

        # 在后台执行任务
        self._execute_task_background(task)
        return task.task_id

    def _execute_task_background(self, task: AsyncTask):
        """在后台执行任务"""
        def task_wrapper():
            try:
                with self.lock:
                    task.status = TaskStatus.RUNNING
                    task.start_time = time.time()

                # 执行任务函数
                result = task.func(*task.args, **task.kwargs)

                with self.lock:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.end_time = time.time()

                    # 更新统计
                    self.stats["completed_tasks"] += 1
                    execution_time = task.end_time - task.start_time
                    self.stats["total_execution_time"] += execution_time
                    self.stats["avg_execution_time"] = (
                        self.stats["total_execution_time"] / self.stats["completed_tasks"]
                    )

            except concurrent.futures.TimeoutError:
                with self.lock:
                    task.status = TaskStatus.TIMEOUT
                    task.error = TimeoutError(f"任务超时: {task.timeout}s")
                    task.end_time = time.time()
                    self.stats["timeout_tasks"] += 1
                    self.stats["failed_tasks"] += 1
            except Exception as e:
                with self.lock:
                    task.status = TaskStatus.FAILED
                    task.error = e
                    task.end_time = time.time()
                    self.stats["failed_tasks"] += 1
            finally:
                # 确保任务从运行列表中移除（如果已完成）
                with self.lock:
                    if task.task_id in self.running_tasks and task.status in [
                        TaskStatus.COMPLETED, TaskStatus.FAILED,
                        TaskStatus.TIMEOUT, TaskStatus.CANCELLED
                    ]:
                        del self.running_tasks[task.task_id]

        # 提交到执行器
        executor = self._get_executor()
        if task.timeout:
            future = executor.submit(task_wrapper)
            try:
                # 设置超时
                future.result(timeout=task.timeout)
            except concurrent.futures.TimeoutError:
                with self.lock:
                    if task.task_id in self.running_tasks:
                        task.status = TaskStatus.TIMEOUT
                        task.error = TimeoutError(f"任务超时: {task.timeout}s")
        else:
            executor.submit(task_wrapper)

    def submit_tool_call(self, tool_func: Callable, *args,
                        tool_name: str = None, timeout: float = 30.0,
                        priority: TaskPriority = TaskPriority.NORMAL,
                        **kwargs) -> str:
        """
        提交工具调用任务

        Args:
            tool_func: 工具函数
            *args: 工具参数
            tool_name: 工具名称（用于标识）
            timeout: 超时时间（秒）
            priority: 任务优先级
            **kwargs: 工具关键字参数

        Returns:
            任务ID
        """
        # 生成任务ID
        with self.lock:
            self.task_counter += 1
            task_id = f"tool_task_{self.task_counter}_{int(time.time())}"

        if tool_name is None:
            tool_name = tool_func.__name__ if hasattr(tool_func, '__name__') else "unknown_tool"

        # 创建任务
        task = AsyncTask(
            task_id=task_id,
            func=tool_func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            metadata={
                "tool_name": tool_name,
                "submitted_time": time.time(),
                "args": args,
                "kwargs": kwargs
            }
        )

        return self.submit_task(task)

    def wait_for_tasks(self, task_ids: List[str], timeout: float = None) -> Dict[str, Any]:
        """
        等待多个任务完成

        Args:
            task_ids: 任务ID列表
            timeout: 总等待超时时间（秒）

        Returns:
            任务结果字典 {task_id: result}
        """
        start_time = time.time()
        results = {}

        while task_ids:
            # 检查超时
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"等待任务超时: {timeout}s")

            completed_ids = []
            for task_id in task_ids:
                task = self.get_task_status(task_id)
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED,
                                 TaskStatus.TIMEOUT, TaskStatus.CANCELLED]:
                    if task.status == TaskStatus.COMPLETED:
                        results[task_id] = task.result
                    else:
                        results[task_id] = task.error
                    completed_ids.append(task_id)

            # 移除已完成的任务
            for task_id in completed_ids:
                task_ids.remove(task_id)

            # 如果还有任务未完成，等待一小段时间
            if task_ids:
                time.sleep(0.1)

        return results

    def get_task_status(self, task_id: str) -> Optional[AsyncTask]:
        """获取任务状态"""
        with self.lock:
            return self.running_tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self.lock:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                task.end_time = time.time()
                return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """获取执行器统计信息"""
        with self.lock:
            stats_copy = self.stats.copy()
            stats_copy["currently_running"] = len([
                t for t in self.running_tasks.values()
                if t.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
            ])
            stats_copy["executor_type"] = "threadpool" if self.use_threadpool else "processpool"
            stats_copy["max_workers"] = self.max_workers

            # 计算成功率
            total_completed = stats_copy["completed_tasks"] + stats_copy["failed_tasks"]
            if total_completed > 0:
                stats_copy["success_rate"] = stats_copy["completed_tasks"] / total_completed
            else:
                stats_copy["success_rate"] = 0.0

        return stats_copy

    def shutdown(self, wait: bool = True):
        """关闭执行器"""
        if self.executor:
            self.executor.shutdown(wait=wait)
            self.executor = None


# ========== 并行任务调度器 ==========

class ParallelTaskScheduler:
    """并行任务调度器"""

    def __init__(self, executor: AsyncToolExecutor = None):
        """
        初始化并行任务调度器

        Args:
            executor: 异步工具执行器实例（如果为None则创建新的）
        """
        self.executor = executor or AsyncToolExecutor(max_workers=4)
        self.task_groups: Dict[str, List[str]] = {}  # 任务组ID -> 任务ID列表
        self.group_counter = 0
        self.lock = threading.Lock()

    def submit_task_group(self, tasks: List[Dict[str, Any]],
                         group_id: str = None) -> str:
        """
        提交一组并行任务

        Args:
            tasks: 任务列表，每个任务是一个字典，包含：
                - func: 任务函数
                - args: 参数列表
                - kwargs: 关键字参数字典
                - tool_name: 工具名称（可选）
                - timeout: 超时时间（可选）
                - priority: 优先级（可选）
            group_id: 任务组ID（如果为None则自动生成）

        Returns:
            任务组ID
        """
        if group_id is None:
            with self.lock:
                self.group_counter += 1
                group_id = f"task_group_{self.group_counter}_{int(time.time())}"

        task_ids = []
        for task_def in tasks:
            # 提取任务定义
            func = task_def.get("func")
            args = task_def.get("args", [])
            kwargs = task_def.get("kwargs", {})
            tool_name = task_def.get("tool_name")
            timeout = task_def.get("timeout", 30.0)
            priority = task_def.get("priority", TaskPriority.NORMAL)

            if not func:
                raise ValueError("任务定义必须包含'func'字段")

            # 提交任务
            task_id = self.executor.submit_tool_call(
                func, *args,
                tool_name=tool_name,
                timeout=timeout,
                priority=priority,
                **kwargs
            )
            task_ids.append(task_id)

        # 记录任务组
        with self.lock:
            self.task_groups[group_id] = task_ids

        return group_id

    def wait_for_group(self, group_id: str, timeout: float = None) -> Dict[str, Any]:
        """
        等待任务组完成

        Args:
            group_id: 任务组ID
            timeout: 超时时间（秒）

        Returns:
            任务结果字典 {task_id: result}
        """
        with self.lock:
            if group_id not in self.task_groups:
                raise ValueError(f"任务组不存在: {group_id}")
            task_ids = self.task_groups[group_id].copy()

        # 等待所有任务完成
        return self.executor.wait_for_tasks(task_ids, timeout)

    def get_group_status(self, group_id: str) -> Dict[str, Any]:
        """
        获取任务组状态

        Args:
            group_id: 任务组ID

        Returns:
            状态字典
        """
        with self.lock:
            if group_id not in self.task_groups:
                return {"error": "任务组不存在"}

            task_ids = self.task_groups[group_id]
            total_tasks = len(task_ids)

            # 统计各状态任务数量
            status_counts = {status.value: 0 for status in TaskStatus}
            task_statuses = {}

            for task_id in task_ids:
                task = self.executor.get_task_status(task_id)
                if task:
                    status_counts[task.status.value] += 1
                    task_statuses[task_id] = {
                        "status": task.status.value,
                        "progress": task.metadata.get("progress", 0.0),
                        "start_time": task.start_time,
                        "tool_name": task.metadata.get("tool_name", "unknown")
                    }

            # 计算完成进度
            completed_tasks = status_counts[TaskStatus.COMPLETED.value]
            failed_tasks = status_counts[TaskStatus.FAILED.value] + status_counts[TaskStatus.TIMEOUT.value]
            progress = (completed_tasks + failed_tasks) / total_tasks if total_tasks > 0 else 0.0

            return {
                "group_id": group_id,
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "progress": progress,
                "status_counts": status_counts,
                "task_statuses": task_statuses
            }

    def cancel_group(self, group_id: str) -> bool:
        """取消任务组中的所有任务"""
        with self.lock:
            if group_id not in self.task_groups:
                return False

            task_ids = self.task_groups[group_id]
            cancelled_count = 0

            for task_id in task_ids:
                if self.executor.cancel_task(task_id):
                    cancelled_count += 1

            # 从任务组中移除
            del self.task_groups[group_id]

            return cancelled_count > 0


# ========== 流式响应处理器 ==========

class StreamingResponseHandler:
    """流式响应处理器"""

    def __init__(self, executor: AsyncToolExecutor = None):
        """
        初始化流式响应处理器

        Args:
            executor: 异步工具执行器实例
        """
        self.executor = executor or AsyncToolExecutor(max_workers=4)
        self.callbacks: Dict[str, List[Callable]] = {
            "progress": [],
            "result": [],
            "error": [],
            "complete": []
        }
        self.active_streams: Dict[str, bool] = {}  # stream_id -> is_active
        self.lock = threading.Lock()

    def register_callback(self, event_type: str, callback: Callable):
        """
        注册事件回调

        Args:
            event_type: 事件类型 ("progress", "result", "error", "complete")
            callback: 回调函数，接受 StreamChunk 参数
        """
        if event_type not in self.callbacks:
            raise ValueError(f"不支持的事件类型: {event_type}")

        with self.lock:
            self.callbacks[event_type].append(callback)

    def _emit_event(self, chunk: StreamChunk):
        """触发事件"""
        event_type = chunk.chunk_type
        if event_type not in self.callbacks:
            return

        callbacks = self.callbacks[event_type].copy()  # 复制以避免在迭代中修改

        for callback in callbacks:
            try:
                callback(chunk)
            except Exception as e:
                print(f"[WARN] 流式回调执行失败: {e}")

    def execute_with_streaming(self, task_defs: List[Dict[str, Any]],
                              stream_id: str = None) -> str:
        """
        执行任务并支持流式响应

        Args:
            task_defs: 任务定义列表
            stream_id: 流ID（如果为None则自动生成）

        Returns:
            流ID
        """
        if stream_id is None:
            stream_id = f"stream_{int(time.time())}_{hash(str(task_defs)) % 10000}"

        with self.lock:
            self.active_streams[stream_id] = True

        # 在后台执行流式任务
        threading.Thread(
            target=self._execute_streaming_background,
            args=(task_defs, stream_id),
            daemon=True
        ).start()

        return stream_id

    def _execute_streaming_background(self, task_defs: List[Dict[str, Any]], stream_id: str):
        """在后台执行流式任务"""
        try:
            # 创建调度器
            scheduler = ParallelTaskScheduler(self.executor)

            # 发送开始事件
            self._emit_event(StreamChunk(
                task_id=stream_id,
                chunk_type="progress",
                content="开始执行并行任务",
                progress=0.0
            ))

            # 提交任务组
            group_id = scheduler.submit_task_group(task_defs)

            # 定期检查进度并发送更新
            total_tasks = len(task_defs)
            completed_tasks = set()

            while len(completed_tasks) < total_tasks:
                # 检查流是否仍然活跃
                with self.lock:
                    if not self.active_streams.get(stream_id, False):
                        break

                # 获取任务组状态
                group_status = scheduler.get_group_status(group_id)

                # 检查新完成的任务
                task_statuses = group_status.get("task_statuses", {})
                for task_id, task_info in task_statuses.items():
                    if task_id not in completed_tasks:
                        status = task_info.get("status")
                        tool_name = task_info.get("tool_name", "unknown")

                        if status == TaskStatus.COMPLETED.value:
                            # 发送结果事件
                            task = self.executor.get_task_status(task_id)
                            if task and task.result is not None:
                                self._emit_event(StreamChunk(
                                    task_id=stream_id,
                                    chunk_type="result",
                                    content={
                                        "task_id": task_id,
                                        "tool_name": tool_name,
                                        "result": task.result
                                    },
                                    progress=len(completed_tasks) / total_tasks
                                ))
                            completed_tasks.add(task_id)

                        elif status in [TaskStatus.FAILED.value, TaskStatus.TIMEOUT.value]:
                            # 发送错误事件
                            task = self.executor.get_task_status(task_id)
                            if task and task.error is not None:
                                self._emit_event(StreamChunk(
                                    task_id=stream_id,
                                    chunk_type="error",
                                    content={
                                        "task_id": task_id,
                                        "tool_name": tool_name,
                                        "error": str(task.error)
                                    },
                                    progress=len(completed_tasks) / total_tasks
                                ))
                            completed_tasks.add(task_id)

                # 发送进度事件
                progress = group_status.get("progress", 0.0)
                self._emit_event(StreamChunk(
                    task_id=stream_id,
                    chunk_type="progress",
                    content=f"任务进度: {progress*100:.1f}%",
                    progress=progress
                ))

                # 等待一小段时间
                time.sleep(0.5)

            # 发送完成事件
            self._emit_event(StreamChunk(
                task_id=stream_id,
                chunk_type="complete",
                content="所有任务执行完成",
                progress=1.0
            ))

        except Exception as e:
            # 发送错误事件
            self._emit_event(StreamChunk(
                task_id=stream_id,
                chunk_type="error",
                content=f"流式执行失败: {str(e)}",
                progress=0.0
            ))
        finally:
            # 标记流为不活跃
            with self.lock:
                if stream_id in self.active_streams:
                    del self.active_streams[stream_id]

    def stop_stream(self, stream_id: str) -> bool:
        """停止流式响应"""
        with self.lock:
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
                return True
        return False

    def is_stream_active(self, stream_id: str) -> bool:
        """检查流是否活跃"""
        with self.lock:
            return self.active_streams.get(stream_id, False)


# ========== 全局异步执行器实例 ==========

# 创建全局异步执行器实例
_async_executor = None
_parallel_scheduler = None
_streaming_handler = None

def get_async_executor(max_workers: int = 4) -> AsyncToolExecutor:
    """获取全局异步执行器实例"""
    global _async_executor
    if _async_executor is None:
        _async_executor = AsyncToolExecutor(max_workers=max_workers)
    return _async_executor

def get_parallel_scheduler() -> ParallelTaskScheduler:
    """获取全局并行任务调度器实例"""
    global _parallel_scheduler
    if _parallel_scheduler is None:
        _parallel_scheduler = ParallelTaskScheduler(get_async_executor())
    return _parallel_scheduler

def get_streaming_handler() -> StreamingResponseHandler:
    """获取全局流式响应处理器实例"""
    global _streaming_handler
    if _streaming_handler is None:
        _streaming_handler = StreamingResponseHandler(get_async_executor())
    return _streaming_handler


# ========== 辅助函数 ==========

def run_tools_parallel(tool_calls: List[Dict[str, Any]], timeout: float = 30.0) -> Dict[str, Any]:
    """
    并行运行多个工具调用

    Args:
        tool_calls: 工具调用列表，每个元素包含：
            - func: 工具函数
            - args: 参数列表
            - kwargs: 关键字参数字典
            - tool_name: 工具名称（可选）
        timeout: 超时时间（秒）

    Returns:
        工具调用结果字典 {task_id: result}
    """
    scheduler = get_parallel_scheduler()

    # 提交任务组
    group_id = scheduler.submit_task_group(tool_calls)

    # 等待任务组完成
    try:
        results = scheduler.wait_for_group(group_id, timeout)
        return results
    except Exception as e:
        raise RuntimeError(f"并行工具调用失败: {e}")


def run_tools_streaming(tool_calls: List[Dict[str, Any]],
                       callback: Callable = None) -> str:
    """
    以流式方式运行多个工具调用

    Args:
        tool_calls: 工具调用列表
        callback: 可选的回调函数，接收 StreamChunk

    Returns:
        流ID
    """
    handler = get_streaming_handler()

    if callback:
        # 注册结果回调
        handler.register_callback("result", callback)

    # 执行流式任务
    stream_id = handler.execute_with_streaming(tool_calls)
    return stream_id


# ========== 性能监控装饰器 ==========

def async_performance_monitor(func):
    """异步性能监控装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # 记录性能指标（这里可以集成到监控系统）
            if hasattr(func, '__name__'):
                tool_name = func.__name__
                print(f"[异步性能监控] {tool_name}: {duration:.3f}s")

            return result
        except Exception as e:
            duration = time.time() - start_time
            print(f"[异步性能监控] {func.__name__} 失败: {e} (耗时: {duration:.3f}s)")
            raise

    return wrapper


# ========== 测试函数 ==========

if __name__ == "__main__":
    """模块测试"""
    print("=== 异步执行器模块测试 ===")

    # 测试工具函数
    def mock_tool(name: str, delay: float = 1.0) -> str:
        """模拟工具函数"""
        time.sleep(delay)
        return f"工具 {name} 执行完成，延迟 {delay}s"

    def failing_tool(name: str) -> str:
        """模拟失败的工具函数"""
        time.sleep(0.5)
        raise ValueError(f"工具 {name} 执行失败")

    # 创建执行器
    executor = AsyncToolExecutor(max_workers=2)

    # 测试单个任务提交
    print("\n1. 测试单个任务提交:")
    task_id = executor.submit_tool_call(mock_tool, "test_tool_1", 1.0, tool_name="mock_tool")
    print(f"   任务ID: {task_id}")

    # 等待任务完成
    time.sleep(1.5)
    task = executor.get_task_status(task_id)
    print(f"   任务状态: {task.status.value}, 结果: {task.result}")

    # 测试并行任务
    print("\n2. 测试并行任务:")
    scheduler = ParallelTaskScheduler(executor)

    task_defs = [
        {"func": mock_tool, "args": ["tool_1", 2.0], "tool_name": "mock_tool_1"},
        {"func": mock_tool, "args": ["tool_2", 1.5], "tool_name": "mock_tool_2"},
        {"func": mock_tool, "args": ["tool_3", 1.0], "tool_name": "mock_tool_3"},
    ]

    group_id = scheduler.submit_task_group(task_defs)
    print(f"   任务组ID: {group_id}")

    # 获取任务组状态
    time.sleep(1.0)
    status = scheduler.get_group_status(group_id)
    print(f"   任务组状态: {status['progress']*100:.1f}% 完成")

    # 等待任务组完成
    results = scheduler.wait_for_group(group_id, timeout=5.0)
    print(f"   任务结果: {len(results)} 个任务完成")

    # 获取执行器统计信息
    stats = executor.get_statistics()
    print(f"\n3. 执行器统计:")
    print(f"   总任务数: {stats['total_tasks']}")
    print(f"   成功任务: {stats['completed_tasks']}")
    print(f"   失败任务: {stats['failed_tasks']}")
    print(f"   平均执行时间: {stats['avg_execution_time']:.3f}s")
    print(f"   成功率: {stats['success_rate']*100:.1f}%")

    # 测试流式响应
    print("\n4. 测试流式响应:")

    def stream_callback(chunk: StreamChunk):
        """流式回调函数"""
        print(f"   [流式回调] {chunk.chunk_type}: {chunk.content}")

    handler = StreamingResponseHandler(executor)
    handler.register_callback("progress", stream_callback)
    handler.register_callback("result", stream_callback)
    handler.register_callback("error", stream_callback)
    handler.register_callback("complete", stream_callback)

    stream_task_defs = [
        {"func": mock_tool, "args": ["stream_tool_1", 1.0], "tool_name": "stream_tool_1"},
        {"func": mock_tool, "args": ["stream_tool_2", 2.0], "tool_name": "stream_tool_2"},
    ]

    stream_id = handler.execute_with_streaming(stream_task_defs)
    print(f"   流ID: {stream_id}")

    # 等待流式任务完成
    time.sleep(3.0)

    # 关闭执行器
    executor.shutdown()

    print("\n[SUCCESS] 异步执行器模块测试通过")