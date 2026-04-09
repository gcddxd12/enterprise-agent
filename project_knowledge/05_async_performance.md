# 异步执行与性能优化

## 概述
异步执行是现代AI Agent系统的关键技术，能够显著提升系统响应速度和并发处理能力。本模块实现了完整的异步执行框架，包括任务调度、并行处理、流式响应和性能监控。

## 1. 异步执行器框架

### 1.1 异步任务定义

**概念**: 将工具调用封装为异步任务，支持优先级、超时和状态跟踪

**代码示例**:
```python
# async_executor.py 第20-52行：异步任务数据类
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional

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
```

**设计要点**:
1. **状态管理**: 明确的任务状态枚举
2. **超时控制**: 支持任务级别的超时设置
3. **优先级**: 多级优先级支持
4. **元数据**: 扩展性强的元数据字段

### 1.2 异步工具执行器

**代码示例**:
```python
# async_executor.py 第67-326行：AsyncToolExecutor类
class AsyncToolExecutor:
    """异步工具执行器"""
    
    def __init__(self, max_workers: int = 4, use_threadpool: bool = True):
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
    
    def submit_tool_call(self, tool_func: Callable, *args,
                        tool_name: str = None, timeout: float = 30.0,
                        priority: TaskPriority = TaskPriority.NORMAL,
                        **kwargs) -> str:
        """提交工具调用任务"""
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
        """等待多个任务完成，返回结果字典"""
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
```

**关键功能**:
1. **线程/进程池管理**: 支持线程池和进程池两种模式
2. **任务提交**: 统一的工具调用接口
3. **结果收集**: 支持等待多个任务完成
4. **性能统计**: 自动收集执行统计信息

## 2. 并行任务调度

### 2.1 并行任务调度器

**代码示例**:
```python
# async_executor.py 第330-480行：ParallelTaskScheduler类
class ParallelTaskScheduler:
    """并行任务调度器"""
    
    def __init__(self, executor: AsyncToolExecutor = None):
        self.executor = executor or AsyncToolExecutor(max_workers=4)
        self.task_groups: Dict[str, List[str]] = {}  # 任务组ID -> 任务ID列表
        self.group_counter = 0
        self.lock = threading.Lock()
    
    def submit_task_group(self, tasks: List[Dict[str, Any]],
                         group_id: str = None) -> str:
        """提交一组并行任务"""
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
    
    def get_group_status(self, group_id: str) -> Dict[str, Any]:
        """获取任务组状态"""
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
```

**调度策略**:
1. **任务组管理**: 批量任务提交和跟踪
2. **状态监控**: 实时获取任务组进度
3. **进度计算**: 基于任务状态计算完成百分比
4. **详细状态**: 每个任务的详细执行信息

## 3. 流式响应处理

### 3.1 流式响应处理器

**代码示例**:
```python
# async_executor.py 第484-674行：StreamingResponseHandler类
class StreamingResponseHandler:
    """流式响应处理器"""
    
    def __init__(self, executor: AsyncToolExecutor = None):
        self.executor = executor or AsyncToolExecutor(max_workers=4)
        self.callbacks: Dict[str, List[Callable]] = {
            "progress": [],
            "result": [],
            "error": [],
            "complete": []
        }
        self.active_streams: Dict[str, bool] = {}  # stream_id -> is_active
        self.lock = threading.Lock()
    
    def execute_with_streaming(self, task_defs: List[Dict[str, Any]],
                              stream_id: str = None) -> str:
        """执行任务并支持流式响应"""
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
```

**流式事件类型**:
1. **progress**: 任务进度更新
2. **result**: 单个工具结果返回
3. **error**: 工具执行错误
4. **complete**: 所有任务完成

## 4. 全局实例管理

### 4.1 单例模式管理

**代码示例**:
```python
# async_executor.py 第678-703行：全局实例管理
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
```

**设计优势**:
1. **延迟初始化**: 按需创建，避免不必要的资源占用
2. **全局访问**: 统一入口，便于管理和配置
3. **配置灵活**: 支持运行时调整参数

## 5. 与LangGraph工作流集成

### 5.1 异步执行节点

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py)

**代码示例**:
```python
# langgraph_agent_with_memory.py 第796-1065行：异步执行节点
def execution_node_async(state: AgentState) -> AgentState:
    """异步执行节点：并行执行规划的任务，收集结果"""
    import time
    start_time = time.time()
    
    print(f"[异步执行节点] 执行任务: {state['plan']}")
    
    if not state["plan"]:
        # ... 空计划处理
    
    # 初始化异步执行器（如果需要）
    global async_executor, parallel_scheduler
    if ASYNC_EXECUTOR_AVAILABLE:
        try:
            if async_executor is None:
                async_executor = get_async_executor(max_workers=4)
            if parallel_scheduler is None:
                parallel_scheduler = get_parallel_scheduler()
        except Exception as e:
            print(f"[WARN] 异步执行器初始化失败，将回退到同步执行: {e}")
            ASYNC_EXECUTOR_AVAILABLE = False
    
    if ASYNC_EXECUTOR_AVAILABLE:
        # 使用异步执行器并行执行任务
        results = {}
        tool_calls = []
        task_to_tool_map = {}  # 映射：任务描述 -> 工具函数和参数
        
        # 解析任务，构建工具调用列表
        for task in state["plan"]:
            if task.startswith("knowledge_search:"):
                query = task.replace("knowledge_search:", "").strip()
                tool_calls.append({
                    "func": knowledge_search.run,
                    "args": [query],
                    "tool_name": "knowledge_search",
                    "timeout": 30.0,
                    "priority": 2  # NORMAL priority
                })
                task_to_tool_map[task] = len(tool_calls) - 1
            
            elif task.startswith("ticket_query:"):
                ticket_id = task.replace("ticket_query:", "").strip()
                tool_calls.append({
                    "func": query_ticket_status.run,
                    "args": [ticket_id],
                    "tool_name": "ticket_query",
                    "timeout": 10.0,
                    "priority": 2
                })
                task_to_tool_map[task] = len(tool_calls) - 1
            
            # ... 其他工具类型类似
        
        if tool_calls:
            print(f"[异步执行节点] 准备并行执行 {len(tool_calls)} 个工具调用")
            
            try:
                # 并行执行工具调用
                from async_executor import run_tools_parallel
                parallel_results = run_tools_parallel(tool_calls, timeout=60.0)
                
                # 映射回任务结果
                for task, tool_idx in task_to_tool_map.items():
                    # ... 结果处理逻辑
                    
            except Exception as e:
                print(f"[ERROR] 并行工具调用失败，将回退到串行执行: {e}")
                # 回退到同步执行
                return execution_node(state)
        
        else:
            results = {}
    else:
        # 异步执行器不可用，回退到同步执行
        print("[WARN] 异步执行器不可用，使用同步执行节点")
        return execution_node(state)
    
    # ... 性能统计和监控
    
    return {**state, "tool_results": results, "step": "validation"}
```

**集成特点**:
1. **优雅降级**: 异步执行失败时自动回退到同步模式
2. **无缝集成**: 与现有工作流状态兼容
3. **性能监控**: 记录异步执行性能指标

## 6. 性能监控与优化

### 6.1 性能监控装饰器

**代码示例**:
```python
# async_executor.py 第760-781行：异步性能监控装饰器
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
```

### 6.2 性能统计指标

**代码示例**:
```python
# async_executor.py 第301-319行：获取执行器统计信息
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
```

**关键指标**:
1. **成功率**: 任务执行成功比例
2. **平均执行时间**: 工具调用平均耗时
3. **并发度**: 同时运行的最大任务数
4. **超时率**: 超时任务比例

## 7. 使用示例

### 7.1 基本使用

**代码示例**:
```python
# async_executor.py 第705-756行：辅助函数
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
```

### 7.2 实际应用场景

**场景1: 并行查询多个数据源**
```python
# 并行查询天气、股票、新闻
tool_calls = [
    {"func": weather_query.run, "args": ["北京"], "tool_name": "weather_query"},
    {"func": stock_query.run, "args": ["AAPL"], "tool_name": "stock_query"},
    {"func": news_search.run, "args": ["科技新闻"], "tool_name": "news_search"}
]

# 并行执行
results = run_tools_parallel(tool_calls, timeout=10.0)
```

**场景2: 流式响应复杂查询**
```python
def handle_stream_chunk(chunk: StreamChunk):
    """处理流式响应块"""
    if chunk.chunk_type == "result":
        print(f"收到结果: {chunk.content['tool_name']}")
    elif chunk.chunk_type == "progress":
        print(f"进度: {chunk.progress*100:.1f}%")

# 流式执行
stream_id = run_tools_streaming(tool_calls, callback=handle_stream_chunk)
```

## 8. 学习总结

### 关键技术要点
1. **异步任务抽象**: 统一的异步任务定义和管理
2. **并行调度算法**: 高效的并行任务调度策略
3. **流式响应协议**: 实时结果返回和进度反馈
4. **性能监控体系**: 全面的性能指标收集和分析

### 最佳实践
1. **线程池配置**: 根据任务类型选择合适的线程/进程池
2. **超时策略**: 设置合理的超时时间，避免资源占用
3. **错误处理**: 完善的错误处理和降级机制
4. **监控集成**: 与监控系统集成，实时跟踪性能

### 性能优化建议
1. **任务分组合并**: 将相关任务分组执行，减少调度开销
2. **内存管理**: 监控内存使用，避免内存泄漏
3. **连接池管理**: 对数据库/API连接使用连接池
4. **缓存策略**: 对频繁调用的工具结果进行缓存

### 常见陷阱
1. **线程安全**: 多线程环境下的数据竞争问题
2. **资源泄漏**: 未正确关闭线程池和连接
3. **死锁风险**: 不合理的锁使用导致死锁
4. **性能瓶颈**: 单点瓶颈影响整体性能

---

**相关文件**:
- [async_executor.py](e:\my_multi_agent\async_executor.py) - 异步执行器完整实现
- [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) - 工作流异步执行节点
- [app_simple.py](e:\my_multi_agent\app_simple.py) - Web界面流式响应集成

**下一步学习**: 监控系统 →