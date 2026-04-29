# 异步执行与性能优化

## 概述
异步执行模块（`async_executor.py`）提供独立的并行任务调度能力，包括线程池管理、任务优先级、超时控制和流式响应。**注意：在v2.0标准ReAct架构中，该模块作为独立工具库存在，并未直接集成到主Agent工作流中**（v1.0旧版曾将异步执行器嵌入execution_node_async）。

## 1. 模块定位（v2.0）

**当前状态**: `async_executor.py` 是独立的工具模块，可在以下场景使用：
- 需要并行调用多个工具时手动使用
- 批量处理的离线任务
- 未来扩展多工具并行场景

**不在主流程中的原因**: v2.0的标准ReAct循环是串行的（LLM推理 → 工具调用 → 观察 → 推理），每次只调用一个或少量工具。异步并行的优势在批量任务中更明显。

## 2. 异步执行器框架

### 2.1 异步任务定义

**文件**: [async_executor.py](e:\my_multi_agent\async_executor.py)

```python
from enum import Enum
from dataclasses import dataclass, field

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class AsyncTask:
    task_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 2.2 AsyncToolExecutor

核心类，管理线程池和任务生命周期：

```python
class AsyncToolExecutor:
    def __init__(self, max_workers: int = 4, use_threadpool: bool = True):
        self.max_workers = max_workers
        self.running_tasks: Dict[str, AsyncTask] = {}
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_execution_time": 0.0,
            "concurrent_max": 0,
        }

    def submit_tool_call(self, tool_func, *args, tool_name=None, timeout=30.0, **kwargs) -> str:
        """提交工具调用任务，返回task_id"""

    def wait_for_tasks(self, task_ids, timeout=None) -> Dict[str, Any]:
        """等待多个任务完成"""
```

## 3. 并行任务调度

### 3.1 ParallelTaskScheduler

支持批量提交任务组，并行执行，统一收集结果：

```python
class ParallelTaskScheduler:
    def submit_task_group(self, tasks: List[Dict], group_id=None) -> str:
        """提交一组并行任务"""

    def get_group_status(self, group_id: str) -> Dict[str, Any]:
        """获取任务组进度状态"""
```

### 3.2 使用示例

```python
from async_executor import run_tools_parallel

# 并行查询多个数据源
tool_calls = [
    {"func": weather_query.run, "args": ["北京"], "tool_name": "weather_query"},
    {"func": stock_query.run, "args": ["AAPL"], "tool_name": "stock_query"},
]

results = run_tools_parallel(tool_calls, timeout=10.0)
```

## 4. 流式响应处理

### 4.1 StreamingResponseHandler

```python
class StreamingResponseHandler:
    def execute_with_streaming(self, task_defs, stream_id=None) -> str:
        """执行任务并支持流式响应"""

    def register_callback(self, event_type: str, callback: Callable):
        """注册回调函数"""
```

**流式事件类型**: progress, result, error, complete

## 5. 全局实例管理

```python
_async_executor = None
_parallel_scheduler = None
_streaming_handler = None

def get_async_executor(max_workers: int = 4) -> AsyncToolExecutor:
def get_parallel_scheduler() -> ParallelTaskScheduler:
def get_streaming_handler() -> StreamingResponseHandler:
```

单例模式 + 延迟初始化，与主Agent的LLM/RAG资源管理风格一致。

## 6. 与v1.0的集成差异

| 维度 | v1.0 (旧版) | v2.0 (当前) |
|------|-----------|-----------|
| 集成方式 | `execution_node_async` 嵌入工作流 | 不在主工作流中使用 |
| 触发条件 | `USE_ASYNC_EXECUTION` 全局开关 | 手动调用 |
| 节点定义 | 独立的异步执行节点（~270行） | 无异步节点 |
| 回退机制 | 异步失败 → `execution_node` 同步 | 天然同步，无需回退 |
| 适用场景 | 所有请求（即使单工具调用也走线程池） | 按需使用（批量任务） |

## 7. 性能优化建议

### 7.1 当前主流程优化
- LLM单例共享（避免重复初始化ChatTongyi）
- RAG检索器延迟加载（启动快）
- 对话历史长度限制（最多10轮）
- 向量缓存（减少重复嵌入计算）

### 7.2 如需并行优化
- 可将多个独立的工具调用（如同时查天气+查日期）通过 `run_tools_parallel` 并行化
- 注意：ReAct循环中LLM通常一次只调用一个工具，所以并行收益有限

## 8. 学习总结

### 关键要点
1. `async_executor.py` 是独立的工具库，不属于主Agent工作流
2. v2.0标准ReAct循环中工具调用是串行的（简单可靠）
3. 批量并行任务场景可手动使用 `run_tools_parallel`
4. 模块采用与主Agent一致的延迟初始化 + 单例模式

### 最佳实践
1. **不要为并行而并行**: 单工具调用的场景用线程池反而增加开销
2. **合理设置超时**: 根据工具类型设置不同的timeout
3. **监控线程池**: 关注并发数、成功率、平均耗时

---

**相关文件**:
- [async_executor.py](e:\my_multi_agent\async_executor.py) — 异步执行器完整实现
- [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) — 主工作流（同步ReAct）

**下一步学习**: 监控系统 →
