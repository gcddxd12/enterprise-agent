# 监控与评估系统

## 概述
企业级AI Agent系统需要完善的监控和评估能力。本监控系统集成了LangSmith跟踪、结构化日志、性能指标收集和智能报警，当前已适配v2.0的3节点工作流（preprocess → agent → postprocess）。

## 1. 系统架构

### 1.1 四大监控支柱

```
┌─────────────────────────────────────────────────────┐
│              监控系统 (MonitoringSystem)              │
├─────────────┬─────────────┬───────────┬─────────────┤
│ LangSmith   │ 结构化日志   │ 指标收集器 │ 报警管理器   │
│ 跟踪器      │ (Structured │ (Metrics  │ (Alert      │
│ (Tracer)    │ Logger)     │ Collector)│ Manager)    │
└─────────────┴─────────────┴───────────┴─────────────┘
```

### 1.2 配置驱动设计

**文件**: [monitoring_config.py](e:\my_multi_agent\monitoring_config.py)

```python
@dataclass
class MonitoringConfig:
    service_name: str = "enterprise-agent"
    environment: str = os.getenv("ENVIRONMENT", "development")
    enable_monitoring: bool = True
    enable_health_checks: bool = True
    enable_dashboard: bool = True
    dashboard_port: int = 8080
```

根据环境自动调整：生产环境低采样率+完整报警，开发环境全采样+详细调试。

## 2. LangSmith集成

**文件**: [monitoring_system.py](e:\my_multi_agent\monitoring_system.py)

```python
class LangSmithTracer:
    def trace_workflow(self, workflow_name, inputs, metadata=None) -> Optional[str]:
        """跟踪工作流执行"""
    def trace_node(self, run_id, node_name, inputs, outputs, metadata=None, error=None):
        """跟踪单个节点执行"""
```

**跟踪功能**: 工作流级跟踪 + 节点级跟踪 + 错误跟踪 + LangSmith Web UI可视化。

## 3. 结构化日志系统

**文件**: [monitoring_system.py](e:\my_multi_agent\monitoring_system.py)

```python
class StructuredLogger:
    def log(self, level: str, event: str, **kwargs):
        """记录结构化日志（JSON格式）"""
```

使用 `structlog` 库，支持JSON和Console两种格式，不可用时回退到基础 `print`。

**标准事件类型**:
- `workflow_started` / `workflow_completed`
- `node_executed` — 注意：v2.0节点名称为 preprocess_node / agent_node / postprocess_node
- `rag_retrieval` / `tool_called` / `error_occurred`

## 4. 指标收集

**文件**: [monitoring_system.py](e:\my_multi_agent\monitoring_system.py)

```python
class MetricsCollector:
    # 关键指标
    - requests_total (Counter): 总请求数
    - requests_duration (Histogram): 请求耗时分布
    - errors_total (Counter): 错误总数
    - tool_calls_total (Counter, labeled by tool_name): 各工具调用次数
    - rag_retrieval_time (Histogram): RAG检索时间分布
    - concurrent_requests (Gauge): 当前并发请求数
```

## 5. 智能报警

```python
class AlertManager:
    def check_thresholds(self) -> List[Dict]:
        """检查阈值并生成报警（错误率、P95响应时间等）"""
```

报警冷却机制：同类型报警至少间隔5分钟，防止报警风暴。

## 6. 与v2.0工作流的集成

### 6.1 集成点

监控系统在当前3节点工作流中的集成位置：

| 节点 | 监控操作 | 文件位置 |
|------|---------|---------|
| preprocess_node | `track_workflow_start` + `track_node_execution` | 第630-646行 |
| agent_node | RAG检索监控（`track_rag_retrieval`） | 第340-406行 |
| postprocess_node | `track_node_execution` + `track_workflow_end` | 第678-698行 |

### 6.2 节点监控示例（preprocess_node）

```python
# langgraph_agent_with_memory.py 第630-646行
def preprocess_node(state: AgentState) -> AgentState:
    if MONITORING_AVAILABLE and monitoring_system:
        tracking_info = monitoring_system.track_workflow_start(
            workflow_name="enterprise_agent_workflow",
            inputs={"user_query": state['user_query']}
        )
    # ... 节点逻辑
```

### 6.3 节点监控示例（postprocess_node）

```python
# langgraph_agent_with_memory.py 第678-698行
def postprocess_node(state: AgentState) -> AgentState:
    if MONITORING_AVAILABLE and monitoring_system:
        monitoring_system.track_node_execution(
            state['tracking_info'],
            node_name="postprocess_node",
            inputs={"final_answer": final_answer[:200]},
            outputs={"adapted_answer": adapted_answer[:200]},
            duration=duration, success=True
        )
        monitoring_system.track_workflow_end(
            state['tracking_info'],
            outputs={"final_answer": adapted_answer[:200]},
            success=True
        )
```

### 6.4 v2.0节点名称一览

与v1.0相比，需要更新的节点名称：

| v1.0 (旧版，已废弃) | v2.0 (当前) |
|--------------------|-----------|
| preprocess_node | preprocess_node（不变） |
| planning_node | **已移除**（逻辑并入agent_node） |
| execution_node / execution_node_async | **已移除**（逻辑并入agent_node） |
| validation_node | **已移除**（逻辑并入agent_node） |
| human_escalation_node | **已移除** |
| postprocess_node | postprocess_node（不变） |
| — | **agent_node**（核心，新增） |

## 7. 配置管理

**文件**: [monitoring_config.py](e:\my_multi_agent\monitoring_config.py)

```python
class ConfigManager:
    def load_from_env(self):
        """从环境变量加载配置"""
    def validate(self) -> bool:
        """验证配置有效性"""
```

## 8. 学习总结

### 关键监控能力
1. **全链路跟踪**: LangSmith集成提供可视化执行流程
2. **结构化日志**: JSON格式便于ELK栈分析
3. **实时指标**: Prometheus指标（Counter/Histogram/Gauge）
4. **智能报警**: 阈值检测 + 冷却机制 + 多通道通知

### v2.0注意事项
- 监控系统代码基本不变，但需要关注节点名称的变化
- `track_node_execution` 中传入的 `node_name` 参数从旧版节点名改为新版
- `agent_node` 是监控重点（包含ReAct循环和工具调用）

### 最佳实践
1. **环境适配**: 开发/预发布/生产自动调整配置
2. **采样策略**: 生产环境使用采样减少开销
3. **报警冷却**: 防止报警风暴
4. **错误隔离**: 监控代码异常不影响主流程

---

**相关文件**:
- [monitoring_system.py](e:\my_multi_agent\monitoring_system.py) — 监控系统完整实现
- [monitoring_config.py](e:\my_multi_agent\monitoring_config.py) — 配置管理系统
- [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) — 工作流中的监控集成点

**下一步学习**: 多模态支持 →
