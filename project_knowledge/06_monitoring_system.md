# 监控与评估系统

## 概述
企业级AI Agent系统需要完善的监控和评估能力，以确保可靠性、性能和可观测性。本监控系统集成了LangSmith跟踪、结构化日志、性能指标收集和智能报警，提供全面的生产就绪监控解决方案。

## 1. 系统架构与设计

### 1.1 四大监控支柱

**监控系统架构**:
```
┌─────────────────────────────────────────────────────┐
│              监控系统 (MonitoringSystem)              │
├─────────────┬─────────────┬───────────┬─────────────┤
│ LangSmith   │ 结构化日志   │ 指标收集器 │ 报警管理器   │
│ 跟踪器      │ (Structured │ (Metrics  │ (Alert      │
│ (Tracer)    │ Logger)     │ Collector)│ Manager)    │
└─────────────┴─────────────┴───────────┴─────────────┘
       │             │            │            │
       ▼             ▼            ▼            ▼
┌─────────────┬─────────────┬───────────┬─────────────┐
│ 工作流跟踪   │ JSON日志     │ Prometheus│ Webhook/    │
│ 节点执行    │ 控制台输出   │ 指标       │ 邮件报警    │
└─────────────┴─────────────┴───────────┴─────────────┘
```

### 1.2 配置驱动设计

**代码示例**:
```python
# monitoring_config.py 第154-291行：完整监控配置
from dataclasses import dataclass, field
from enum import Enum

class LogFormat(Enum):
    """日志格式枚举"""
    JSON = "json"
    CONSOLE = "console"
    PLAIN = "plain"

@dataclass
class MonitoringConfig:
    """完整监控配置"""
    # 组件配置
    langsmith: LangSmithConfig = field(default_factory=LangSmithConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    
    # 系统配置
    service_name: str = "enterprise-agent"
    environment: str = os.getenv("ENVIRONMENT", "development")
    version: str = "1.0.0"
    instance_id: str = os.getenv("INSTANCE_ID", "default")
    
    # 功能开关
    enable_monitoring: bool = True
    enable_health_checks: bool = True
    enable_dashboard: bool = True
    dashboard_port: int = 8080
    
    def __post_init__(self):
        # 根据环境调整配置
        if self.environment == "production":
            self._apply_production_settings()
        elif self.environment == "staging":
            self._apply_staging_settings()
        elif self.environment == "development":
            self._apply_development_settings()
    
    def _apply_production_settings(self):
        """应用生产环境设置"""
        self.logging.level = "INFO"
        self.logging.format = LogFormat.JSON
        self.alerts.enabled = True
        self.tracing.sampling_rate = 0.1  # 生产环境10%采样
        self.metrics.enabled = True
```

**环境感知配置**:
1. **生产环境**: 高安全性，低采样率，完整报警
2. **预发布环境**: 中等采样率，详细日志，报警启用
3. **开发环境**: 全采样，详细调试，报警禁用

## 2. LangSmith集成

### 2.1 LangSmith跟踪器

**代码示例**:
```python
# monitoring_system.py 第214-307行：LangSmithTracer类
class LangSmithTracer:
    """LangSmith工作流跟踪器"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.client = None
        self.enabled = False
        
        # 获取LangSmith配置
        langsmith_config = get_langsmith_config(config)
        api_key = langsmith_config['api_key']
        project = langsmith_config['project']
        endpoint = langsmith_config['endpoint']
        enabled = langsmith_config.get('enabled', True)
        
        if LANGSMITH_AVAILABLE and api_key and enabled:
            try:
                os.environ["LANGSMITH_API_KEY"] = api_key
                os.environ["LANGSMITH_PROJECT"] = project
                os.environ["LANGSMITH_ENDPOINT"] = endpoint
                
                self.client = Client()
                self.enabled = True
                print(f"[INFO] LangSmith tracer initialized for project: {project}")
            except Exception as e:
                print(f"[ERROR] Failed to initialize LangSmith tracer: {e}")
                self.enabled = False
    
    def trace_workflow(self, workflow_name: str, inputs: Dict[str, Any], metadata: Dict[str, Any] = None) -> Optional[str]:
        """跟踪工作流执行"""
        if not self.enabled or not self.client:
            return None
        
        try:
            run_tree = RunTree(
                name=workflow_name,
                inputs=inputs,
                metadata=metadata or {},
                project_name=self.config.langsmith.project,
            )
            
            run_tree.end(outputs={"status": "started"})
            self.client.create_run(run_tree)
            
            return str(run_tree.id)
        except Exception as e:
            print(f"[ERROR] Failed to trace workflow: {e}")
            return None
    
    def trace_node(self,
                   run_id: str,
                   node_name: str,
                   inputs: Dict[str, Any],
                   outputs: Dict[str, Any],
                   metadata: Dict[str, Any] = None,
                   error: Optional[str] = None):
        """跟踪单个节点执行"""
        if not self.enabled or not self.client:
            return
        
        try:
            run_tree = RunTree(
                id=run_id,
                name=node_name,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata or {},
                error=error,
            )
            
            self.client.create_run(run_tree)
        except Exception as e:
            print(f"[ERROR] Failed to trace node: {e}")
```

**跟踪功能**:
1. **工作流级跟踪**: 记录完整工作流执行
2. **节点级跟踪**: 记录每个节点的输入输出
3. **错误跟踪**: 捕获和记录执行错误
4. **可视化**: 提供LangSmith Web界面查看执行详情

### 2.2 跟踪URL生成

**代码示例**:
```python
def get_trace_url(self, run_id: str) -> Optional[str]:
    """获取跟踪URL"""
    if not self.enabled or not self.client:
        return None
    
    try:
        # LangSmith URL格式
        langsmith_config = get_langsmith_config(self.config)
        base_url = langsmith_config['endpoint'].replace("api.", "")
        project = langsmith_config['project']
        return f"{base_url}/project/{project}/r/{run_id}"
    except:
        return None
```

## 3. 结构化日志系统

### 3.1 结构化日志记录器

**代码示例**:
```python
# monitoring_system.py 第311-410行：StructuredLogger类
class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = None
        
        # 获取日志配置
        logging_config = get_logging_config(config)
        self.log_level = logging_config['level']
        self.log_format = logging_config['format']
        self.log_file = logging_config['file_path']
        self.enable_structured_logging = logging_config['enable_structured_logging']
        
        if STRUCTLOG_AVAILABLE and self.enable_structured_logging:
            try:
                # 配置structlog
                structlog.configure(
                    processors=[
                        structlog.stdlib.filter_by_level,
                        structlog.stdlib.add_logger_name,
                        structlog.stdlib.add_log_level,
                        structlog.stdlib.PositionalArgumentsFormatter(),
                        structlog.processors.TimeStamper(fmt="iso"),
                        structlog.processors.StackInfoRenderer(),
                        structlog.processors.format_exc_info,
                        structlog.processors.UnicodeDecoder(),
                        structlog.processors.JSONRenderer() if log_format == "json" else structlog.dev.ConsoleRenderer(),
                    ],
                    context_class=dict,
                    logger_factory=structlog.stdlib.LoggerFactory(),
                    wrapper_class=structlog.stdlib.BoundLogger,
                    cache_logger_on_first_use=True,
                )
                
                self.logger = structlog.get_logger("enterprise_agent")
                
                # 配置日志文件
                if self.log_file:
                    os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                    import logging
                    file_handler = logging.FileHandler(self.log_file)
                    file_handler.setLevel(getattr(logging, self.log_level))
                    
                    root_logger = logging.getLogger()
                    root_logger.addHandler(file_handler)
                
                print(f"[INFO] Structured logger initialized (format: {log_format})")
            except Exception as e:
                print(f"[ERROR] Failed to initialize structured logger: {e}")
                self.logger = None
        else:
            print("[INFO] Using basic logging (structlog not available or disabled)")
    
    def log(self,
            level: str,
            event: str,
            **kwargs):
        """记录结构化日志"""
        try:
            if self.logger:
                log_method = getattr(self.logger, level.lower(), self.logger.info)
                log_method(event, **kwargs)
            else:
                # 回退到基础日志
                timestamp = datetime.now().isoformat()
                log_entry = {
                    "timestamp": timestamp,
                    "level": level,
                    "event": event,
                    **kwargs
                }
                
                if self.log_format == "json":
                    print(json.dumps(log_entry, ensure_ascii=False))
                else:
                    print(f"[{level}] {event}: {kwargs}")
        except Exception as e:
            print(f"[ERROR] Failed to log: {e}")
```

**日志特性**:
1. **JSON格式**: 便于ELK栈处理和分析
2. **结构化数据**: 标准化的字段和事件类型
3. **多输出**: 同时支持控制台和文件输出
4. **回退机制**: structlog不可用时使用基础日志

### 3.2 日志事件类型

**标准事件类型**:
- `workflow_started`: 工作流开始执行
- `workflow_completed`: 工作流完成
- `node_executed`: 节点执行完成
- `rag_retrieval`: RAG检索操作
- `tool_called`: 工具调用
- `error_occurred`: 错误发生

## 4. 指标收集与监控

### 4.1 指标收集器

**代码示例**:
```python
# monitoring_system.py 第413-586行：MetricsCollector类
class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # 获取指标配置
        metrics_config = get_metrics_config(config)
        self.enabled = metrics_config['enabled'] and PROMETHEUS_AVAILABLE
        self.collect_interval_seconds = metrics_config['collect_interval_seconds']
        
        # 指标定义
        self.metrics = {}
        self.historical_data = defaultdict(lambda: deque(maxlen=1000))
        
        if self.enabled:
            try:
                # 创建Prometheus指标
                self.metrics = {
                    "requests_total": Counter("agent_requests_total", "Total number of requests"),
                    "requests_duration": Histogram("agent_requests_duration_seconds", "Request duration in seconds", buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)),
                    "errors_total": Counter("agent_errors_total", "Total number of errors"),
                    "tool_calls_total": Counter("agent_tool_calls_total", "Total number of tool calls", ["tool_name"]),
                    "tool_errors_total": Counter("agent_tool_errors_total", "Total number of tool errors", ["tool_name"]),
                    "response_time": Histogram("agent_response_time_seconds", "Response time distribution", buckets=(0.1, 0.5, 1.0, 2.0, 5.0)),
                    "concurrent_requests": Gauge("agent_concurrent_requests", "Number of concurrent requests"),
                    "memory_usage": Gauge("agent_memory_usage_bytes", "Memory usage in bytes"),
                    "rag_retrieval_time": Histogram("agent_rag_retrieval_time_seconds", "RAG retrieval time distribution"),
                }
                
                print("[INFO] Prometheus metrics collector initialized")
                
                # 启动指标收集线程
                if self.collect_interval_seconds > 0:
                    self._start_collection_thread()
                
            except Exception as e:
                print(f"[ERROR] Failed to initialize metrics collector: {e}")
                self.enabled = False
    
    def record_request(self, duration: float, success: bool = True):
        """记录请求指标"""
        if not self.enabled:
            return
        
        try:
            self.metrics["requests_total"].inc()
            self.metrics["requests_duration"].observe(duration)
            self.metrics["response_time"].observe(duration)
            
            if not success:
                self.metrics["errors_total"].inc()
            
            # 记录历史数据
            self.historical_data["request_durations"].append(duration)
            self.historical_data["request_success"].append(1 if success else 0)
        except Exception as e:
            print(f"[ERROR] Failed to record request metrics: {e}")
    
    def record_tool_call(self, tool_name: str, duration: float, success: bool = True):
        """记录工具调用指标"""
        if not self.enabled:
            return
        
        try:
            self.metrics["tool_calls_total"].labels(tool_name=tool_name).inc()
            
            if not success:
                self.metrics["tool_errors_total"].labels(tool_name=tool_name).inc()
            
            # 记录历史数据
            key = f"tool_{tool_name}_durations"
            self.historical_data[key].append(duration)
        except Exception as e:
            print(f"[ERROR] Failed to record tool call metrics: {e}")
```

**关键指标类型**:
1. **计数器 (Counter)**: 请求数、错误数、工具调用数
2. **直方图 (Histogram)**: 响应时间分布、RAG检索时间
3. **仪表盘 (Gauge)**: 并发请求数、内存使用量
4. **标签支持**: 支持按工具名称等维度细分

### 4.2 性能统计计算

**代码示例**:
```python
def get_statistics(self) -> Dict[str, Any]:
    """获取统计信息"""
    stats = {}
    
    # 计算请求统计
    durations = list(self.historical_data["request_durations"])
    successes = list(self.historical_data["request_success"])
    
    if durations:
        stats["request_count"] = len(durations)
        stats["avg_response_time"] = sum(durations) / len(durations)
        stats["min_response_time"] = min(durations)
        stats["max_response_time"] = max(durations)
        
        # 计算P95
        sorted_durations = sorted(durations)
        p95_index = int(len(sorted_durations) * 0.95)
        stats["p95_response_time"] = sorted_durations[p95_index] if p95_index < len(sorted_durations) else sorted_durations[-1]
    
    if successes:
        success_count = sum(successes)
        total_count = len(successes)
        stats["success_rate"] = success_count / total_count if total_count > 0 else 1.0
        stats["error_rate"] = 1 - stats["success_rate"]
    
    # 工具统计
    tool_stats = {}
    for key in self.historical_data:
        if key.startswith("tool_"):
            tool_name = key[5:].replace("_durations", "")
            durations = list(self.historical_data[key])
            
            if durations:
                tool_stats[tool_name] = {
                    "call_count": len(durations),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                }
    
    stats["tools"] = tool_stats
    
    return stats
```

**统计维度**:
1. **响应时间**: 平均值、最小值、最大值、P95
2. **成功率**: 基于历史数据的成功/失败率计算
3. **工具性能**: 每个工具的平均执行时间和调用次数
4. **趋势分析**: 基于时间窗口的指标变化

## 5. 智能报警系统

### 5.1 报警管理器

**代码示例**:
```python
# monitoring_system.py 第590-726行：AlertManager类
class AlertManager:
    """报警管理器"""
    
    def __init__(self, config: MonitoringConfig, metrics_collector: MetricsCollector):
        self.config = config
        self.metrics_collector = metrics_collector
        
        # 获取报警配置
        alerts_config = get_alerts_config(config)
        self.enabled = alerts_config['enabled']
        self.webhook_url = alerts_config['webhook_url']
        self.email = alerts_config['email']
        self.thresholds = alerts_config['thresholds']
        self.last_alert_time = {}
    
    def check_thresholds(self) -> List[Dict[str, Any]]:
        """检查阈值并生成报警"""
        if not self.enabled:
            return []
        
        alerts = []
        stats = self.metrics_collector.get_statistics()
        
        # 检查错误率阈值
        error_rate = stats.get("error_rate", 0)
        error_rate_threshold = self.thresholds.get("error_rate", 0.05)
        if error_rate > error_rate_threshold:
            alert = self._create_alert(
                "error_rate_exceeded",
                f"错误率超过阈值: {error_rate:.2%} > {error_rate_threshold:.2%}",
                {"error_rate": error_rate, "threshold": error_rate_threshold},
                severity="high"
            )
            alerts.append(alert)
        
        # 检查响应时间阈值
        p95_response_time = stats.get("p95_response_time", 0)
        response_time_threshold = self.thresholds.get("response_time_p95_ms", self.thresholds.get("response_time_p95", 3000))
        if p95_response_time * 1000 > response_time_threshold:
            alert = self._create_alert(
                "response_time_exceeded",
                f"P95响应时间超过阈值: {p95_response_time*1000:.0f}ms > {response_time_threshold}ms",
                {"p95_response_time_ms": p95_response_time * 1000, "threshold": response_time_threshold},
                severity="medium"
            )
            alerts.append(alert)
        
        # 发送报警
        for alert in alerts:
            self._send_alert(alert)
        
        return alerts
    
    def _send_alert(self, alert: Dict[str, Any]):
        """发送报警"""
        # 检查是否应该发送（防止重复报警）
        alert_key = f"{alert['alert_type']}_{alert['severity']}"
        current_time = time.time()
        
        # 相同类型的报警至少间隔5分钟
        if alert_key in self.last_alert_time and current_time - self.last_alert_time[alert_key] < 300:
            return
        
        self.last_alert_time[alert_key] = current_time
        
        try:
            # 记录日志
            print(f"[ALERT] {alert['message']}")
            
            # 发送到webhook
            if self.webhook_url:
                self._send_webhook_alert(alert)
            
            # 发送邮件
            if self.email:
                self._send_email_alert(alert)
                
        except Exception as e:
            print(f"[ERROR] Failed to send alert: {e}")
```

**报警类型**:
1. **错误率报警**: 系统错误率超过阈值
2. **响应时间报警**: P95响应时间超过阈值
3. **工具失败报警**: 特定工具失败率过高
4. **资源报警**: 内存、CPU使用率过高

### 5.2 报警冷却机制

**代码示例**:
```python
def _send_alert(self, alert: Dict[str, Any]):
    """发送报警（带冷却机制）"""
    # 检查是否应该发送（防止重复报警）
    alert_key = f"{alert['alert_type']}_{alert['severity']}"
    current_time = time.time()
    
    # 相同类型的报警至少间隔5分钟
    if alert_key in self.last_alert_time and current_time - self.last_alert_time[alert_key] < 300:
        return
    
    self.last_alert_time[alert_key] = current_time
    # ... 发送报警逻辑
```

**冷却策略**:
1. **错误率报警**: 5分钟冷却
2. **响应时间报警**: 5分钟冷却
3. **工具失败报警**: 10分钟冷却
4. **资源报警**: 15分钟冷却

## 6. 性能监控器

### 6.1 性能数据采样

**代码示例**:
```python
# monitoring_system.py 第858-933行：PerformanceMonitor类
class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # 获取指标配置
        metrics_config = get_metrics_config(config)
        self.sample_rate = metrics_config['performance_sample_rate']
        self.performance_data = defaultdict(list)
        
        print(f"[INFO] Performance monitor initialized (sample rate: {self.sample_rate})")
    
    def record_performance(self,
                           operation: str,
                           duration: float,
                           metadata: Dict[str, Any] = None):
        """记录性能数据"""
        import random
        
        # 采样
        if random.random() > self.sample_rate:
            return
        
        data_point = {
            "timestamp": time.time(),
            "duration": duration,
            "operation": operation,
            "metadata": metadata or {}
        }
        
        self.performance_data[operation].append(data_point)
        
        # 保持数据量可控
        if len(self.performance_data[operation]) > 1000:
            self.performance_data[operation] = self.performance_data[operation][-1000:]
    
    def get_performance_report(self, operation: str = None) -> Dict[str, Any]:
        """获取性能报告"""
        if operation:
            data = self.performance_data.get(operation, [])
        else:
            data = []
            for op_data in self.performance_data.values():
                data.extend(op_data)
        
        if not data:
            return {"count": 0, "message": "No performance data available"}
        
        durations = [d["duration"] for d in data]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        # 计算百分位数
        sorted_durations = sorted(durations)
        percentiles = {}
        for p in [50, 75, 90, 95, 99]:
            index = int(len(sorted_durations) * p / 100)
            if index >= len(sorted_durations):
                index = len(sorted_durations) - 1
            percentiles[f"p{p}"] = sorted_durations[index]
        
        return {
            "count": len(data),
            "avg_duration": avg_duration,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "percentiles": percentiles,
            "sample_rate": self.sample_rate,
            "time_range": {
                "start": min(d["timestamp"] for d in data),
                "end": max(d["timestamp"] for d in data)
            }
        }
```

**性能监控特性**:
1. **智能采样**: 可配置采样率，平衡精度和性能
2. **百分位数计算**: P50、P75、P90、P95、P99
3. **时间窗口分析**: 基于时间范围的性能趋势
4. **操作细分**: 按操作类型分类统计

## 7. 监控系统集成

### 7.1 监控系统主类

**代码示例**:
```python
# monitoring_system.py 第728-855行：MonitoringSystem类
class MonitoringSystem:
    """监控系统主类"""
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        
        # 初始化组件
        self.tracer = LangSmithTracer(self.config)
        self.logger = StructuredLogger(self.config)
        self.metrics = MetricsCollector(self.config)
        self.alerts = AlertManager(self.config, self.metrics)
        
        # 性能监控
        self.performance_monitor = None
        metrics_config = get_metrics_config(self.config)
        if metrics_config['enable_performance_monitoring']:
            self.performance_monitor = PerformanceMonitor(self.config)
        
        print("[INFO] Monitoring system initialized")
    
    def track_workflow_start(self,
                             workflow_name: str,
                             inputs: Dict[str, Any],
                             metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """跟踪工作流开始"""
        trace_id = self.tracer.trace_workflow(workflow_name, inputs, metadata)
        
        self.logger.info(
            "workflow_started",
            workflow_name=workflow_name,
            inputs=inputs,
            trace_id=trace_id
        )
        
        self.metrics.increment_concurrent_requests(1)
        
        return {
            "trace_id": trace_id,
            "start_time": time.time(),
            "workflow_name": workflow_name
        }
    
    def track_workflow_end(self,
                           tracking_info: Dict[str, Any],
                           outputs: Dict[str, Any],
                           success: bool = True,
                           error: Optional[str] = None):
        """跟踪工作流结束"""
        duration = time.time() - tracking_info.get("start_time", time.time())
        
        self.metrics.record_request(duration, success)
        self.metrics.increment_concurrent_requests(-1)
        
        self.logger.info(
            "workflow_completed",
            workflow_name=tracking_info.get("workflow_name"),
            duration=duration,
            success=success,
            error=error,
            trace_id=tracking_info.get("trace_id")
        )
        
        # 检查阈值并发送报警
        self.alerts.check_thresholds()
    
    def track_node_execution(self,
                             tracking_info: Dict[str, Any],
                             node_name: str,
                             inputs: Dict[str, Any],
                             outputs: Dict[str, Any],
                             duration: float,
                             success: bool = True,
                             error: Optional[str] = None):
        """跟踪节点执行"""
        self.tracer.trace_node(
            tracking_info.get("trace_id", ""),
            node_name,
            inputs,
            outputs,
            {"duration": duration, "success": success},
            error
        )
        
        self.logger.info(
            "node_executed",
            node_name=node_name,
            duration=duration,
            success=success,
            error=error,
            workflow_name=tracking_info.get("workflow_name")
        )
        
        # 如果是工具节点，记录工具指标
        if node_name.startswith("tool_"):
            tool_name = node_name[5:]
            self.metrics.record_tool_call(tool_name, duration, success)
```

### 7.2 监控装饰器

**代码示例**:
```python
# monitoring_system.py 第949-1040行：监控装饰器
def monitor_workflow(workflow_name: str = "default_workflow"):
    """监控工作流执行的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitoring = get_monitoring_system()
            
            # 提取输入参数
            inputs = {}
            if args:
                inputs["args"] = str(args)
            if kwargs:
                inputs.update(kwargs)
            
            # 跟踪工作流开始
            tracking_info = monitoring.track_workflow_start(workflow_name, inputs)
            
            try:
                # 执行函数
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 跟踪工作流成功结束
                monitoring.track_workflow_end(
                    tracking_info,
                    {"result": str(result)[:500]},  # 限制输出大小
                    success=True
                )
                
                return result
                
            except Exception as e:
                # 跟踪工作流失败结束
                duration = time.time() - start_time
                monitoring.track_workflow_end(
                    tracking_info,
                    {"error": str(e)},
                    success=False,
                    error=str(e)
                )
                monitoring.log_event("ERROR", "workflow_failed", error=str(e), workflow_name=workflow_name)
                raise
        
        return wrapper
    
    return decorator

def monitor_node(node_name: str):
    """监控节点执行的装饰器"""
    def decorator(func):
        def wrapper(tracking_info, *args, **kwargs):
            monitoring = get_monitoring_system()
            
            # 提取输入
            inputs = {"args": str(args), "kwargs": kwargs}
            
            try:
                # 执行节点
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 跟踪节点执行
                monitoring.track_node_execution(
                    tracking_info,
                    node_name,
                    inputs,
                    {"result": str(result)[:500]},
                    duration,
                    success=True
                )
                
                return result
                
            except Exception as e:
                # 跟踪节点失败
                duration = time.time() - start_time
                monitoring.track_node_execution(
                    tracking_info,
                    node_name,
                    inputs,
                    {"error": str(e)},
                    duration,
                    success=False,
                    error=str(e)
                )
                raise
        
        return wrapper
    
    return decorator
```

## 8. 配置管理

### 8.1 配置管理器

**代码示例**:
```python
# monitoring_config.py 第300-441行：ConfigManager类
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig.create_default_config()
        self.config_file = None
        self.watcher = None
    
    def load_from_env(self):
        """从环境变量加载配置"""
        # LangSmith配置
        self.config.langsmith.api_key = os.getenv("LANGSMITH_API_KEY", self.config.langsmith.api_key)
        self.config.langsmith.project = os.getenv("LANGSMITH_PROJECT", self.config.langsmith.project)
        self.config.langsmith.endpoint = os.getenv("LANGSMITH_ENDPOINT", self.config.langsmith.endpoint)
        
        # 日志配置
        self.config.logging.level = os.getenv("LOG_LEVEL", self.config.logging.level)
        log_format = os.getenv("LOG_FORMAT")
        if log_format:
            try:
                self.config.logging.format = LogFormat(log_format.lower())
            except ValueError:
                print(f"[WARN] Invalid log format: {log_format}, using default")
        
        # 报警配置
        self.config.alerts.webhook_url = os.getenv("ALERT_WEBHOOK_URL", self.config.alerts.webhook_url)
        self.config.alerts.email = os.getenv("ALERT_EMAIL", self.config.alerts.email)
        self.config.alerts.slack_webhook = os.getenv("SLACK_WEBHOOK_URL", self.config.alerts.slack_webhook)
        
        # 系统配置
        self.config.environment = os.getenv("ENVIRONMENT", self.config.environment)
        self.config.instance_id = os.getenv("INSTANCE_ID", self.config.instance_id)
        
        print("[INFO] Configuration loaded from environment variables")
        return self
    
    def validate(self) -> bool:
        """验证配置有效性"""
        errors = []
        
        # 验证LangSmith配置
        if self.config.langsmith.enabled and not self.config.langsmith.api_key:
            errors.append("LangSmith API key is required when LangSmith is enabled")
        
        # 验证日志配置
        if self.config.logging.enable_file_logging and not self.config.logging.file_path:
            errors.append("Log file path is required when file logging is enabled")
        
        # 验证报警配置
        if self.config.alerts.enabled:
            if (not self.config.alerts.webhook_url and
                not self.config.alerts.email and
                not self.config.alerts.slack_webhook and
                not self.config.alerts.enable_console_alerts):
                errors.append("At least one alert destination must be configured")
        
        if errors:
            print("[ERROR] Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("[INFO] Configuration validation passed")
        return True
```

### 8.2 配置验证规则

**验证项目**:
1. **LangSmith配置**: API密钥验证、项目名称验证
2. **日志配置**: 文件路径验证、日志级别验证
3. **报警配置**: 至少一个报警目标
4. **阈值配置**: 阈值数值范围验证

## 9. 与LangGraph工作流集成

### 9.1 工作流节点监控集成

**代码示例**:
```python
# langgraph_agent_with_memory.py 中的监控集成示例
def planning_node(state: AgentState) -> AgentState:
    """规划节点（带监控）"""
    import time
    start_time = time.time()
    
    # 获取监控系统
    if MONITORING_AVAILABLE and monitoring_system and 'tracking_info' in state:
        try:
            # 记录节点开始
            monitoring_system.track_node_execution(
                state['tracking_info'],
                node_name="planning_node",
                inputs={"user_query": state['user_query']},
                outputs={},  # 初始为空
                duration=0.0,
                success=True
            )
        except Exception as e:
            print(f"[WARN] 规划节点监控跟踪失败: {e}")
    
    # ... 节点逻辑
    
    duration = time.time() - start_time
    
    # 监控节点完成
    if MONITORING_AVAILABLE and monitoring_system and 'tracking_info' in state:
        try:
            monitoring_system.track_node_execution(
                state['tracking_info'],
                node_name="planning_node",
                inputs={"user_query": state['user_query']},
                outputs={"plan": tasks},
                duration=duration,
                success=len(tasks) > 0
            )
        except Exception as e:
            print(f"[WARN] 规划节点监控跟踪失败: {e}")
    
    return {**state, "plan": tasks, "step": "execution"}
```

## 10. 学习总结

### 关键监控能力
1. **全链路跟踪**: LangSmith集成提供可视化执行流程
2. **结构化日志**: 机器可读的JSON日志，便于分析
3. **实时指标**: Prometheus指标，支持Grafana可视化
4. **智能报警**: 阈值检测和多种通知方式

### 最佳实践
1. **环境适配**: 根据环境自动调整监控配置
2. **采样策略**: 生产环境使用采样减少开销
3. **报警冷却**: 防止报警风暴，确保可操作性
4. **验证机制**: 配置验证确保系统正确性

### 部署建议
1. **开发环境**: 禁用LangSmith，全采样，详细日志
2. **测试环境**: 启用基本监控，中等采样率
3. **生产环境**: 完整监控，低采样率，严格报警

### 扩展方向
1. **自定义指标**: 添加业务特定指标
2. **分布式追踪**: 支持多服务调用链追踪
3. **AI性能分析**: 集成模型性能监控
4. **成本监控**: 跟踪API调用成本

---

**相关文件**:
- [monitoring_system.py](e:\my_multi_agent\monitoring_system.py) - 监控系统完整实现
- [monitoring_config.py](e:\my_multi_agent\monitoring_config.py) - 配置管理系统
- [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) - 工作流监控集成

**下一步学习**: 多模态支持 →