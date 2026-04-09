#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控和评估系统 - 企业智能客服Agent第四阶段优化

功能：
1. LangSmith集成：工作流跟踪、调试、性能分析
2. 结构化日志：JSON格式日志，便于ELK栈集成
3. 关键指标监控：响应时间、成功率、工具使用统计
4. 错误追踪和报警：异常检测和通知

注意：部分功能需要配置相应服务（如LangSmith API密钥）

作者：gcddxd12
版本：1.0.0
创建日期：2026-04-09
"""

import os
import sys
import json
import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic
from enum import Enum
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    print("[WARN] structlog not available, using basic logging")

try:
    from langsmith import Client, traceable, RunTree
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    print("[WARN] langsmith not available, tracing disabled")

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("[WARN] prometheus_client not available, metrics disabled")

# ========== 配置导入 ==========

# 导入配置类
try:
    from monitoring_config import (
        MonitoringConfig,
        LogFormat,
        LangSmithConfig,
        LoggingConfig,
        MetricsConfig,
        AlertsConfig,
        TracingConfig
    )
    CONFIG_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] monitoring_config module not available: {e}")
    CONFIG_MODULE_AVAILABLE = False

    # 定义简单配置类作为回退
    @dataclass
    class SimpleMonitoringConfig:
        """简单监控配置（回退）"""
        # LangSmith配置
        langsmith_api_key: Optional[str] = None
        langsmith_project: str = "enterprise-agent-monitoring"
        langsmith_endpoint: str = "https://api.smith.langchain.com"

        # 日志配置
        log_level: str = "INFO"
        log_format: str = "json"
        log_file: Optional[str] = "./logs/agent_monitoring.log"
        enable_structured_logging: bool = True

        # 指标配置
        enable_metrics: bool = True
        metrics_port: int = 9090
        collect_interval_seconds: int = 30

        # 报警配置
        enable_alerts: bool = True
        alert_webhook_url: Optional[str] = None
        alert_email: Optional[str] = None
        alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
            "error_rate": 0.05,
            "response_time_p95": 3000,
            "tool_failure_rate": 0.1,
        })

        # 性能监控
        enable_performance_monitoring: bool = True
        performance_sample_rate: float = 0.1

        def __post_init__(self):
            if not self.langsmith_api_key:
                self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
            if not self.alert_webhook_url:
                self.alert_webhook_url = os.getenv("ALERT_WEBHOOK_URL")
            if not self.alert_email:
                self.alert_email = os.getenv("ALERT_EMAIL")

    # 使用简单配置
    MonitoringConfig = SimpleMonitoringConfig

# ========== 配置访问辅助函数 ==========

def get_langsmith_config(config: MonitoringConfig) -> Dict[str, Any]:
    """获取LangSmith配置（兼容简单和嵌套结构）"""
    try:
        # 尝试嵌套结构
        if hasattr(config, 'langsmith') and hasattr(config.langsmith, 'api_key'):
            return {
                'api_key': config.langsmith.api_key,
                'project': config.langsmith.project,
                'endpoint': config.langsmith.endpoint,
                'enabled': config.langsmith.enabled
            }
    except AttributeError:
        pass

    # 使用简单结构
    return {
        'api_key': getattr(config, 'langsmith_api_key', None),
        'project': getattr(config, 'langsmith_project', 'enterprise-agent-monitoring'),
        'endpoint': getattr(config, 'langsmith_endpoint', 'https://api.smith.langchain.com'),
        'enabled': getattr(config, 'langsmith_enabled', True)
    }

def get_logging_config(config: MonitoringConfig) -> Dict[str, Any]:
    """获取日志配置（兼容简单和嵌套结构）"""
    try:
        # 尝试嵌套结构
        if hasattr(config, 'logging') and hasattr(config.logging, 'level'):
            return {
                'level': config.logging.level,
                'format': config.logging.format.value if hasattr(config.logging.format, 'value') else config.logging.format,
                'file_path': config.logging.file_path,
                'enable_structured_logging': config.logging.enable_structured_logging
            }
    except AttributeError:
        pass

    # 使用简单结构
    return {
        'level': getattr(config, 'log_level', 'INFO'),
        'format': getattr(config, 'log_format', 'json'),
        'file_path': getattr(config, 'log_file', './logs/agent_monitoring.log'),
        'enable_structured_logging': getattr(config, 'enable_structured_logging', True)
    }

def get_alerts_config(config: MonitoringConfig) -> Dict[str, Any]:
    """获取报警配置（兼容简单和嵌套结构）"""
    try:
        # 尝试嵌套结构
        if hasattr(config, 'alerts') and hasattr(config.alerts, 'enabled'):
            return {
                'enabled': config.alerts.enabled,
                'webhook_url': config.alerts.webhook_url,
                'email': config.alerts.email,
                'thresholds': config.alerts.thresholds
            }
    except AttributeError:
        pass

    # 使用简单结构
    return {
        'enabled': getattr(config, 'enable_alerts', True),
        'webhook_url': getattr(config, 'alert_webhook_url', None),
        'email': getattr(config, 'alert_email', None),
        'thresholds': getattr(config, 'alert_thresholds', {
            "error_rate": 0.05,
            "response_time_p95": 3000,
            "tool_failure_rate": 0.1,
        })
    }

def get_metrics_config(config: MonitoringConfig) -> Dict[str, Any]:
    """获取指标配置（兼容简单和嵌套结构）"""
    try:
        # 尝试嵌套结构
        if hasattr(config, 'metrics') and hasattr(config.metrics, 'enabled'):
            return {
                'enabled': config.metrics.enabled,
                'port': config.metrics.port,
                'collect_interval_seconds': config.metrics.collect_interval_seconds,
                'enable_performance_monitoring': getattr(config.metrics, 'enable_performance_monitoring', True),
                'performance_sample_rate': getattr(config.metrics, 'performance_sample_rate', 0.1)
            }
    except AttributeError:
        pass

    # 使用简单结构
    return {
        'enabled': getattr(config, 'enable_metrics', True),
        'port': getattr(config, 'metrics_port', 9090),
        'collect_interval_seconds': getattr(config, 'collect_interval_seconds', 30),
        'enable_performance_monitoring': getattr(config, 'enable_performance_monitoring', True),
        'performance_sample_rate': getattr(config, 'performance_sample_rate', 0.1)
    }

# ========== LangSmith跟踪器 ==========

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
        else:
            print(f"[INFO] LangSmith tracer disabled (missing API key: {not api_key}, enabled: {enabled}, package: {LANGSMITH_AVAILABLE})")

    def trace_workflow(self, workflow_name: str, inputs: Dict[str, Any], metadata: Dict[str, Any] = None) -> Optional[str]:
        """跟踪工作流执行"""
        if not self.enabled or not self.client:
            return None

        try:
            # 获取项目名称
            langsmith_config = get_langsmith_config(self.config)
            project_name = langsmith_config['project']

            run_tree = RunTree(
                name=workflow_name,
                inputs=inputs,
                metadata=metadata or {},
                project_name=project_name,
            )

            # 这里可以添加更多跟踪信息
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

# ========== 结构化日志记录器 ==========

class StructuredLogger:
    """结构化日志记录器"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = None

        # 获取并存储日志配置
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

    def info(self, event: str, **kwargs):
        """记录INFO级别日志"""
        self.log("INFO", event, **kwargs)

    def warning(self, event: str, **kwargs):
        """记录WARNING级别日志"""
        self.log("WARNING", event, **kwargs)

    def error(self, event: str, **kwargs):
        """记录ERROR级别日志"""
        self.log("ERROR", event, **kwargs)

    def debug(self, event: str, **kwargs):
        """记录DEBUG级别日志"""
        self.log("DEBUG", event, **kwargs)

    def critical(self, event: str, **kwargs):
        """记录CRITICAL级别日志"""
        self.log("CRITICAL", event, **kwargs)

# ========== 指标收集器 ==========

class MetricsCollector:
    """指标收集器"""

    def __init__(self, config: MonitoringConfig):
        self.config = config

        # 获取指标配置
        metrics_config = get_metrics_config(config)
        self.enabled = metrics_config['enabled'] and PROMETHEUS_AVAILABLE
        self.collect_interval_seconds = metrics_config['collect_interval_seconds']
        self.enable_performance_monitoring = metrics_config['enable_performance_monitoring']
        self.performance_sample_rate = metrics_config['performance_sample_rate']

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
        else:
            print("[INFO] Metrics collection disabled")

    def _start_collection_thread(self):
        """启动指标收集线程"""
        def collect_metrics():
            import psutil
            import threading as th

            process = psutil.Process()

            while True:
                try:
                    # 收集内存使用
                    memory_info = process.memory_info()
                    self.metrics["memory_usage"].set(memory_info.rss)

                    # 收集线程数
                    thread_count = th.active_count()
                    # 可以添加更多系统指标

                    time.sleep(self.collect_interval_seconds)
                except Exception as e:
                    print(f"[ERROR] Failed to collect metrics: {e}")
                    time.sleep(60)  # 出错后等待更长时间

        thread = threading.Thread(target=collect_metrics, daemon=True)
        thread.start()
        print(f"[INFO] Metrics collection thread started (interval: {self.collect_interval_seconds}s)")

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

    def record_rag_retrieval(self, duration: float, num_results: int):
        """记录RAG检索指标"""
        if not self.enabled:
            return

        try:
            self.metrics["rag_retrieval_time"].observe(duration)

            # 记录历史数据
            self.historical_data["rag_durations"].append(duration)
            self.historical_data["rag_results_count"].append(num_results)
        except Exception as e:
            print(f"[ERROR] Failed to record RAG metrics: {e}")

    def increment_concurrent_requests(self, delta: int = 1):
        """增加/减少并发请求计数"""
        if not self.enabled:
            return

        try:
            self.metrics["concurrent_requests"].inc(delta)
        except Exception as e:
            print(f"[ERROR] Failed to update concurrent requests: {e}")

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

# ========== 报警管理器 ==========

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

        if self.enabled:
            print("[INFO] Alert manager initialized")
        else:
            print("[INFO] Alert manager disabled")

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

    def _create_alert(self,
                      alert_type: str,
                      message: str,
                      data: Dict[str, Any],
                      severity: str = "medium") -> Dict[str, Any]:
        """创建报警对象"""
        return {
            "alert_type": alert_type,
            "message": message,
            "data": data,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "service": "enterprise-agent"
        }

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

    def _send_webhook_alert(self, alert: Dict[str, Any]):
        """发送webhook报警"""
        try:
            import requests

            payload = {
                "text": f"🚨 Enterprise Agent Alert: {alert['message']}",
                "attachments": [{
                    "title": f"Alert: {alert['alert_type']}",
                    "text": json.dumps(alert, indent=2, ensure_ascii=False),
                    "color": "danger" if alert["severity"] == "high" else "warning",
                }]
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                print(f"[INFO] Alert sent to webhook successfully")
            else:
                print(f"[ERROR] Failed to send alert to webhook: {response.status_code}")

        except Exception as e:
            print(f"[ERROR] Webhook alert failed: {e}")

    def _send_email_alert(self, alert: Dict[str, Any]):
        """发送邮件报警"""
        # 这里实现邮件发送逻辑
        # 实际部署时需要配置SMTP服务器
        print(f"[INFO] Email alert would be sent to {self.email}")
        print(f"Subject: Enterprise Agent Alert - {alert['alert_type']}")
        print(f"Body: {alert['message']}")

# ========== 监控系统主类 ==========

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

    def track_rag_retrieval(self,
                            query: str,
                            duration: float,
                            num_results: int,
                            success: bool = True):
        """跟踪RAG检索"""
        self.metrics.record_rag_retrieval(duration, num_results)

        self.logger.info(
            "rag_retrieval",
            query=query,
            duration=duration,
            num_results=num_results,
            success=success
        )

    def log_event(self,
                  level: str,
                  event: str,
                  **kwargs):
        """记录事件日志"""
        self.logger.log(level, event, **kwargs)

    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return self.metrics.get_statistics()

    def get_trace_url(self, trace_id: str) -> Optional[str]:
        """获取跟踪URL"""
        return self.tracer.get_trace_url(trace_id)

# ========== 性能监控器 ==========

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

# ========== 单例实例 ==========

_monitoring_system_instance = None

def get_monitoring_system(config: MonitoringConfig = None) -> MonitoringSystem:
    """获取监控系统单例实例"""
    global _monitoring_system_instance

    if _monitoring_system_instance is None:
        _monitoring_system_instance = MonitoringSystem(config)

    return _monitoring_system_instance

# ========== 装饰器 ==========

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

# ========== 测试函数 ==========

def test_monitoring_system():
    """测试监控系统"""
    print("=== 测试监控系统 ===")

    try:
        # 创建配置
        config = MonitoringConfig(
            # 使用默认配置
            langsmith=LangSmithConfig(
                api_key=os.getenv("LANGSMITH_API_KEY"),
                enabled=True,
                trace_all_nodes=True,
                sample_rate=1.0
            ),
            logging=LoggingConfig(
                level="INFO",
                format=LogFormat.CONSOLE,
                enable_structured_logging=True,
                enable_file_logging=False,
                enable_console_logging=True
            ),
            metrics=MetricsConfig(
                enabled=True,
                enable_prometheus=False,
                enable_custom_metrics=True
            ),
            alerts=AlertsConfig(
                enabled=True,
                webhook_url=None,
                email=None,
                slack_webhook=None
            ),
            tracing=TracingConfig(
                enabled=True,
                enable_request_tracing=True,
                enable_response_tracing=True
            ),
            service_name="test-monitoring-system",
            environment="test",
            instance_id="test-instance"
        )

        # 创建监控系统
        monitoring = MonitoringSystem(config)

        # 测试日志
        monitoring.log_event("INFO", "test_log", message="This is a test log")
        monitoring.log_event("ERROR", "test_error", error="Test error", code=500)

        # 测试工作流跟踪
        tracking_info = monitoring.track_workflow_start(
            "test_workflow",
            {"query": "test query", "user_id": "test_user"}
        )

        # 模拟一些处理
        time.sleep(0.1)

        # 测试节点跟踪
        monitoring.track_node_execution(
            tracking_info,
            "test_node",
            {"input": "test"},
            {"output": "result"},
            0.05,
            success=True
        )

        # 测试工具调用跟踪
        monitoring.track_node_execution(
            tracking_info,
            "tool_knowledge_search",
            {"query": "test query"},
            {"results": ["result1", "result2"]},
            0.2,
            success=True
        )

        # 测试RAG检索跟踪
        monitoring.track_rag_retrieval("test query", 0.15, 3, success=True)

        # 工作流结束
        monitoring.track_workflow_end(
            tracking_info,
            {"final_answer": "Test completed"},
            success=True
        )

        # 获取统计信息
        stats = monitoring.get_statistics()
        print(f"统计信息: {json.dumps(stats, indent=2, ensure_ascii=False)}")

        # 测试报警
        alerts = monitoring.alerts.check_thresholds()
        print(f"生成的报警: {len(alerts)} 个")

        print("\n[SUCCESS] 监控系统测试完成")
        return True

    except Exception as e:
        print(f"[FAILED] 监控系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# ========== 主入口 ==========

if __name__ == "__main__":
    print("监控和评估系统模块")
    print("功能: LangSmith跟踪、结构化日志、指标收集、报警管理")

    # 运行测试
    test_monitoring_system()