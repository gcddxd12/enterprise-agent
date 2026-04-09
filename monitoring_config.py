#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控系统配置模块

提供监控系统的配置管理，支持：
1. 从环境变量加载配置
2. 配置文件管理
3. 配置验证和默认值设置

作者：gcddxd12
版本：1.0.0
创建日期：2026-04-09
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("[WARN] PyYAML not available, YAML config files not supported")

# ========== 配置枚举 ==========

class LogFormat(Enum):
    """日志格式枚举"""
    JSON = "json"
    CONSOLE = "console"
    PLAIN = "plain"

class AlertSeverity(Enum):
    """报警严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# ========== 配置数据类 ==========

@dataclass
class LangSmithConfig:
    """LangSmith配置"""
    api_key: Optional[str] = None
    project: str = "enterprise-agent-monitoring"
    endpoint: str = "https://api.smith.langchain.com"
    enabled: bool = True
    trace_all_nodes: bool = True
    sample_rate: float = 1.0  # 跟踪采样率

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("LANGSMITH_API_KEY")

        if not self.enabled and self.api_key:
            # 有API密钥但禁用，需要提示
            print("[INFO] LangSmith tracing is disabled despite API key being available")

@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: LogFormat = LogFormat.JSON
    file_path: Optional[str] = "./logs/agent_monitoring.log"
    enable_structured_logging: bool = True
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    max_file_size_mb: int = 100
    backup_count: int = 5
    json_indent: Optional[int] = None

    def __post_init__(self):
        # 确保日志目录存在
        if self.file_path and self.enable_file_logging:
            log_dir = os.path.dirname(self.file_path)
            os.makedirs(log_dir, exist_ok=True)

@dataclass
class MetricsConfig:
    """指标配置"""
    enabled: bool = True
    port: int = 9090
    collect_interval_seconds: int = 30
    enable_prometheus: bool = True
    enable_custom_metrics: bool = True
    metrics_prefix: str = "enterprise_agent_"
    export_interval_seconds: int = 60

    # 性能监控
    enable_performance_monitoring: bool = True
    performance_sample_rate: float = 0.1  # 10%采样率
    performance_history_size: int = 1000

@dataclass
class AlertsConfig:
    """报警配置"""
    enabled: bool = True
    webhook_url: Optional[str] = None
    email: Optional[str] = None
    slack_webhook: Optional[str] = None
    enable_console_alerts: bool = True

    # 报警阈值
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate": 0.05,  # 错误率超过5%
        "response_time_p95_ms": 3000,  # P95响应时间超过3秒
        "tool_failure_rate": 0.1,  # 工具失败率超过10%
        "concurrent_requests_max": 100,  # 最大并发请求数
        "memory_usage_gb": 2.0,  # 内存使用超过2GB
    })

    # 报警冷却时间（秒）
    cooldown_periods: Dict[str, int] = field(default_factory=lambda: {
        "error_rate": 300,  # 5分钟
        "response_time": 300,
        "tool_failure": 600,  # 10分钟
        "memory_usage": 900,  # 15分钟
    })

    def __post_init__(self):
        if not self.webhook_url:
            self.webhook_url = os.getenv("ALERT_WEBHOOK_URL")

        if not self.email:
            self.email = os.getenv("ALERT_EMAIL")

        if not self.slack_webhook:
            self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL")

@dataclass
class TracingConfig:
    """跟踪配置"""
    enabled: bool = True
    enable_request_tracing: bool = True
    enable_response_tracing: bool = True
    enable_error_tracing: bool = True
    enable_performance_tracing: bool = True

    # 跟踪数据保留
    trace_retention_days: int = 30
    max_trace_size_kb: int = 1024  # 单个跟踪最大大小

    # 采样配置
    sampling_rate: float = 1.0  # 100%采样
    adaptive_sampling: bool = False
    min_sampling_rate: float = 0.01  # 最低1%采样率

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

    # 高级配置
    buffer_size: int = 1000  # 监控数据缓冲区大小
    flush_interval_seconds: int = 5  # 数据刷新间隔
    max_workers: int = 4  # 监控工作线程数

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

    def _apply_staging_settings(self):
        """应用预发布环境设置"""
        self.logging.level = "DEBUG"
        self.logging.format = LogFormat.CONSOLE
        self.alerts.enabled = True
        self.tracing.sampling_rate = 0.5  # 预发布环境50%采样
        self.metrics.enabled = True

    def _apply_development_settings(self):
        """应用开发环境设置"""
        self.logging.level = "DEBUG"
        self.logging.format = LogFormat.CONSOLE
        self.alerts.enabled = False
        self.tracing.sampling_rate = 1.0  # 开发环境100%采样
        self.metrics.enabled = False
        self.langsmith.enabled = False  # 开发环境默认禁用LangSmith

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    def to_yaml(self) -> str:
        """转换为YAML字符串"""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML export")

        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)

    def save_to_file(self, filepath: str, format: str = "json"):
        """保存配置到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if format.lower() == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.to_json())
        elif format.lower() == "yaml" and YAML_AVAILABLE:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.to_yaml())
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"[INFO] Configuration saved to: {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> 'MonitoringConfig':
        """从文件加载配置"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML is required for YAML config files")
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'MonitoringConfig':
        """从字典创建配置对象"""
        # 处理嵌套配置
        langsmith_data = data.get('langsmith', {})
        logging_data = data.get('logging', {})
        metrics_data = data.get('metrics', {})
        alerts_data = data.get('alerts', {})
        tracing_data = data.get('tracing', {})

        # 处理日志格式枚举
        if 'format' in logging_data:
            logging_data['format'] = LogFormat(logging_data['format'])

        return cls(
            langsmith=LangSmithConfig(**langsmith_data),
            logging=LoggingConfig(**logging_data),
            metrics=MetricsConfig(**metrics_data),
            alerts=AlertsConfig(**alerts_data),
            tracing=TracingConfig(**tracing_data),
            service_name=data.get('service_name', 'enterprise-agent'),
            environment=data.get('environment', 'development'),
            version=data.get('version', '1.0.0'),
            instance_id=data.get('instance_id', 'default'),
            enable_monitoring=data.get('enable_monitoring', True),
            enable_health_checks=data.get('enable_health_checks', True),
            enable_dashboard=data.get('enable_dashboard', True),
            dashboard_port=data.get('dashboard_port', 8080),
            buffer_size=data.get('buffer_size', 1000),
            flush_interval_seconds=data.get('flush_interval_seconds', 5),
            max_workers=data.get('max_workers', 4)
        )

    @classmethod
    def create_default_config(cls) -> 'MonitoringConfig':
        """创建默认配置"""
        return cls()

# ========== 配置管理器 ==========

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

    def load_from_file(self, filepath: str):
        """从文件加载配置"""
        self.config = MonitoringConfig.load_from_file(filepath)
        self.config_file = filepath
        print(f"[INFO] Configuration loaded from file: {filepath}")
        return self

    def save_to_file(self, filepath: str = None, format: str = "json"):
        """保存配置到文件"""
        if filepath is None:
            if self.config_file is None:
                filepath = "./config/monitoring_config.json"
            else:
                filepath = self.config_file

        self.config.save_to_file(filepath, format)
        self.config_file = filepath
        return self

    def watch_for_changes(self, callback: Optional[callable] = None):
        """监视配置变化（可选功能）"""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            if self.config_file is None:
                print("[WARN] Cannot watch config changes: no config file specified")
                return

            class ConfigChangeHandler(FileSystemEventHandler):
                def on_modified(self, event):
                    if event.src_path == self.config_file:
                        print(f"[INFO] Config file changed: {event.src_path}")
                        self.load_from_file(self.config_file)
                        if callback:
                            callback(self.config)

            event_handler = ConfigChangeHandler()
            self.watcher = Observer()
            self.watcher.schedule(event_handler, os.path.dirname(self.config_file), recursive=False)
            self.watcher.start()
            print(f"[INFO] Config file watcher started for: {self.config_file}")

        except ImportError:
            print("[WARN] watchdog not installed, config watching disabled")
        except Exception as e:
            print(f"[ERROR] Failed to start config watcher: {e}")

    def stop_watching(self):
        """停止监视配置变化"""
        if self.watcher:
            self.watcher.stop()
            self.watcher.join()
            print("[INFO] Config file watcher stopped")

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

        # 验证阈值配置
        for threshold_name, threshold_value in self.config.alerts.thresholds.items():
            if not isinstance(threshold_value, (int, float)):
                errors.append(f"Threshold '{threshold_name}' must be a number")
            elif threshold_value < 0:
                errors.append(f"Threshold '{threshold_name}' cannot be negative")

        if errors:
            print("[ERROR] Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False

        print("[INFO] Configuration validation passed")
        return True

    def get_config_for_component(self, component_name: str) -> Dict[str, Any]:
        """获取组件特定配置"""
        component_configs = {
            "langsmith": self.config.langsmith,
            "logging": self.config.logging,
            "metrics": self.config.metrics,
            "alerts": self.config.alerts,
            "tracing": self.config.tracing,
        }

        config = component_configs.get(component_name)
        if config:
            return asdict(config)
        else:
            raise ValueError(f"Unknown component: {component_name}")

# ========== 默认配置实例 ==========

def get_default_config() -> MonitoringConfig:
    """获取默认配置"""
    return MonitoringConfig.create_default_config()

def get_config_manager() -> ConfigManager:
    """获取配置管理器实例"""
    config = MonitoringConfig.create_default_config()
    manager = ConfigManager(config)
    manager.load_from_env()
    return manager

# ========== 测试函数 ==========

def test_config_module():
    """测试配置模块"""
    print("=== 测试配置模块 ===")

    try:
        # 创建默认配置
        config = get_default_config()
        print(f"默认配置创建成功")
        print(f"环境: {config.environment}")
        print(f"日志级别: {config.logging.level}")
        print(f"日志格式: {config.logging.format.value}")

        # 测试配置管理器
        manager = get_config_manager()
        print(f"\n配置管理器创建成功")

        # 验证配置
        is_valid = manager.validate()
        print(f"配置验证: {'通过' if is_valid else '失败'}")

        # 测试JSON导出
        json_config = config.to_json(indent=2)
        print(f"\nJSON配置 (前200字符): {json_config[:200]}...")

        # 测试文件保存
        test_config_dir = "./test_config"
        os.makedirs(test_config_dir, exist_ok=True)

        json_file = os.path.join(test_config_dir, "monitoring_config.json")
        config.save_to_file(json_file)
        print(f"配置已保存到: {json_file}")

        # 测试文件加载
        loaded_config = MonitoringConfig.load_from_file(json_file)
        print(f"配置已从文件加载")

        # 清理
        import shutil
        shutil.rmtree(test_config_dir, ignore_errors=True)

        print("\n[SUCCESS] 配置模块测试完成")
        return True

    except Exception as e:
        print(f"[FAILED] 配置模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# ========== 主入口 ==========

if __name__ == "__main__":
    print("监控系统配置模块")
    print("功能: 配置管理、环境变量加载、配置文件支持")

    # 运行测试
    test_config_module()