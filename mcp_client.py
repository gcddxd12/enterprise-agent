"""
MCP Client Manager — 管理外部 MCP Server 连接，将 MCP 工具包装为 LangChain Tool。

通过 subprocess 启动 MCP Server（stdin/stdout JSON-RPC 2.0），
发现工具列表，包装为 LangChain StructuredTool，统一集成到 Agent 的工具分发中。

纯同步实现，与 agent_node 的同步 ReAct 循环无缝配合。
"""

import json
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from langchain_core.tools import StructuredTool

import yaml
from pydantic import create_model, Field


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""


@dataclass
class MCPToolInfo:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    server_name: str


class MCPClientManager:
    """MCP 客户端管理器：连接 MCP Server，发现工具，执行调用"""

    def __init__(self, config_path: str = "./mcp_servers.yaml"):
        self.config_path = config_path
        self.servers: Dict[str, MCPServerConfig] = {}
        self.tools: Dict[str, MCPToolInfo] = {}  # tool_name -> info
        self._processes: Dict[str, subprocess.Popen] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._load_config()

    def _load_config(self):
        """从 YAML 加载 MCP Server 配置"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            servers = data.get("servers", {})
            for key, s in servers.items():
                self.servers[key] = MCPServerConfig(
                    name=s.get("name", key),
                    command=s.get("command", "python"),
                    args=s.get("args", []),
                    env=s.get("env", {}),
                    enabled=s.get("enabled", True),
                    description=s.get("description", ""),
                )
            print(f"[MCPClient] 加载了 {len(self.servers)} 个 MCP Server 配置")
        except FileNotFoundError:
            print(f"[MCPClient] 配置文件不存在: {self.config_path}")
        except Exception as e:
            print(f"[MCPClient] 加载配置失败: {e}")

    def connect_all(self) -> int:
        """连接所有启用的 MCP Server，发现并注册工具。返回工具总数。"""
        total = 0
        for key, config in self.servers.items():
            if not config.enabled:
                print(f"[MCPClient] 跳过已禁用的 Server: {config.name}")
                continue
            try:
                count = self._connect_server(key, config)
                total += count
                print(f"[MCPClient] {config.name}: 发现 {count} 个工具")
            except Exception as e:
                print(f"[MCPClient] 连接 {config.name} 失败: {e}")
        print(f"[MCPClient] 总计发现 {total} 个 MCP 工具")
        return total

    def _connect_server(self, key: str, config: MCPServerConfig) -> int:
        """连接单个 MCP Server，发送 initialize + tools/list，解析工具列表"""
        # 准备环境变量（确保 UTF-8 编码）
        import os
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env.update(config.env)

        # 启动子进程
        try:
            process = subprocess.Popen(
                [config.command] + config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=env,
                bufsize=1,  # 行缓冲
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"无法启动 MCP Server (命令: {config.command}): {e}")

        self._processes[key] = process
        self._locks[key] = threading.Lock()

        try:
            # 1. 发送 initialize
            init_resp = self._send_request(key, "initialize", {})
            if "error" in init_resp:
                raise RuntimeError(f"初始化失败: {init_resp['error']}")
            server_info = init_resp.get("result", {}).get("serverInfo", {})
            print(f"[MCPClient] {config.name} 已连接 ({server_info.get('name', 'unknown')} v{server_info.get('version', '?')})")

            # 2. 发送 initialized 通知
            self._send_notification(key, "notifications/initialized")

            # 3. 获取工具列表
            tools_resp = self._send_request(key, "tools/list", {})
            if "error" in tools_resp:
                raise RuntimeError(f"获取工具列表失败: {tools_resp['error']}")

            tools = tools_resp.get("result", {}).get("tools", [])
            for t in tools:
                tool_info = MCPToolInfo(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=t.get("inputSchema", {}),
                    server_name=config.name,
                )
                self.tools[t["name"]] = tool_info

            return len(tools)

        except Exception as e:
            # 连接失败时清理进程
            self._disconnect_server(key)
            raise e

    def _send_request(self, key: str, method: str, params: dict) -> dict:
        """发送 JSON-RPC 请求并接收响应"""
        import uuid
        req = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex[:8],
            "method": method,
            "params": params,
        }
        return self._communicate(key, req)

    def _send_notification(self, key: str, method: str, params: dict | None = None) -> None:
        """发送 JSON-RPC 通知（无需响应）"""
        notif = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        self._write_line(key, json.dumps(notif, ensure_ascii=False))

    def _communicate(self, key: str, req: dict) -> dict:
        """向 MCP Server 发送请求并读取响应（线程安全）"""
        lock = self._locks.get(key)
        if not lock:
            raise RuntimeError(f"MCP Server '{key}' 未连接")

        process = self._processes.get(key)
        if not process or process.poll() is not None:
            raise RuntimeError(f"MCP Server '{key}' 已断开")

        with lock:
            try:
                # 发送请求
                self._write_line(key, json.dumps(req, ensure_ascii=False))

                # 读取响应
                response_line = process.stdout.readline()
                if not response_line:
                    raise RuntimeError(f"MCP Server '{key}' 无响应（可能已崩溃）")

                resp = json.loads(response_line.strip())
                return resp

            except json.JSONDecodeError as e:
                raise RuntimeError(f"MCP Server '{key}' 返回的响应无法解析: {e}")
            except Exception as e:
                raise RuntimeError(f"与 MCP Server '{key}' 通信失败: {e}")

    def _write_line(self, key: str, line: str) -> None:
        """向 MCP Server 写入一行"""
        process = self._processes.get(key)
        if not process or process.poll() is not None:
            raise RuntimeError(f"MCP Server '{key}' 已断开")
        process.stdin.write(line + "\n")
        process.stdin.flush()

    def call_tool(self, tool_name: str, arguments: Dict) -> str:
        """调用 MCP 工具并返回文本结果"""
        tool = self.tools.get(tool_name)
        if not tool:
            return f"MCP 工具不存在: {tool_name}"

        # 找到该工具所属的 Server key
        server_key = None
        for key, config in self.servers.items():
            if config.name == tool.server_name:
                server_key = key
                break

        if not server_key:
            return f"未找到工具 {tool_name} 所属的 MCP Server"

        try:
            resp = self._send_request(server_key, "tools/call", {
                "name": tool_name,
                "arguments": arguments,
            })

            if "error" in resp:
                return f"MCP 调用失败: {resp['error']}"

            result = resp.get("result", {})
            content = result.get("content", [])

            # 拼接所有 text content
            texts = []
            for block in content:
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
            return "\n".join(texts) if texts else "MCP 工具返回了空结果"

        except Exception as e:
            # 检查进程是否存活
            process = self._processes.get(server_key)
            if process and process.poll() is not None:
                stderr = process.stderr.read() if process.stderr else ""
                return f"MCP Server '{tool.server_name}' 已崩溃 (exitcode={process.poll()}). stderr: {stderr[:200]}"
            return f"MCP 工具调用失败: {e}"

    def list_tools(self) -> List[MCPToolInfo]:
        """返回所有已发现 MCP 工具的信息列表"""
        return list(self.tools.values())

    def get_tool_descriptions(self) -> str:
        """生成 MCP 工具的文本描述（用于追加到系统提示）"""
        if not self.tools:
            return ""
        lines = ["\n## MCP 外部工具（来自 MCP Server）"]
        for t in self.tools.values():
            lines.append(f"- **{t.name}**: {t.description}（来自: {t.server_name}）")
        return "\n".join(lines)

    def close_all(self):
        """关闭所有 MCP Server 连接"""
        for key in list(self._processes.keys()):
            self._disconnect_server(key)

    def _disconnect_server(self, key: str):
        """断开单个 MCP Server"""
        process = self._processes.pop(key, None)
        self._locks.pop(key, None)
        if process:
            try:
                process.stdin.close()
                process.stdout.close()
                process.stderr.close()
            except Exception:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            print(f"[MCPClient] 已断开: {key}")

    def get_status(self) -> Dict:
        """返回 MCP 连接状态摘要"""
        status = {
            "total_servers": len(self.servers),
            "connected_servers": len([k for k in self._processes if self._processes[k].poll() is None]),
            "total_tools": len(self.tools),
            "tools_by_server": {},
        }
        for t in self.tools.values():
            srv = t.server_name
            if srv not in status["tools_by_server"]:
                status["tools_by_server"][srv] = []
            status["tools_by_server"][srv].append(t.name)
        return status


# ========== LangChain Tool 包装 ==========


def _build_pydantic_model(name: str, schema: Dict) -> type:
    """从 MCP 的 JSON Schema 动态构建 pydantic 模型（用于 args_schema）"""
    type_map = {
        "string": (str, ...),
        "integer": (int, ...),
        "number": (float, ...),
        "boolean": (bool, ...),
    }
    fields = {}
    props = schema.get("properties", {})
    required = schema.get("required", [])

    for field_name, field_info in props.items():
        py_type, default = type_map.get(field_info.get("type", "string"), (str, ...))
        is_required = field_name in required
        if is_required:
            fields[field_name] = (py_type, Field(description=field_info.get("description", "")))
        else:
            fields[field_name] = (Optional[py_type], Field(default=None, description=field_info.get("description", "")))

    if not fields:
        fields["query"] = (str, Field(default="", description="查询参数"))

    return create_model(f"MCP_{name}_args", **fields)


def create_mcp_langchain_tool(mcp_tool: MCPToolInfo, manager: MCPClientManager) -> StructuredTool:
    """将 MCP 工具信息包装为 LangChain StructuredTool"""

    # 从 JSON Schema 动态生成参数模型
    args_model = _build_pydantic_model(mcp_tool.name, mcp_tool.parameters)

    def _execute(**kwargs) -> str:
        """执行 MCP 工具调用"""
        arguments = {k: v for k, v in kwargs.items() if v is not None}
        return manager.call_tool(mcp_tool.name, arguments)

    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description,
        func=_execute,
        args_schema=args_model,
    )


# ========== 全局单例（延迟初始化） ==========

_mcp_manager: Optional[MCPClientManager] = None


def get_mcp_manager() -> Optional[MCPClientManager]:
    """获取全局 MCPClientManager 实例"""
    global _mcp_manager
    return _mcp_manager


def init_mcp_tools(config_path: str = "./mcp_servers.yaml") -> List[StructuredTool]:
    """初始化 MCP 连接并返回 LangChain 工具列表"""
    global _mcp_manager
    if _mcp_manager is not None:
        print("[MCP] MCPClientManager 已初始化，跳过")
        return []

    try:
        _mcp_manager = MCPClientManager(config_path=config_path)
        count = _mcp_manager.connect_all()

        if count == 0:
            print("[MCP] 未发现任何 MCP 工具（可能所有 Server 都被禁用或连接失败）")
            return []

        tools = []
        for info in _mcp_manager.list_tools():
            lt = create_mcp_langchain_tool(info, _mcp_manager)
            tools.append(lt)
            print(f"[MCP] 已注册工具: {info.name} (来自 {info.server_name})")

        return tools
    except Exception as e:
        print(f"[MCP] 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def shutdown_mcp():
    """关闭所有 MCP 连接"""
    global _mcp_manager
    if _mcp_manager:
        _mcp_manager.close_all()
        _mcp_manager = None
