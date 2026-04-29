# MCP工具系统

## 概述

MCP (Model Context Protocol) 是一种标准化协议，允许LLM Agent通过JSON-RPC 2.0 over stdio发现和调用外部工具。本项目实现了MCP客户端，Agent可以连接外部MCP Server获取业务工具（账单查询、工单系统等），Mock Server演示完整链路。

**核心价值**：数据源替换只需改Server内部实现，Agent代码零改动。

## 1. 核心架构

```
用户 → Agent (ReAct Loop, 13个工具)
         ├── 本地工具 (5): knowledge_search, query_ticket_status, escalate_to_human,
         │                 get_current_date, use_skill
         └── MCP 工具 (8): billing_query_*, ticket_*
              └── MCPClientManager
                   ├── MCP Server 1: billing_server  (子进程, stdin/stdout JSON-RPC)
                   └── MCP Server 2: ticket_server   (子进程, stdin/stdout JSON-RPC)
```

## 2. 工作流程

```
1. 启动 → MCPClientManager 加载 mcp_servers.yaml
2. 对每个启用的Server：启动子进程 → 发送 initialize → 发送 tools/list
3. 发现的所有工具 包装为 LangChain StructuredTool（动态pydantic args_schema）
4. 注册到 get_tools() 和 tool_executors 字典
5. Agent ReAct循环中，LLM调用MCP工具 → MCPClientManager.call_tool() → JSON-RPC → 返回结果
```

## 3. 文件清单

| 文件 | 说明 |
|------|------|
| [mcp_client.py](e:\my_multi_agent\mcp_client.py) | MCPClientManager + LangChain工具包装 + 全局单例 |
| [mcp_servers.yaml](e:\my_multi_agent\mcp_servers.yaml) | MCP Server连接配置 |
| [mcp_servers/billing_server.py](e:\my_multi_agent\mcp_servers\billing_server.py) | Mock账单查询MCP Server (4工具) |
| [mcp_servers/ticket_server.py](e:\my_multi_agent\mcp_servers\ticket_server.py) | Mock工单系统MCP Server (4工具) |

## 4. MCP协议实现

### 4.1 JSON-RPC 2.0 over stdio

MCP Server子进程通过stdin/stdout通信，每行一个JSON对象：

```
初始化:
  → {"jsonrpc":"2.0","id":"1","method":"initialize","params":{}}
  ← {"jsonrpc":"2.0","id":"1","result":{"protocolVersion":"2024-11-05",...}}
  → {"jsonrpc":"2.0","method":"notifications/initialized"}

发现工具:
  → {"jsonrpc":"2.0","id":"2","method":"tools/list","params":{}}
  ← {"jsonrpc":"2.0","id":"2","result":{"tools":[...]}}

调用工具:
  → {"jsonrpc":"2.0","id":"3","method":"tools/call","params":{"name":"xxx","arguments":{...}}}
  ← {"jsonrpc":"2.0","id":"3","result":{"content":[{"type":"text","text":"..."}]}}
```

### 4.2 线程安全

每个Server连接配有 `threading.Lock`，确保并发请求串行化，避免跨请求响应混乱。

## 5. Mixed MCP Server实现

### 5.1 billing_server (账单查询服务)

**工具列表**:

| 工具名 | 参数 | 返回 |
|--------|------|------|
| `billing_query_balance` | phone | 余额、账户状态、用户等级、信用额度 |
| `billing_query_monthly` | phone, month? | 月账单明细（套餐费/通话费/流量费/增值业务） |
| `billing_query_flow_remaining` | phone | 套餐总流量、已用流量、剩余流量 |
| `billing_query_pay_history` | phone | 近期缴费/充值记录 |

### 5.2 ticket_server (工单系统服务)

**工具列表**:

| 工具名 | 参数 | 返回 |
|--------|------|------|
| `ticket_query_detail` | ticket_id | 工单详情（类型、状态、处理人、处理记录） |
| `ticket_list_by_phone` | phone | 用户所有工单列表 |
| `ticket_create` | phone, issue_type, description | 新工单号、创建确认 |
| `ticket_urge` | ticket_id | 催办确认 |

## 6. LangChain工具包装

MCP工具需要包装为LangChain `Tool` 对象才能被 `llm.bind_tools()` 识别：

```python
def create_mcp_langchain_tool(mcp_tool: MCPToolInfo, manager) -> StructuredTool:
    # 从MCP的inputSchema (JSON Schema) 动态构建pydantic args_schema
    args_model = _build_pydantic_model(mcp_tool.name, mcp_tool.parameters)

    def _execute(**kwargs) -> str:
        return manager.call_tool(mcp_tool.name, kwargs)

    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description,
        func=_execute,
        args_schema=args_model,  # 动态pydantic模型
    )
```

## 7. 集成到Agent

### 7.1 工具注册 (get_tools修改)

```python
def get_tools():
    AGENT_TOOLS = [
        knowledge_search, query_ticket_status, escalate_to_human,
        get_current_date, create_use_skill_tool(),
    ]
    # 动态添加MCP工具
    mcp_tools = init_mcp_tools()
    AGENT_TOOLS.extend(mcp_tools)
    return AGENT_TOOLS
```

### 7.2 工具执行 (agent_node修改)

MCP工具执行器动态注入 `tool_executors` 字典：

```python
mcp_mgr = get_mcp_manager()
for info in mcp_mgr.list_tools():
    tool_executors[info.name] = lambda a, name=info.name: mcp_mgr.call_tool(
        name, a if isinstance(a, dict) else {"query": str(a)}
    )
```

### 7.3 系统提示注入

`build_system_prompt()` 自动追加MCP工具描述，让LLM感知可用的外部工具：

```markdown
## MCP 外部工具（来自 MCP Server）
- **billing_query_balance**: 查询用户账户余额。输入手机号...（来自: 账单查询服务）
- **ticket_query_detail**: 查询工单详细信息。输入工单号...（来自: 工单系统服务）
...
```

## 8. 从Mock切换到真实数据

Mock Server只是工具壳，换数据源不改Agent代码：

### 方式1：改Server内部 (直连数据库)
```python
# ticket_server.py 中只改 handle_query_detail
def handle_query_detail(ticket_id: str) -> str:
    conn = pymysql.connect(host="10.x.x.x", ...)
    row = conn.execute("SELECT * FROM tickets WHERE id=%s", [ticket_id])
    return format_ticket(dict(row))
```

### 方式2：换配置指向新Server (HTTP API)
```yaml
# mcp_servers.yaml
servers:
  ticket:
    command: python
    args: ["ticket_api_server.py"]  # 新Server调用HTTP API
```

### 方式3：连接第三方MCP Server
```yaml
servers:
  real_ticket:
    command: npx
    args: ["-y", "@company/mcp-ticket-server"]
    env:
      API_KEY: "${TICKET_API_KEY}"
```

**关键**：只要新Server提供相同工具名（如 `ticket_query_detail`），Agent无需任何修改。

## 9. MCP Server开发规范

开发新的MCP Server需遵循以下约定：

1. 通过stdin逐行读取JSON请求，stdout逐行输出JSON响应
2. 日志输出到stderr（不污染stdout的JSON流）
3. 响应 `initialize`（返回serverInfo和capabilities）
4. 响应 `tools/list`（返回工具列表，含name/description/inputSchema）
5. 响应 `tools/call`（接收name和arguments，返回content数组）
6. 设置 `PYTHONIOENCODING=utf-8` 确保中文正常编码

## 10. 当前工具全景

| 类别 | 工具数 | 工具名 |
|------|--------|--------|
| 知识检索 | 1 | knowledge_search |
| 工单管理 | 1+4 | query_ticket_status + ticket_* |
| 转人工 | 1 | escalate_to_human |
| 日期查询 | 1 | get_current_date |
| 技能加载 | 1 | use_skill |
| 账单查询 | 4 | billing_query_* |
| **总计** | **13** | |
