"""
Mock 工单系统 MCP Server — 中国移动客服工单系统

通过 stdin/stdout 提供 JSON-RPC 2.0 协议，模拟工单系统的 MCP 能力。
演示工具：工单查询(ticket_query_detail)、工单列表(ticket_list_by_phone)、
工单创建(ticket_create)、工单催办(ticket_urge)。

纯标准库实现，遵循 MCP JSON-RPC 规范。
"""

import sys
import json
import uuid
from datetime import datetime, timedelta


def log(msg: str) -> None:
    print(f"[ticket-server] {msg}", file=sys.stderr, flush=True)


# ========== Mock 数据 ==========

MOCK_TICKETS = {
    "TK-123456": {
        "id": "TK-123456",
        "phone": "13800001111",
        "type": "网络投诉",
        "status": "处理中",
        "created_at": "2026-04-25 10:30",
        "handler": "客服专员王芳",
        "description": "用户反映家中5G信号不稳定，网页加载缓慢",
        "updates": [
            {"time": "2026-04-25 10:30", "content": "工单创建，已派单至网络部"},
            {"time": "2026-04-26 09:00", "content": "工程师李工已接单，计划27日上门检测"},
        ],
    },
    "TK-789012": {
        "id": "TK-789012",
        "phone": "13800002222",
        "type": "账单争议",
        "status": "已解决",
        "created_at": "2026-04-20 14:20",
        "handler": "客服专员李明",
        "description": "用户质疑4月账单流量费8元，认为未超出套餐",
        "updates": [
            {"time": "2026-04-20 14:20", "content": "工单创建"},
            {"time": "2026-04-21 10:00", "content": "已核实：用户确实超出套餐1.2GB，计费正确，已短信通知用户"},
        ],
    },
    "TK-555888": {
        "id": "TK-555888",
        "phone": "13800001111",
        "type": "套餐变更申请",
        "status": "待处理",
        "created_at": "2026-04-28 08:00",
        "handler": "系统自动",
        "description": "用户申请从58元套餐变更为88元5G畅享套餐",
        "updates": [
            {"time": "2026-04-28 08:00", "content": "系统自动受理，等待审核"},
        ],
    },
}

# 按手机号组织的工单索引
_PHONE_TICKETS = {}
for t in MOCK_TICKETS.values():
    phone = t["phone"]
    if phone not in _PHONE_TICKETS:
        _PHONE_TICKETS[phone] = []
    _PHONE_TICKETS[phone].append(t)


# ========== 工具处理函数 ==========


def handle_query_detail(ticket_id: str) -> str:
    t = MOCK_TICKETS.get(ticket_id)
    if not t:
        return f"未找到工单 {ticket_id}，请确认工单号是否正确。常用工单号：TK-123456, TK-789012, TK-555888"

    lines = [
        f"【工单详情】{t['id']}",
        f"类型: {t['type']} | 状态: {t['status']}",
        f"手机号: {t['phone']}",
        f"创建时间: {t['created_at']}",
        f"当前处理人: {t['handler']}",
        f"问题描述: {t['description']}",
        f"",
        f"--- 处理记录 ---",
    ]
    for u in t["updates"]:
        lines.append(f"  [{u['time']}] {u['content']}")

    lines.append("（Mock 数据，仅供演示）")
    return "\n".join(lines)


def handle_list_by_phone(phone: str) -> str:
    tickets = _PHONE_TICKETS.get(phone, [])
    if not tickets:
        return f"手机号 {phone} 暂无相关工单。"

    lines = [f"【{phone} 的工单列表】共 {len(tickets)} 条"]
    lines.append("")
    for t in tickets:
        lines.append(f"  {t['id']} | {t['type']} | {t['status']} | {t['created_at']}")
    lines.append("")
    lines.append("如需查看详情，请提供工单号。")
    lines.append("（Mock 数据，仅供演示）")
    return "\n".join(lines)


def handle_create(phone: str, issue_type: str, description: str) -> str:
    new_id = f"TK-{uuid.uuid4().hex[:6].upper()}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    log(f"创建新工单: {new_id}")

    return "\n".join([
        f"【工单创建成功】",
        f"工单号: {new_id}",
        f"手机号: {phone}",
        f"类型: {issue_type}",
        f"描述: {description}",
        f"创建时间: {now}",
        f"状态: 待处理",
        f"",
        f"我们的客服人员会尽快处理您的问题，通常24小时内会有回复。",
        f"您可通过工单号随时查询处理进度。",
        f"（Mock 数据，仅供演示）",
    ])


def handle_urge(ticket_id: str) -> str:
    t = MOCK_TICKETS.get(ticket_id)
    if not t:
        return f"未找到工单 {ticket_id}。"

    return "\n".join([
        f"【催办成功】工单 {ticket_id}",
        f"当前状态: {t['status']}",
        f"处理人: {t['handler']}",
        f"",
        f"已为您提交催办请求，系统将优先处理并通知处理人加急处理。",
        f"建议您保持电话畅通，以便工作人员联系确认。",
        f"（Mock 数据，仅供演示）",
    ])


TOOL_HANDLERS = {
    "ticket_query_detail": {
        "handler": lambda args: handle_query_detail(args.get("ticket_id", "")),
        "description": "查询工单详细信息。输入工单号（如TK-123456），返回工单状态、处理记录、处理人等完整信息。",
        "params": {"ticket_id": {"type": "string", "description": "工单号，格式 TK-xxxxxx", "required": True}},
    },
    "ticket_list_by_phone": {
        "handler": lambda args: handle_list_by_phone(args.get("phone", "")),
        "description": "查询用户的所有工单列表。输入手机号，返回该号码下的所有工单（状态、类型、创建时间）。",
        "params": {"phone": {"type": "string", "description": "手机号", "required": True}},
    },
    "ticket_create": {
        "handler": lambda args: handle_create(
            args.get("phone", ""),
            args.get("issue_type", "一般咨询"),
            args.get("description", ""),
        ),
        "description": "为用户创建新工单。需要手机号、问题类型和详细描述。用于记录投诉、故障报修、业务申请等。",
        "params": {
            "phone": {"type": "string", "description": "用户手机号", "required": True},
            "issue_type": {"type": "string", "description": "问题类型：网络投诉、账单争议、套餐变更申请、故障报修、其他", "required": True},
            "description": {"type": "string", "description": "问题详细描述", "required": True},
        },
    },
    "ticket_urge": {
        "handler": lambda args: handle_urge(args.get("ticket_id", "")),
        "description": "催办工单。用户对处理进度不满意时，可对已有工单进行催办，加速处理。",
        "params": {"ticket_id": {"type": "string", "description": "要催办的工单号", "required": True}},
    },
}


def build_tools_list() -> list:
    tools = []
    for name, info in TOOL_HANDLERS.items():
        properties = {}
        required = []
        for pname, pinfo in info["params"].items():
            properties[pname] = {
                "type": pinfo["type"],
                "description": pinfo["description"],
            }
            if pinfo.get("required"):
                required.append(pname)
        tools.append({
            "name": name,
            "description": info["description"],
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        })
    return tools


# ========== JSON-RPC 处理 ==========


def handle_request(req: dict) -> dict | None:
    method = req.get("method", "")
    req_id = req.get("id")

    log(f"收到请求: {method}")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "ticket-server", "version": "1.0.0"},
                "capabilities": {"tools": {}},
            },
        }

    elif method == "notifications/initialized":
        return None

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": build_tools_list()},
        }

    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        log(f"调用工具: {tool_name}, 参数: {arguments}")

        tool = TOOL_HANDLERS.get(tool_name)
        if not tool:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"错误：未知工具 '{tool_name}'"}],
                    "isError": True,
                },
            }

        try:
            result_text = tool["handler"](arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                    "isError": False,
                },
            }
        except Exception as e:
            log(f"工具执行错误: {e}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"执行失败: {e}"}],
                    "isError": True,
                },
            }

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def main() -> None:
    log("工单系统 MCP Server 启动，等待连接...")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            log(f"JSON 解析错误: {e}")
            continue

        resp = handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()

    log("Server 退出")


if __name__ == "__main__":
    main()
