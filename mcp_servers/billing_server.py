"""
Mock 账单查询 MCP Server — 中国移动计费系统

通过 stdin/stdout 提供 JSON-RPC 2.0 协议，模拟真实账单系统的 MCP 能力。
演示工具：余额查询(billing_query_balance)、月账单(billing_query_monthly)、
流量余量(billing_query_flow_remaining)、缴费记录(billing_query_pay_history)。

纯标准库实现，无需 mcp SDK，遵循 MCP JSON-RPC 规范。
"""

import sys
import json
import uuid
from datetime import datetime, timedelta


def log(msg: str) -> None:
    """输出日志到 stderr（stdout 用于 JSON-RPC，不能污染）"""
    print(f"[billing-server] {msg}", file=sys.stderr, flush=True)


# ========== Mock 数据 ==========

MOCK_ACCOUNTS = {
    "13800001111": {
        "name": "张三",
        "balance": 86.50,
        "level": "五星用户",
        "credit": 500,
    },
    "13800002222": {
        "name": "李四",
        "balance": 12.30,
        "level": "三星用户",
        "credit": 200,
    },
    "13800003333": {
        "name": "王五",
        "balance": -5.00,
        "level": "二星用户",
        "credit": 100,
    },
}


def get_account(phone: str) -> dict | None:
    return MOCK_ACCOUNTS.get(phone, MOCK_ACCOUNTS.get("13800001111"))


# ========== 工具处理函数 ==========


def handle_query_balance(phone: str) -> str:
    account = get_account(phone)
    if not account:
        return f"未找到手机号 {phone} 的账户信息，请确认号码是否正确。"
    status = "欠费" if account["balance"] < 0 else "正常"
    lines = [
        f"【余额查询】{account['name']} | {phone}",
        f"当前余额: {account['balance']:.2f} 元",
        f"账户状态: {status}",
        f"用户等级: {account['level']} | 信用额度: {account['credit']} 元",
        f"（Mock 数据，仅供演示）",
    ]
    return "\n".join(lines)


def handle_query_monthly(phone: str, month: str | None = None) -> str:
    account = get_account(phone)
    if not account:
        return f"未找到手机号 {phone} 的账户信息。"
    month = month or datetime.now().strftime("%Y-%m")
    lines = [
        f"【{month} 月账单】{account['name']} | {phone}",
        f"套餐月租: 58.00 元",
        f"通话费:   12.50 元（国内通话 42 分钟）",
        f"流量费:    8.00 元（超出套餐 1.2GB）",
        f"增值业务:  0.00 元",
        f"------------------------",
        f"合计:     78.50 元",
        f"（Mock 数据，仅供演示）",
    ]
    return "\n".join(lines)


def handle_query_flow_remaining(phone: str) -> str:
    account = get_account(phone)
    if not account:
        return f"未找到手机号 {phone} 的账户信息。"
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"【流量余量查询】{account['name']} | {phone}",
        f"套餐总流量: 30 GB",
        f"已使用:     18.5 GB",
        f"剩余:       11.5 GB",
        f"结算日期:   每月 1 日",
        f"查询时间:   {today}",
        f"（Mock 数据，仅供演示）",
    ]
    return "\n".join(lines)


def handle_query_pay_history(phone: str, limit: int = 5) -> str:
    account = get_account(phone)
    if not account:
        return f"未找到手机号 {phone} 的账户信息。"
    lines = [f"【最近 {limit} 笔缴费记录】{account['name']} | {phone}"]
    base_date = datetime.now()
    for i in range(limit):
        date = (base_date - timedelta(days=30 * i)).strftime("%Y-%m-%d")
        amount = [100, 50, 80, 200, 30][i] if i < 5 else 50
        channel = ["微信", "支付宝", "中国移动APP", "营业厅", "银行代扣"][i] if i < 5 else "微信"
        lines.append(f"  {date}  +{amount:.2f} 元  ({channel})")
    lines.append("（Mock 数据，仅供演示）")
    return "\n".join(lines)


TOOL_HANDLERS = {
    "billing_query_balance": {
        "handler": lambda args: handle_query_balance(args.get("phone", "")),
        "description": "查询用户账户余额。输入手机号，返回当前余额、账户状态、用户等级和信用额度。",
        "params": {"phone": {"type": "string", "description": "要查询的手机号", "required": True}},
    },
    "billing_query_monthly": {
        "handler": lambda args: handle_query_monthly(
            args.get("phone", ""), args.get("month")
        ),
        "description": "查询用户月消费明细。输入手机号，返回当月或指定月份的账单明细（套餐费、通话费、流量费等分类）。",
        "params": {
            "phone": {"type": "string", "description": "要查询的手机号", "required": True},
            "month": {"type": "string", "description": "查询月份，如 2026-04，为空则查当月", "required": False},
        },
    },
    "billing_query_flow_remaining": {
        "handler": lambda args: handle_query_flow_remaining(args.get("phone", "")),
        "description": "查询用户当前流量余量。输入手机号，返回套餐总流量、已用流量、剩余流量。",
        "params": {"phone": {"type": "string", "description": "要查询的手机号", "required": True}},
    },
    "billing_query_pay_history": {
        "handler": lambda args: handle_query_pay_history(
            args.get("phone", ""), args.get("limit", 5)
        ),
        "description": "查询用户近期缴费记录。输入手机号，返回最近几笔充值/缴费记录。",
        "params": {
            "phone": {"type": "string", "description": "要查询的手机号", "required": True},
            "limit": {"type": "integer", "description": "返回记录数，默认5笔", "required": False},
        },
    },
}


def build_tools_list() -> list:
    """构建 MCP tools/list 响应"""
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
        tools.append(
            {
                "name": name,
                "description": info["description"],
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        )
    return tools


# ========== JSON-RPC 处理 ==========


def handle_request(req: dict) -> dict:
    """处理单个 JSON-RPC 请求"""
    method = req.get("method", "")
    req_id = req.get("id")

    log(f"收到请求: {method}")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "billing-server",
                    "version": "1.0.0",
                },
                "capabilities": {
                    "tools": {},
                },
            },
        }

    elif method == "notifications/initialized":
        # 通知不需要响应
        return None

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": build_tools_list(),
            },
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
                    "content": [
                        {"type": "text", "text": f"错误：未知工具 '{tool_name}'"}
                    ],
                    "isError": True,
                },
            }

        try:
            result_text = tool["handler"](arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {"type": "text", "text": result_text}
                    ],
                    "isError": False,
                },
            }
        except Exception as e:
            log(f"工具执行错误: {e}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {"type": "text", "text": f"执行失败: {e}"}
                    ],
                    "isError": True,
                },
            }

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        }


def main() -> None:
    """主循环：从 stdin 读取 JSON-RPC 请求，向 stdout 写入响应"""
    log("账单查询 MCP Server 启动，等待连接...")

    # 使用缓冲读取，一次读取一行 JSON
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
