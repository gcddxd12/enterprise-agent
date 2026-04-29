"""
内存仓储实现（演示/测试用）

数据均为模拟数据，通过 REPO_BACKEND=memory 激活。
替换为真实实现时，只需实现 base.py 中的接口。
"""

from typing import Dict, Any, List, Optional
from .base import TicketRepository, BillingRepository, KnowledgeRepository, EscalationRepository


class MemoryTicketRepository(TicketRepository):
    """内存工单仓储 — 6 条模拟工单"""

    def __init__(self):
        self._tickets: Dict[str, Dict[str, Any]] = {
            "TK-123456": {
                "ticket_id": "TK-123456",
                "phone": "13800001111",
                "type": "网络投诉",
                "status": "处理中",
                "priority": "高",
                "created_at": "2026-04-25 14:30",
                "handler": "张工",
                "detail": "用户反映5G信号不稳定，室内基本无信号，已派单至网络优化中心",
                "history": [
                    {"time": "2026-04-25 14:30", "action": "创建工单"},
                    {"time": "2026-04-25 15:00", "action": "派单至网络优化中心"},
                    {"time": "2026-04-26 09:00", "action": "工程师已联系用户，预约上门检测"},
                ],
            },
            "TK-789012": {
                "ticket_id": "TK-789012",
                "phone": "13800001111",
                "type": "账单争议",
                "status": "已解决",
                "priority": "中",
                "created_at": "2026-04-20 10:00",
                "handler": "李工",
                "detail": "用户质疑4月账单多扣费58元，经核实为定向流量包自动续费，已退款",
                "history": [
                    {"time": "2026-04-20 10:00", "action": "创建工单"},
                    {"time": "2026-04-20 16:00", "action": "核实账单明细"},
                    {"time": "2026-04-21 11:00", "action": "确认误扣费，发起退款"},
                    {"time": "2026-04-22 08:00", "action": "退款到账，工单关闭"},
                ],
            },
            "TK-555888": {
                "ticket_id": "TK-555888",
                "phone": "13900002222",
                "type": "套餐变更",
                "status": "待处理",
                "priority": "低",
                "created_at": "2026-04-28 16:00",
                "handler": "王工",
                "detail": "用户申请从98元套餐降级为58元套餐，需确认合约期内是否有违约金",
                "history": [
                    {"time": "2026-04-28 16:00", "action": "创建工单"},
                ],
            },
            "TK-111222": {
                "ticket_id": "TK-111222",
                "phone": "13700003333",
                "type": "SIM卡补换",
                "status": "已完成",
                "priority": "高",
                "created_at": "2026-04-27 09:00",
                "handler": "赵工",
                "detail": "用户异地补卡，通过身份验证后办理，新SIM卡已邮寄",
                "history": [
                    {"time": "2026-04-27 09:00", "action": "创建工单"},
                    {"time": "2026-04-27 10:00", "action": "身份验证通过"},
                    {"time": "2026-04-27 14:00", "action": "SIM卡寄出，工单关闭"},
                ],
            },
            "TK-333444": {
                "ticket_id": "TK-333444",
                "phone": "13600004444",
                "type": "宽带报修",
                "status": "处理中",
                "priority": "高",
                "created_at": "2026-04-28 08:00",
                "handler": "孙工",
                "detail": "宽带频繁掉线，光猫指示灯异常，已派单至装维团队",
                "history": [
                    {"time": "2026-04-28 08:00", "action": "创建工单"},
                    {"time": "2026-04-28 08:30", "action": "远程检测光猫离线"},
                    {"time": "2026-04-28 09:00", "action": "派单至装维团队，预约上门"},
                ],
            },
        }

    def get_status(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        return self._tickets.get(ticket_id.strip().upper())

    def list_by_phone(self, phone: str) -> List[Dict[str, Any]]:
        return [t for t in self._tickets.values() if t["phone"] == phone.strip()]


class MemoryBillingRepository(BillingRepository):
    """内存账单仓储 — 2 个模拟用户"""

    def __init__(self):
        self._accounts: Dict[str, Dict[str, Any]] = {
            "13800001111": {
                "phone": "13800001111",
                "name": "张三",
                "balance": 86.50,
                "plan_name": "5G畅享套餐98元档",
                "flow_total": 30,  # GB
                "flow_used": 18.5,
                "voice_total": 500,  # 分钟
                "voice_used": 120,
                "credit_level": "三星",
                "monthly_bills": {
                    "2026-04": {
                        "plan_fee": 98.00,
                        "voice_extra": 12.50,
                        "data_extra": 8.00,
                        "value_added": 0,
                        "discount": -10.00,
                        "total": 108.50,
                    },
                    "2026-03": {
                        "plan_fee": 98.00,
                        "voice_extra": 5.00,
                        "data_extra": 0,
                        "value_added": 6.00,
                        "discount": -10.00,
                        "total": 99.00,
                    },
                },
            },
            "13900002222": {
                "phone": "13900002222",
                "name": "李四",
                "balance": 23.80,
                "plan_name": "5G畅享套餐58元档",
                "flow_total": 10,
                "flow_used": 9.2,
                "voice_total": 200,
                "voice_used": 185,
                "credit_level": "二星",
                "monthly_bills": {
                    "2026-04": {
                        "plan_fee": 58.00,
                        "voice_extra": 25.00,
                        "data_extra": 15.00,
                        "value_added": 0,
                        "discount": 0,
                        "total": 98.00,
                    },
                },
            },
        }

    def get_balance(self, phone: str) -> Optional[Dict[str, Any]]:
        return self._accounts.get(phone.strip())

    def get_monthly_bill(self, phone: str, month: str) -> Optional[Dict[str, Any]]:
        account = self._accounts.get(phone.strip())
        if account:
            return account.get("monthly_bills", {}).get(month)
        return None

    def get_flow_remaining(self, phone: str) -> Optional[Dict[str, Any]]:
        account = self._accounts.get(phone.strip())
        if account:
            return {
                "phone": account["phone"],
                "flow_total": account["flow_total"],
                "flow_used": account["flow_used"],
                "flow_remaining": account["flow_total"] - account["flow_used"],
                "plan_name": account["plan_name"],
            }
        return None


class MemoryKnowledgeRepository(KnowledgeRepository):
    """内存知识库仓储 — 中国移动常见业务知识"""

    def __init__(self):
        self._knowledge = [
            {
                "content": "中国移动提供多种5G套餐，从39元至399元不等，包含不同额度的国内流量和语音通话时间。"
                "用户可通过中国移动APP、10086热线或营业厅查询和办理套餐变更。",
                "category": "套餐资费",
            },
            {
                "content": "中国移动FTTR全屋光宽带采用XGS-PON+Wi-Fi7融合组网，实现家庭全域光纤入室、无缝漫游，"
                "支持多设备并发高速上网。提供500M/1000M/2000M等多档速率选择。",
                "category": "宽带业务",
            },
            {
                "content": "中国移动5G套餐包含通用流量和定向流量。通用流量可在国内任意网络环境下使用；"
                "定向流量适用于指定APP（如抖音、微信、腾讯视频等）免流。超出套餐后按套外资费计费。",
                "category": "流量业务",
            },
            {
                "content": "中国移动提供异地补卡服务。用户需携带本人有效身份证件到指定营业厅办理，"
                "部分省市支持线上视频认证后邮寄新卡。补卡费用10元/张。",
                "category": "SIM卡业务",
            },
            {
                "content": "话费余额包含通用余额、赠送余额和专用余额三类。通用余额可用于所有消费；"
                "赠送余额通常有有效期限制；专用余额仅限特定业务（如流量包）使用。",
                "category": "话费账单",
            },
            {
                "content": "投诉处理流程：客服受理→问题核实→制定方案→处理解决→回访确认。"
                "普通投诉24小时内回复，紧急投诉2小时内响应。升级投诉可拨打10086转人工或通过工信部申诉。",
                "category": "投诉处理",
            },
            {
                "content": "定向流量和通用流量的区别：通用流量在所有网络环境和APP中均可使用；"
                "定向流量仅限特定合作APP（如抖音、爱奇艺、微信等）使用，且不包含APP内广告和第三方链接。",
                "category": "流量业务",
            },
            {
                "content": "中国移动物联网业务提供NB-IoT、LTE Cat.1、5G NR等多种接入方式，"
                "支持智能抄表、车联网、智慧农业、工业物联网等场景。提供设备管理平台OneNET。",
                "category": "物联网",
            },
            {
                "content": "中国移动云电脑是基于云计算技术的虚拟桌面服务，用户可通过任何终端访问云端Windows/Linux桌面。"
                "支持弹性扩容、数据云端存储、多设备同步。",
                "category": "云计算",
            },
            {
                "content": "中国移动网络安全服务包括DDoS防护、Web应用防火墙、数据加密传输、"
                "安全态势感知等。为政企客户提供等级保护2.0合规方案。",
                "category": "网络安全",
            },
        ]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = []
        query_lower = query.lower()
        for item in self._knowledge:
            score = 0
            content_lower = item["content"].lower()
            # 简单关键词匹配打分
            for word in query_lower.replace("，", " ").replace("。", " ").split():
                if len(word) >= 2 and word in content_lower:
                    score += 1
            if score > 0:
                results.append({"content": item["content"], "category": item["category"], "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


class MemoryEscalationRepository(EscalationRepository):
    """内存转人工仓储"""

    def escalate(self, query: str, priority: str = "normal") -> Dict[str, Any]:
        return {
            "escalation_id": f"ESC-{hash(query) % 100000:05d}",
            "status": "已提交",
            "message": "感谢您的耐心，我已将您的问题转接给人工客服，他们将尽快与您联系（预计5分钟内）。",
            "priority": priority,
            "query": query,
        }
