"""
仓储抽象基类

定义数据访问接口，所有实现（memory/real）必须遵循相同契约。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class TicketRepository(ABC):
    """工单数据仓储"""

    @abstractmethod
    def get_status(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        """查询单条工单信息，无结果返回 None"""
        ...

    @abstractmethod
    def list_by_phone(self, phone: str) -> List[Dict[str, Any]]:
        """按手机号查询用户的所有工单"""
        ...


class BillingRepository(ABC):
    """账单数据仓储"""

    @abstractmethod
    def get_balance(self, phone: str) -> Optional[Dict[str, Any]]:
        """查询账户余额"""
        ...

    @abstractmethod
    def get_monthly_bill(self, phone: str, month: str) -> Optional[Dict[str, Any]]:
        """查询月度账单明细"""
        ...

    @abstractmethod
    def get_flow_remaining(self, phone: str) -> Optional[Dict[str, Any]]:
        """查询剩余流量"""
        ...


class KnowledgeRepository(ABC):
    """知识库仓储（RAG fallback 用，优先级低于向量检索）"""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """关键词检索知识条目，返回 [{content, category, score}]"""
        ...


class EscalationRepository(ABC):
    """升级/转人工 仓储"""

    @abstractmethod
    def escalate(self, query: str, priority: str = "normal") -> Dict[str, Any]:
        """提交转人工申请，返回工单信息"""
        ...
