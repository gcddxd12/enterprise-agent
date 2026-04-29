"""
仓储层 — 数据访问抽象

提供统一的数据访问接口，支持通过环境变量切换实现：
  - memory: 内存模拟数据（默认，无需外部依赖）
  - real:   真实数据库/API（待实现）

用法:
    from repositories import get_ticket_repo, get_billing_repo

    repo = get_ticket_repo()
    ticket = repo.get_status("TK-123456")
"""

import os
from typing import Optional
from .base import TicketRepository, BillingRepository, KnowledgeRepository, EscalationRepository
from .memory_repo import (
    MemoryTicketRepository,
    MemoryBillingRepository,
    MemoryKnowledgeRepository,
    MemoryEscalationRepository,
)

_REPO_BACKEND = os.getenv("REPO_BACKEND", "memory")

_ticket_repo: Optional[TicketRepository] = None
_billing_repo: Optional[BillingRepository] = None
_knowledge_repo: Optional[KnowledgeRepository] = None
_escalation_repo: Optional[EscalationRepository] = None


def get_ticket_repo() -> TicketRepository:
    global _ticket_repo
    if _ticket_repo is None:
        if _REPO_BACKEND == "memory":
            _ticket_repo = MemoryTicketRepository()
        else:
            raise ValueError(f"不支持的 REPO_BACKEND: {_REPO_BACKEND}")
    return _ticket_repo


def get_billing_repo() -> BillingRepository:
    global _billing_repo
    if _billing_repo is None:
        if _REPO_BACKEND == "memory":
            _billing_repo = MemoryBillingRepository()
        else:
            raise ValueError(f"不支持的 REPO_BACKEND: {_REPO_BACKEND}")
    return _billing_repo


def get_knowledge_repo() -> KnowledgeRepository:
    global _knowledge_repo
    if _knowledge_repo is None:
        if _REPO_BACKEND == "memory":
            _knowledge_repo = MemoryKnowledgeRepository()
        else:
            raise ValueError(f"不支持的 REPO_BACKEND: {_REPO_BACKEND}")
    return _knowledge_repo


def get_escalation_repo() -> EscalationRepository:
    global _escalation_repo
    if _escalation_repo is None:
        if _REPO_BACKEND == "memory":
            _escalation_repo = MemoryEscalationRepository()
        else:
            raise ValueError(f"不支持的 REPO_BACKEND: {_REPO_BACKEND}")
    return _escalation_repo
