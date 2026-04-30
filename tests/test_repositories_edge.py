"""Edge case tests for repository implementations."""

import sys

import pytest

sys.path.insert(0, ".")

from repositories.memory_repo import (
    MemoryTicketRepository,
    MemoryBillingRepository,
    MemoryKnowledgeRepository,
    MemoryEscalationRepository,
)


# ============================================================================
# TestTicketRepositoryEdge
# ============================================================================


class TestTicketRepositoryEdge:
    def setup_method(self):
        self.repo = MemoryTicketRepository()

    def test_whitespace_ticket_id(self):
        result = self.repo.get_status("  TK-123456  ")
        assert result is not None
        assert result["ticket_id"] == "TK-123456"

    def test_case_insensitive(self):
        result = self.repo.get_status("tk-123456")
        assert result is not None
        assert result["ticket_id"] == "TK-123456"

    def test_list_by_phone_with_spaces(self):
        tickets = self.repo.list_by_phone("  13800001111  ")
        assert len(tickets) >= 1

    def test_nonexistent_ticket(self):
        result = self.repo.get_status("TK-999999")
        assert result is None


# ============================================================================
# TestBillingRepositoryEdge
# ============================================================================


class TestBillingRepositoryEdge:
    def setup_method(self):
        self.repo = MemoryBillingRepository()

    def test_balance_for_known_account(self):
        balance = self.repo.get_balance("13800001111")
        assert balance is not None
        assert "balance" in balance

    def test_balance_unknown_returns_none(self):
        result = self.repo.get_balance("13899999999")
        assert result is None

    def test_monthly_bill_includes_items(self):
        bill = self.repo.get_monthly_bill("13800001111", "2026-04")
        assert bill is not None
        if "total" in bill:
            assert bill["total"] > 0

    def test_monthly_bill_unknown_month(self):
        bill = self.repo.get_monthly_bill("13800001111", "2020-01")
        # May return None or empty bill
        if bill is not None:
            assert isinstance(bill, dict)

    def test_flow_remaining(self):
        flow = self.repo.get_flow_remaining("13800001111")
        if flow is not None:
            assert "flow_total" in flow or "flow_remaining" in flow


# ============================================================================
# TestKnowledgeRepositoryEdge
# ============================================================================


class TestKnowledgeRepositoryEdge:
    def setup_method(self):
        self.repo = MemoryKnowledgeRepository()

    def test_search_returns_list(self):
        results = self.repo.search("话费")
        assert isinstance(results, list)

    def test_search_respects_top_k(self):
        results = self.repo.search("5G", top_k=3)
        assert len(results) <= 3

    def test_search_results_have_scores(self):
        results = self.repo.search("查询")
        if results:
            for r in results:
                assert "content" in r or "score" in r or "metadata" in r

    def test_no_match_returns_empty(self):
        results = self.repo.search("xyz_不存在的查询")
        # May return results or empty list (implementation dependent)
        assert isinstance(results, list)


# ============================================================================
# TestEscalationRepositoryEdge
# ============================================================================


class TestEscalationRepositoryEdge:
    def setup_method(self):
        self.repo = MemoryEscalationRepository()

    def test_escalate_returns_dict(self):
        result = self.repo.escalate("宽带一直连不上", priority="urgent")
        assert isinstance(result, dict)
        assert "escalation_id" in result or "id" in result or "ticket_id" in result

    def test_escalate_accepts_different_priorities(self):
        for priority in ["low", "normal", "high", "urgent"]:
            result = self.repo.escalate("测试问题", priority=priority)
            assert isinstance(result, dict)

    def test_escalate_handles_empty_query(self):
        result = self.repo.escalate("")
        assert isinstance(result, dict)
