"""
仓储层单元测试
验证所有仓储接口和内存实现。
"""


class TestTicketRepository:
    def test_get_known_ticket(self):
        from repositories import get_ticket_repo
        repo = get_ticket_repo()
        ticket = repo.get_status("TK-123456")
        assert ticket is not None
        assert ticket["ticket_id"] == "TK-123456"
        assert ticket["type"] == "网络投诉"
        assert "处理中" in ticket["status"]
        assert "history" in ticket

    def test_get_unknown_ticket(self):
        from repositories import get_ticket_repo
        repo = get_ticket_repo()
        assert repo.get_status("TK-000000") is None

    def test_get_case_insensitive(self):
        from repositories import get_ticket_repo
        repo = get_ticket_repo()
        ticket = repo.get_status("tk-123456")
        assert ticket is not None
        assert ticket["ticket_id"] == "TK-123456"

    def test_list_by_phone(self):
        from repositories import get_ticket_repo
        repo = get_ticket_repo()
        tickets = repo.list_by_phone("13800001111")
        assert len(tickets) >= 2
        ids = {t["ticket_id"] for t in tickets}
        assert "TK-123456" in ids
        assert "TK-789012" in ids

    def test_list_by_unknown_phone(self):
        from repositories import get_ticket_repo
        repo = get_ticket_repo()
        assert repo.list_by_phone("19900000000") == []


class TestBillingRepository:
    def test_get_balance(self):
        from repositories import get_billing_repo
        repo = get_billing_repo()
        account = repo.get_balance("13800001111")
        assert account is not None
        assert account["name"] == "张三"
        assert account["balance"] == 86.50
        assert account["plan_name"] == "5G畅享套餐98元档"

    def test_get_balance_unknown(self):
        from repositories import get_billing_repo
        repo = get_billing_repo()
        assert repo.get_balance("19900000000") is None

    def test_get_monthly_bill(self):
        from repositories import get_billing_repo
        repo = get_billing_repo()
        bill = repo.get_monthly_bill("13800001111", "2026-04")
        assert bill is not None
        assert bill["plan_fee"] == 98.00
        assert "total" in bill

    def test_get_monthly_bill_unknown_month(self):
        from repositories import get_billing_repo
        repo = get_billing_repo()
        assert repo.get_monthly_bill("13800001111", "2025-01") is None

    def test_get_flow_remaining(self):
        from repositories import get_billing_repo
        repo = get_billing_repo()
        flow = repo.get_flow_remaining("13800001111")
        assert flow is not None
        assert flow["flow_total"] == 30
        assert flow["flow_remaining"] == 30 - 18.5


class TestKnowledgeRepository:
    def test_search_returns_results(self):
        from repositories import get_knowledge_repo
        repo = get_knowledge_repo()
        results = repo.search("5G套餐")
        assert len(results) > 0
        assert "content" in results[0]
        assert "category" in results[0]
        assert results[0]["score"] > 0

    def test_search_no_match(self):
        from repositories import get_knowledge_repo
        repo = get_knowledge_repo()
        results = repo.search("xyz不存在的查询内容")
        assert results == []

    def test_search_respects_top_k(self):
        from repositories import get_knowledge_repo
        repo = get_knowledge_repo()
        results = repo.search("中国移动", top_k=2)
        assert len(results) <= 2

    def test_search_scores_sorted(self):
        from repositories import get_knowledge_repo
        repo = get_knowledge_repo()
        results = repo.search("5G套餐流量")
        if len(results) >= 2:
            assert results[0]["score"] >= results[1]["score"]


class TestEscalationRepository:
    def test_escalate_returns_ticket(self):
        from repositories import get_escalation_repo
        repo = get_escalation_repo()
        result = repo.escalate("我要投诉套餐乱扣费", priority="high")
        assert "escalation_id" in result
        assert result["status"] == "已提交"
        assert result["priority"] == "high"
        assert "人工" in result["message"]
