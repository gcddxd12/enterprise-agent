import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from dotenv import load_dotenv
from enterprise_agent import (
    knowledge_search,
    query_ticket_status,
    get_current_date,
    planning_agent,
    execution_agent,
    validation_agent
)

load_dotenv()

# 判断是否在 CI 环境（GitHub Actions 会自动设置 CI=true）
CI = os.getenv("CI") is not None

@pytest.mark.skipif(CI, reason="Skipping test that requires API in CI")
def test_knowledge_search():
    result = knowledge_search.run("如何重置密码")
    assert isinstance(result, str)
    assert len(result) > 10
    assert "密码" in result or "重置" in result

def test_ticket_query_found():
    result = query_ticket_status.run("TK-123456")
    assert "处理中" in result or "受理" in result
    assert "TK-123456" in result

def test_ticket_query_not_found():
    result = query_ticket_status.run("TK-999999")
    assert "未找到" in result

def test_date_query():
    result = get_current_date.run("")
    import re
    assert re.match(r'\d{4}-\d{2}-\d{2}', result) is not None

def test_planning_agent():
    tasks = planning_agent("查询工单 TK-123456")
    assert isinstance(tasks, list)
    assert any("ticket_query" in task for task in tasks)
    tasks = planning_agent("今天几号")
    assert any("date_query" in task for task in tasks)
    tasks = planning_agent("如何重置密码")
    assert any("knowledge_search" in task for task in tasks)

def test_execution_agent():
    tasks = ["ticket_query: TK-123456"]
    results = execution_agent(tasks)
    assert "ticket_query: TK-123456" in results
    assert "处理中" in results["ticket_query: TK-123456"]

def test_validation_agent():
    answer = "您的工单 TK-123456 已受理，正在处理中"
    validated = validation_agent("查询工单", answer)
    assert "受理" in validated
    answer = "2025-03-21"
    validated = validation_agent("今天几号", answer)
    assert "2025-03-21" in validated
    answer = "不知道"
    validated = validation_agent("测试", answer)
    assert "无法确定" in validated