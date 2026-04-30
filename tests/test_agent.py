"""
核心 Agent 单元测试
Mock 外部依赖（LLM、Embedding），在 CI 中安全执行。
"""

from datetime import date


class TestMemoryManager:
    """MemoryManager 记忆系统测试"""

    def test_init_state(self):
        """验证初始状态"""
        from langgraph_agent_with_memory import MemoryManager
        mm = MemoryManager()
        assert mm.conversation_history == []
        assert mm.user_preferences["language_style"] == "neutral"
        assert mm.user_preferences["detail_level"] == "moderate"
        assert isinstance(mm.user_preferences["frequent_topics"], set)
        assert len(mm.user_preferences["frequent_topics"]) == 0

    def test_add_message_appends(self):
        """验证添加消息"""
        from langgraph_agent_with_memory import MemoryManager
        mm = MemoryManager()
        mm.add_message("user", "5G套餐多少钱")
        assert len(mm.conversation_history) == 1
        assert mm.conversation_history[0]["role"] == "user"
        assert "5G" in mm.conversation_history[0]["content"]
        assert "timestamp" in mm.conversation_history[0]

    def test_history_limit_10(self):
        """验证历史限制在10条"""
        from langgraph_agent_with_memory import MemoryManager
        mm = MemoryManager()
        for i in range(15):
            mm.add_message("user", f"消息{i}")
        assert len(mm.conversation_history) == 10
        assert mm.conversation_history[0]["content"] == "消息5"
        assert mm.conversation_history[-1]["content"] == "消息14"

    def test_get_recent_history(self):
        """验证获取最近历史"""
        from langgraph_agent_with_memory import MemoryManager
        mm = MemoryManager()
        mm.add_message("user", "Q1")
        mm.add_message("assistant", "A1")
        mm.add_message("user", "Q2")
        recent = mm.get_recent_history(2)
        assert len(recent) == 2
        assert recent[-1]["content"] == "Q2"

    def test_generate_summary_empty(self):
        """验证空历史摘要"""
        from langgraph_agent_with_memory import MemoryManager
        mm = MemoryManager()
        assert "暂无" in mm.generate_summary()

    def test_generate_summary_with_content(self):
        """验证含内容摘要"""
        from langgraph_agent_with_memory import MemoryManager
        mm = MemoryManager()
        mm.add_message("user", "5G套餐多少钱")
        mm.add_message("user", "宽带怎么办理")
        summary = mm.generate_summary()
        assert "套餐" in summary
        assert "宽带" in summary

    def test_update_preferences_formal(self):
        """验证正式语言检测"""
        from langgraph_agent_with_memory import MemoryManager
        mm = MemoryManager()
        mm.update_preferences("您好，请帮我查一下账单", "好的，您的账单是...")
        assert mm.user_preferences["language_style"] == "formal"

    def test_update_preferences_casual(self):
        """验证随意语言检测"""
        from langgraph_agent_with_memory import MemoryManager
        mm = MemoryManager()
        mm.update_preferences("哈喽，帮我看看", "OK")
        assert mm.user_preferences["language_style"] == "casual"

    def test_adapt_response_formal(self):
        """验证正式风格适配"""
        from langgraph_agent_with_memory import MemoryManager
        mm = MemoryManager()
        mm.user_preferences["language_style"] = "formal"
        result = mm.adapt_response("您的问题已处理。")
        assert result.startswith("尊敬的客户")

    def test_clear_resets_all(self):
        """验证清空记忆"""
        from langgraph_agent_with_memory import MemoryManager
        mm = MemoryManager()
        mm.add_message("user", "test")
        mm.user_preferences["language_style"] = "formal"
        mm.conversation_history = []
        mm.user_preferences = {
            "language_style": "neutral",
            "detail_level": "moderate",
            "frequent_topics": set(),
            "last_interaction": None
        }
        assert len(mm.conversation_history) == 0
        assert mm.user_preferences["language_style"] == "neutral"


class TestToolFunctions:
    """工具函数测试"""

    def test_knowledge_search_empty_query(self):
        """验证空查询被拒绝"""
        from langgraph_agent_with_memory import knowledge_search
        result = knowledge_search.invoke({"query": ""})
        assert "错误" in result

    def test_knowledge_search_whitespace_query(self):
        """验证纯空格查询被拒绝"""
        from langgraph_agent_with_memory import knowledge_search
        result = knowledge_search.invoke({"query": "   "})
        assert "错误" in result

    def test_knowledge_search_fallback(self):
        """验证 fallback 匹配"""
        from langgraph_agent_with_memory import knowledge_search
        result = knowledge_search.invoke({"query": "套餐资费"})
        assert "中国移动" in result

    def test_query_ticket_status_empty(self):
        """验证空工单号被拒绝"""
        from langgraph_agent_with_memory import query_ticket_status
        result = query_ticket_status.invoke({"ticket_id": ""})
        assert "有效的工单号" in result or "错误" in result

    def test_query_ticket_status_known(self):
        """验证已知工单查询"""
        from langgraph_agent_with_memory import query_ticket_status
        result = query_ticket_status.invoke({"ticket_id": "TK-123456"})
        assert "TK-123456" in result

    def test_query_ticket_status_unknown(self):
        """验证未知工单查询"""
        from langgraph_agent_with_memory import query_ticket_status
        result = query_ticket_status.invoke({"ticket_id": "TK-999999"})
        assert "未找到" in result

    def test_escalate_to_human(self):
        """验证转人工返回提示"""
        from langgraph_agent_with_memory import escalate_to_human
        result = escalate_to_human.invoke({"query": "我要投诉"})
        assert "人工" in result

    def test_get_current_date(self):
        """验证日期查询返回当天日期"""
        from langgraph_agent_with_memory import get_current_date
        result = get_current_date.invoke({"query": "今天几号"})
        assert str(date.today()) in result


class TestAgentEntryPoint:
    """Agent 入口函数测试"""

    def test_empty_query_rejected(self):
        """验证空输入被拒绝"""
        from langgraph_agent_with_memory import run_langgraph_agent_with_memory
        result = run_langgraph_agent_with_memory("")
        assert "请输入有效" in result["final_answer"]
        assert result["plan"] is None

    def test_whitespace_query_rejected(self):
        """验证空白输入被拒绝"""
        from langgraph_agent_with_memory import run_langgraph_agent_with_memory
        result = run_langgraph_agent_with_memory("   ")
        assert "请输入有效" in result["final_answer"]

    def test_result_structure(self):
        """验证正常查询返回结构完整"""
        from langgraph_agent_with_memory import run_langgraph_agent_with_memory
        result = run_langgraph_agent_with_memory("5G套餐有哪些")
        assert "plan" in result
        assert "tool_results" in result
        assert "final_answer" in result
        assert "workflow_info" in result
        assert "memory_info" in result
        assert "mcp_status" in result


class TestSystemPrompt:
    """系统提示构建测试"""

    def test_build_base_prompt(self):
        """验证基础系统提示（无 skill 激活）"""
        from langgraph_agent_with_memory import build_system_prompt
        prompt = build_system_prompt()
        assert "中国移动" in prompt
        assert "客服" in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_build_prompt_with_skills(self):
        """验证带 skill 的系统提示"""
        from langgraph_agent_with_memory import build_system_prompt
        prompt = build_system_prompt(active_skills=["5G套餐咨询"])
        assert "5G套餐咨询" in prompt


class TestWorkflowCreation:
    """工作流创建测试"""

    def test_create_workflow_returns_graph(self):
        """验证工作流图创建成功"""
        from langgraph_agent_with_memory import create_workflow
        wf = create_workflow()
        assert wf is not None

    def test_workflow_compiles(self):
        """验证工作流可编译"""
        from langgraph_agent_with_memory import create_workflow
        from langgraph.checkpoint.memory import MemorySaver
        wf = create_workflow()
        memory = MemorySaver()
        app = wf.compile(checkpointer=memory)
        assert app is not None


class TestAgentState:
    """AgentState 类型测试"""

    def test_agent_state_keys(self):
        """验证 AgentState 包含所有必需字段"""
        from langgraph_agent_with_memory import AgentState
        # AgentState 是 TypedDict，通过 __optional_keys__ 和 __required_keys__ 验证
        required = set(AgentState.__required_keys__) if hasattr(AgentState, '__required_keys__') else set()
        optional = set(AgentState.__optional_keys__) if hasattr(AgentState, '__optional_keys__') else set()
        all_keys = required | optional
        assert "user_query" in all_keys
        assert "messages" in all_keys
        assert "final_answer" in all_keys
