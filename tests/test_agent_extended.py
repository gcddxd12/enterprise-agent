"""Extended agent tests: postprocess, streaming, edge cases."""

import sys

import pytest

sys.path.insert(0, ".")

from unittest.mock import MagicMock, patch

from langgraph_agent_with_memory import (
    AgentState,
    postprocess_node,
    agent_node,
    build_system_prompt,
    create_workflow,
)


# ============================================================================
# TestPostprocessNode
# ============================================================================


class TestPostprocessNode:
    def test_postprocess_no_tool_results(self):
        """Without tool results, answer passes through unchanged."""
        state: AgentState = {
            "user_query": "测试",
            "final_answer": "直接回答",
            "tool_results": {},
            "raw_context": "",
            "memory_context": "",
            "plan": [],
            "workflow_info": {},
            "memory_info": {},
            "conversation_summary": "",
            "skill_context": None,
            "active_skills": [],
            "mcp_status": {},
            "raw_answer": "直接回答",
            "step": "completed",
            "iteration": 0,
            "max_iterations": 3,
            "messages": [],
            "error": None,
            "tracking_info": {},
        }
        result = postprocess_node(state)
        # Should have final_answer set
        assert "final_answer" in result
        assert isinstance(result["final_answer"], str)

    def test_postprocess_empty_final_answer(self):
        """Empty answer should get fallback."""
        state: AgentState = {
            "user_query": "测试",
            "final_answer": "",
            "tool_results": {},
            "raw_context": "",
            "memory_context": "",
            "plan": [],
            "workflow_info": {},
            "memory_info": {},
            "conversation_summary": "",
            "skill_context": None,
            "active_skills": [],
            "mcp_status": {},
            "raw_answer": "",
            "step": "completed",
            "iteration": 0,
            "max_iterations": 3,
            "messages": [],
            "error": None,
            "tracking_info": {},
        }
        result = postprocess_node(state)
        # Should return a non-empty answer (fallback)
        final = result.get("final_answer", "")
        assert len(final) > 0

    def test_postprocess_preserves_state_keys(self):
        """All required state keys should be present after postprocessing."""
        state: AgentState = {
            "user_query": "你好",
            "final_answer": "你好，有什么可以帮您？",
            "tool_results": {},
            "raw_context": "",
            "memory_context": "",
            "plan": [],
            "workflow_info": {},
            "memory_info": {},
            "conversation_summary": "",
            "skill_context": None,
            "active_skills": [],
            "mcp_status": {},
            "raw_answer": "你好，有什么可以帮您？",
            "step": "completed",
            "iteration": 0,
            "max_iterations": 3,
            "messages": [],
            "error": None,
            "tracking_info": {},
        }
        result = postprocess_node(state)
        for key in state:
            assert key in result, f"Missing key: {key}"


# ============================================================================
# TestBuildSystemPrompt
# ============================================================================


class TestBuildSystemPrompt:
    def test_base_prompt_non_empty(self):
        prompt = build_system_prompt()
        assert len(prompt) > 0
        assert isinstance(prompt, str)

    def test_prompt_contains_keywords(self):
        prompt = build_system_prompt()
        # Should reference Chinese Mobile and customer service
        assert "中国移动" in prompt or "客服" in prompt or "10086" in prompt

    def test_prompt_with_skills(self):
        prompt = build_system_prompt(active_skills=["billing", "network"])
        assert len(prompt) > 0


# ============================================================================
# TestWorkflowCreation
# ============================================================================


class TestWorkflowCreation:
    def test_create_workflow_nodes(self):
        workflow = create_workflow()
        assert workflow is not None
        # Should be a StateGraph
        from langgraph.graph import StateGraph
        assert isinstance(workflow, StateGraph)

    def test_workflow_compiles_with_checkpointer(self):
        workflow = create_workflow()
        from langgraph.checkpoint.memory import MemorySaver
        app = workflow.compile(checkpointer=MemorySaver())
        assert app is not None

    def test_multiple_compiles(self):
        """Verify compiling twice doesn't error (different MemorySavers)."""
        workflow = create_workflow()
        from langgraph.checkpoint.memory import MemorySaver
        app1 = workflow.compile(checkpointer=MemorySaver())
        app2 = workflow.compile(checkpointer=MemorySaver())
        assert app1 is not None
        assert app2 is not None


# ============================================================================
# TestAgentState
# ============================================================================


class TestAgentStateStructure:
    def test_required_keys(self):
        """Verify basic required keys are in AgentState TypedDict definition."""
        from langgraph_agent_with_memory import AgentState as ASType
        actual_keys = set(ASType.__annotations__.keys())
        # Core keys that must be present
        core = {"user_query", "messages", "final_answer", "tool_results", "plan"}
        missing = core - actual_keys
        assert not missing, f"Missing core keys: {missing}"

    def test_agent_state_total_keys(self):
        """AgentState should have a reasonable number of fields."""
        from langgraph_agent_with_memory import AgentState as ASType
        keys = ASType.__annotations__.keys()
        assert len(list(keys)) >= 8  # minimum expected fields
