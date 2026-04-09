#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试完整Agent工作流
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph_agent_with_memory import create_workflow, get_memory_manager

def test_simple_query():
    """测试简单查询"""
    print("=== 测试完整Agent工作流 ===")

    # 创建工作流
    workflow = create_workflow()

    # 测试查询
    test_queries = [
        "如何重置密码",
        "今天日期是什么",
        "查询北京的天气",
        "查询苹果股票"
    ]

    for query in test_queries:
        print(f"\n查询: '{query}'")
        try:
            # 准备初始状态
            initial_state = {
                "user_query": query,
                "messages": [],
                "user_preferences": {},
                "plan": None,
                "tool_results": None,
                "final_answer": None,
                "step": "planning",
                "iteration": 0,
                "max_iterations": 3,
                "needs_human_escalation": False,
                "answer_quality": None,
                "conversation_summary": None
            }

            # 执行工作流
            result = workflow.invoke(initial_state)

            print(f"最终答案: {result.get('final_answer', '无')[:100]}...")
            print(f"步骤: {result.get('step')}")
            print(f"迭代次数: {result.get('iteration')}")

        except Exception as e:
            print(f"查询执行失败: {e}")

def main():
    print("完整Agent工作流测试")
    print("=" * 50)

    test_simple_query()

    print("\n" + "=" * 50)
    print("测试完成")

if __name__ == "__main__":
    main()