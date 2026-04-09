#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试优化后的LangGraph Agent功能
"""

import sys
import io

# 设置Windows控制台编码
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def test_langgraph_agent():
    """测试带记忆的LangGraph Agent"""
    print("=== 测试优化后的LangGraph Agent ===")

    try:
        from langgraph_agent_with_memory import run_langgraph_agent_with_memory
        from langgraph_agent_with_memory import clear_memory, get_conversation_history

        # 清空记忆
        clear_memory()

        # 测试1: 基础查询
        print("\n1. 测试基础查询: 如何重置密码？")
        result1 = run_langgraph_agent_with_memory("如何重置密码？")
        print(f"   回答: {result1['final_answer'][:60]}...")
        print(f"   工具使用: {list(result1['tool_results'].keys()) if result1['tool_results'] else '无'}")

        # 测试2: 记忆功能
        print("\n2. 测试记忆功能: 再问一下密码问题")
        result2 = run_langgraph_agent_with_memory("再问一下密码问题")
        print(f"   回答: {result2['final_answer'][:80]}...")
        print(f"   记忆检测: {'有上下文提示' if '注意' in result2['final_answer'] else '无上下文提示'}")

        # 测试3: 天气查询工具
        print("\n3. 测试天气查询工具: 北京天气怎么样？")
        result3 = run_langgraph_agent_with_memory("北京天气怎么样？")
        print(f"   回答: {result3['final_answer']}")
        print(f"   工具使用: weather_query (检测: {'weather_query' in str(result3['tool_results'])})")

        # 测试4: 转人工功能
        print("\n4. 测试转人工功能: 帮我转人工客服")
        result4 = run_langgraph_agent_with_memory("帮我转人工客服")
        print(f"   回答: {result4['final_answer'][:60]}...")
        print(f"   工作流转: {result4['workflow_info']['path'] if 'path' in result4['workflow_info'] else '未知'}")

        # 显示记忆状态
        history = get_conversation_history()
        print(f"\n=== 测试完成 ===")
        print(f"对话历史: {len(history)} 条消息")
        print(f"用户偏好: {result4['memory_info']['user_preferences']}")

        return True

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_langgraph_agent()
    sys.exit(0 if success else 1)