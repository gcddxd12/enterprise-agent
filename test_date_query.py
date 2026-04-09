#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试日期查询功能
"""

import sys
import io

# 设置Windows控制台编码
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def test_date_query():
    print("=== 测试日期查询功能 ===")

    try:
        from langgraph_agent_with_memory import run_langgraph_agent_with_memory
        from langgraph_agent_with_memory import clear_memory

        # 清空记忆
        clear_memory()

        # 测试日期查询
        query = "今天日期是什么"
        print(f"测试查询: {query}")

        result = run_langgraph_agent_with_memory(query)

        print(f"\n=== 完整结果 ===")
        print(f"final_answer: {repr(result.get('final_answer'))}")
        print(f"final_answer type: {type(result.get('final_answer'))}")
        print(f"final_answer is None: {result.get('final_answer') is None}")
        print(f"final_answer empty string: {result.get('final_answer') == ''}")

        print(f"\n=== 详细结果 ===")
        print(f"plan: {result.get('plan')}")
        print(f"tool_results: {result.get('tool_results')}")
        print(f"workflow_info: {result.get('workflow_info')}")
        print(f"memory_info: {result.get('memory_info')}")

        # 检查工具结果
        if result.get('tool_results'):
            for task, res in result['tool_results'].items():
                print(f"  {task}: {repr(res)}")

        return result.get('final_answer') is not None and result.get('final_answer') != ''

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_date_query()
    print(f"\n测试结果: {'成功' if success else '失败'}")
    sys.exit(0 if success else 1)