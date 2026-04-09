#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试高级RAG系统集成
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph_agent_with_memory import knowledge_search as knowledge_search_tool, init_advanced_rag, get_memory_manager

def test_knowledge_search():
    """测试knowledge_search工具"""
    print("=== 测试knowledge_search工具 ===")

    # 获取实际的函数（不是工具对象）
    knowledge_search_func = knowledge_search_tool.func

    # 测试1：重置密码查询
    print("\n测试1：重置密码查询")
    result1 = knowledge_search_func("如何重置密码")
    print(f"结果: {result1}")

    # 测试2：产品价格查询
    print("\n测试2：产品价格查询")
    result2 = knowledge_search_func("产品价格是多少")
    print(f"结果: {result2}")

    # 测试3：技术支持查询
    print("\n测试3：技术支持查询")
    result3 = knowledge_search_func("技术支持时间")
    print(f"结果: {result3}")

    # 测试4：未知查询
    print("\n测试4：未知查询")
    result4 = knowledge_search_func("如何部署系统")
    print(f"结果: {result4}")

    # 测试记忆功能
    print("\n=== 测试记忆功能 ===")
    memory_manager = get_memory_manager()
    print(f"用户偏好: {memory_manager.user_preferences}")
    print(f"最近对话历史: {memory_manager.get_recent_history(3)}")

def test_advanced_rag_initialization():
    """测试高级RAG系统初始化"""
    print("\n=== 测试高级RAG系统初始化 ===")

    try:
        # 重新初始化高级RAG
        from langgraph_agent_with_memory import init_advanced_rag, advanced_rag_retriever
        retriever = init_advanced_rag()

        if retriever:
            print("高级RAG检索器已成功初始化")
            print(f"检索器类型: {type(retriever)}")

            # 测试检索
            print("\n测试高级RAG检索功能:")
            try:
                test_query = "如何重置密码"
                results = retriever.retrieve(test_query)
                print(f"查询: '{test_query}'")
                print(f"检索到 {len(results)} 个结果")
                if results:
                    for i, doc in enumerate(results[:2]):
                        print(f"  结果 {i+1}: {doc.page_content[:80]}...")
                        print(f"    来源: {doc.metadata.get('source', '未知')}")
                        print(f"    分数: {doc.metadata.get('final_score', 0):.3f}")
            except Exception as e:
                print(f"检索测试失败: {e}")
        else:
            print("高级RAG检索器未初始化，使用模拟模式")
    except Exception as e:
        print(f"高级RAG初始化测试失败: {e}")

def main():
    print("高级RAG系统集成测试")
    print("=" * 50)

    # 初始化内存管理器
    memory_manager = get_memory_manager()
    print(f"内存管理器已初始化")

    # 测试知识检索
    test_knowledge_search()

    # 测试高级RAG初始化
    test_advanced_rag_initialization()

    print("\n" + "=" * 50)
    print("测试完成")

if __name__ == "__main__":
    main()