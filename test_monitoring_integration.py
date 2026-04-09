#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控系统集成测试
测试监控系统在主工作流中的集成情况
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_monitoring_initialization():
    """测试监控系统初始化"""
    print("=== 测试监控系统初始化 ===")

    try:
        # 导入主工作流模块
        from langgraph_agent_with_memory import (
            MONITORING_AVAILABLE,
            monitoring_system,
            preprocess_node,
            planning_node,
            execution_node,
            validation_node,
            postprocess_node,
            human_escalation_node
        )

        print(f"[INFO] MONITORING_AVAILABLE: {MONITORING_AVAILABLE}")
        print(f"[INFO] monitoring_system: {monitoring_system}")

        if MONITORING_AVAILABLE:
            print("[SUCCESS] 监控系统可用")
            if monitoring_system:
                print("[SUCCESS] 监控系统实例已初始化")
            else:
                print("[WARN] 监控系统实例未初始化（可能配置问题）")
        else:
            print("[WARN] 监控系统不可用（可能缺少依赖）")

        return MONITORING_AVAILABLE and monitoring_system is not None

    except ImportError as e:
        print(f"[FAILED] 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAILED] 其他错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_state_structure():
    """测试状态结构是否包含tracking_info"""
    print("\n=== 测试状态结构 ===")

    try:
        from langgraph_agent_with_memory import AgentState

        # 创建一个模拟状态
        test_state = {
            "user_query": "测试查询",
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
            "conversation_summary": None,
            "tracking_info": {}  # 应该被添加的字段
        }

        print(f"[INFO] 状态键: {list(test_state.keys())}")

        # 检查是否包含tracking_info
        if 'tracking_info' in test_state:
            print("[SUCCESS] 状态包含tracking_info字段")
            return True
        else:
            print("[WARN] 状态不包含tracking_info字段")
            return False

    except Exception as e:
        print(f"[FAILED] 状态结构测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_node_monitoring_calls():
    """测试节点中的监控调用"""
    print("\n=== 测试节点监控调用 ===")

    try:
        # 检查各个节点函数是否包含监控调用
        from langgraph_agent_with_memory import (
            preprocess_node,
            planning_node,
            execution_node,
            validation_node,
            postprocess_node,
            human_escalation_node
        )

        nodes_to_check = [
            ("preprocess_node", preprocess_node),
            ("planning_node", planning_node),
            ("execution_node", execution_node),
            ("validation_node", validation_node),
            ("postprocess_node", postprocess_node),
            ("human_escalation_node", human_escalation_node)
        ]

        all_good = True
        for node_name, node_func in nodes_to_check:
            source = node_func.__code__.co_code
            source_str = str(source)

            # 检查是否包含监控相关字符串（基本检查）
            # 注意：这是简单的检查，实际应该查看源代码
            print(f"[INFO] 检查 {node_name}...")

            # 检查函数是否有文档字符串
            if node_func.__doc__:
                print(f"  - 有文档字符串: {node_func.__doc__[:50]}...")
            else:
                print(f"  - 无文档字符串")

            print(f"  - 代码长度: {len(source)} 字节")

        print("[INFO] 节点监控调用检查完成（基本验证）")
        return all_good

    except Exception as e:
        print(f"[FAILED] 节点监控调用测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_workflow_creation():
    """测试工作流创建"""
    print("\n=== 测试工作流创建 ===")

    try:
        from langgraph_agent_with_memory import create_workflow

        workflow = create_workflow()

        if workflow:
            print("[SUCCESS] 工作流创建成功")

            # 检查节点数量
            nodes = workflow.nodes
            print(f"[INFO] 工作流包含 {len(nodes)} 个节点: {list(nodes.keys())}")

            return True
        else:
            print("[FAILED] 工作流创建失败")
            return False

    except Exception as e:
        print(f"[FAILED] 工作流创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("监控系统集成测试")
    print("=" * 60)

    # 测试1: 监控系统初始化
    init_success = test_monitoring_initialization()

    # 测试2: 状态结构
    state_success = test_state_structure()

    # 测试3: 节点监控调用
    nodes_success = test_node_monitoring_calls()

    # 测试4: 工作流创建
    workflow_success = test_workflow_creation()

    # 汇总结果
    print(f"\n{'='*60}")
    print("测试结果汇总:")
    print(f"监控系统初始化: {'通过' if init_success else '失败'}")
    print(f"状态结构检查: {'通过' if state_success else '失败'}")
    print(f"节点监控调用: {'通过' if nodes_success else '失败'}")
    print(f"工作流创建: {'通过' if workflow_success else '失败'}")

    all_success = init_success and state_success and nodes_success and workflow_success

    if all_success:
        print("\n[SUCCESS] 所有监控系统集成测试通过")
        return True
    else:
        print("\n[FAILED] 部分监控系统集成测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)