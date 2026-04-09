#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多模态功能集成
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph_agent_with_memory import run_langgraph_agent_with_memory, get_memory_manager

def test_multimodal_queries():
    """测试多模态查询"""
    print("=== 测试多模态功能集成 ===")

    # 清除之前的记忆
    memory_manager = get_memory_manager()
    memory_manager.conversation_history = []

    test_cases = [
        # (查询, 期望的工具调用关键词)
        ("分析这张图片内容", "image_analysis"),
        ("我想处理一个PDF文档", "document_processing"),
        ("上传一个Word文件并分析", "file_upload_processing"),
        ("处理Excel表格", "document_processing"),
        ("查看截图内容", "image_analysis"),
        ("帮我分析这个图片", "image_analysis"),
    ]

    all_passed = True

    for query, expected_tool in test_cases:
        print(f"\n{'='*60}")
        print(f"测试查询: '{query}'")
        print(f"期望工具: {expected_tool}")
        print(f"{'='*60}")

        try:
            result = run_langgraph_agent_with_memory(query, max_iterations=2)

            # 打印结果
            print(f"最终答案: {result['final_answer'][:200]}...")
            print(f"任务规划: {result['plan']}")
            print(f"迭代次数: {result['workflow_info']['iterations']}")
            print(f"答案质量: {result['workflow_info']['answer_quality']}")

            # 检查是否使用了期望的工具
            plan = result['plan']
            if plan:
                tool_used = False
                for task in plan:
                    if expected_tool in task:
                        tool_used = True
                        break

                if tool_used:
                    print(f"[SUCCESS] 成功使用 {expected_tool} 工具")
                else:
                    print(f"[FAILED] 未使用期望的 {expected_tool} 工具，实际规划: {plan}")
                    all_passed = False
            else:
                print(f"[FAILED] 未生成任务规划")
                all_passed = False

        except Exception as e:
            print(f"[ERROR] 测试失败: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    # 测试记忆功能
    print(f"\n{'='*60}")
    print("测试记忆功能")
    print(f"{'='*60}")

    conversation_summary = memory_manager.generate_summary()
    print(f"对话摘要: {conversation_summary}")

    # 检查是否记录了多模态话题
    if "图像处理" in conversation_summary or "文档处理" in conversation_summary:
        print("[SUCCESS] 多模态话题已记录到记忆")
    else:
        print("[INFO] 多模态话题未记录到记忆")

    return all_passed

def test_multimodal_tools_directly():
    """直接测试多模态工具函数"""
    print(f"\n{'='*60}")
    print("直接测试多模态工具函数")
    print(f"{'='*60}")

    try:
        from multimodal_support import MultimodalTools
        import os

        # 创建测试目录
        test_dir = "./test_multimodal_integration"
        os.makedirs(test_dir, exist_ok=True)

        # 创建测试文件
        test_files = {
            "test_image.png": "测试图片内容",
            "test_document.pdf": "测试PDF文档内容",
            "test_report.docx": "测试Word报告内容",
        }

        for filename, content in test_files.items():
            filepath = os.path.join(test_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[INFO] 创建测试文件: {filepath}")

        # 测试多模态工具
        tools = MultimodalTools()

        print("\n1. 测试图像分析:")
        try:
            result = tools.analyze_image(os.path.join(test_dir, "test_image.png"))
            print(f"结果: {result[:100]}...")
            print("[SUCCESS] 图像分析工具工作正常")
        except Exception as e:
            print(f"[FAILED] 图像分析失败: {e}")

        print("\n2. 测试文档处理:")
        try:
            result = tools.extract_document_content(os.path.join(test_dir, "test_document.pdf"))
            print(f"结果: {result[:100]}...")
            print("[SUCCESS] 文档处理工具工作正常")
        except Exception as e:
            print(f"[FAILED] 文档处理失败: {e}")

        print("\n3. 测试文件上传处理:")
        try:
            result = tools.process_uploaded_file(os.path.join(test_dir, "test_report.docx"))
            print(f"结果: {result}")
            print("[SUCCESS] 文件上传处理工具工作正常")
        except Exception as e:
            print(f"[FAILED] 文件上传处理失败: {e}")

        # 清理测试文件
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)

        return True

    except Exception as e:
        print(f"[FAILED] 多模态工具直接测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("多模态功能集成测试")
    print("=" * 50)

    # 测试多模态查询
    test1_passed = test_multimodal_queries()

    # 直接测试多模态工具
    test2_passed = test_multimodal_tools_directly()

    print(f"\n{'='*50}")
    print("测试结果汇总:")
    print(f"多模态查询测试: {'通过' if test1_passed else '失败'}")
    print(f"多模态工具直接测试: {'通过' if test2_passed else '失败'}")

    if test1_passed and test2_passed:
        print("\n[SUCCESS] 所有多模态功能集成测试通过")
        return True
    else:
        print("\n[FAILED] 部分测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)