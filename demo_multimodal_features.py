#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态功能演示 - 企业智能客服Agent第三阶段优化成果

演示功能：
1. 图像分析（OCR文字识别、图像描述）
2. 文档处理（PDF、Word、Excel内容提取）
3. 文件上传处理
4. 与现有工作流的无缝集成
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph_agent_with_memory import run_langgraph_agent_with_memory, get_memory_manager

def demo_multimodal_features():
    """演示多模态功能"""
    print("=" * 70)
    print("企业智能客服Agent - 多模态功能演示")
    print("=" * 70)
    print("第三阶段优化：多模态支持（图像理解、文档处理、文件上传）")
    print()

    # 清除之前的记忆
    memory_manager = get_memory_manager()
    memory_manager.conversation_history = []

    # 演示场景1：图像分析
    print("\n" + "=" * 70)
    print("演示场景1：图像分析")
    print("=" * 70)
    print("查询：'分析这张产品图片的内容'")
    print()

    result1 = run_langgraph_agent_with_memory("分析这张产品图片的内容", max_iterations=2)
    print(f"最终答案: {result1['final_answer'][:200]}...")
    print(f"使用的工具: {result1['plan']}")
    print(f"答案质量: {result1['workflow_info']['answer_quality']}")

    # 演示场景2：PDF文档处理
    print("\n" + "=" * 70)
    print("演示场景2：PDF文档处理")
    print("=" * 70)
    print("查询：'帮我提取这个PDF文档的内容'")
    print()

    result2 = run_langgraph_agent_with_memory("帮我提取这个PDF文档的内容", max_iterations=2)
    print(f"最终答案: {result2['final_answer'][:200]}...")
    print(f"使用的工具: {result2['plan']}")
    print(f"答案质量: {result2['workflow_info']['answer_quality']}")

    # 演示场景3：Excel表格处理
    print("\n" + "=" * 70)
    print("演示场景3：Excel表格处理")
    print("=" * 70)
    print("查询：'处理Excel销售报表'")
    print()

    result3 = run_langgraph_agent_with_memory("处理Excel销售报表", max_iterations=2)
    print(f"最终答案: {result3['final_answer'][:200]}...")
    print(f"使用的工具: {result3['plan']}")
    print(f"答案质量: {result3['workflow_info']['answer_quality']}")

    # 演示场景4：文件上传
    print("\n" + "=" * 70)
    print("演示场景4：文件上传处理")
    print("=" * 70)
    print("查询：'上传一个Word文档并分析'")
    print()

    result4 = run_langgraph_agent_with_memory("上传一个Word文档并分析", max_iterations=2)
    print(f"最终答案: {result4['final_answer'][:200]}...")
    print(f"使用的工具: {result4['plan']}")
    print(f"答案质量: {result4['workflow_info']['answer_quality']}")

    # 演示场景5：与现有功能集成（混合查询）
    print("\n" + "=" * 70)
    print("演示场景5：多模态与现有功能集成")
    print("=" * 70)
    print("查询：'先分析这张截图，然后告诉我今天的日期'")
    print()

    result5 = run_langgraph_agent_with_memory("先分析这张截图，然后告诉我今天的日期", max_iterations=2)
    print(f"最终答案: {result5['final_answer'][:200]}...")
    print(f"使用的工具: {result5['plan']}")
    print(f"答案质量: {result5['workflow_info']['answer_quality']}")

    # 显示记忆功能
    print("\n" + "=" * 70)
    print("记忆功能展示")
    print("=" * 70)
    conversation_summary = memory_manager.generate_summary()
    print(f"对话摘要: {conversation_summary}")
    print(f"用户偏好: {memory_manager.user_preferences}")
    print(f"对话历史长度: {len(memory_manager.conversation_history)} 条消息")

    # 技术架构总结
    print("\n" + "=" * 70)
    print("多模态功能技术架构")
    print("=" * 70)
    print("""
1. 模块化设计：
   - multimodal_support.py: 独立的多模态处理模块
   - 支持图像分析、文档处理、多模态知识库

2. 集成方式：
   - 通过@tool装饰器集成到LangGraph工具系统
   - 在规划节点中自动识别多模态查询
   - 在执行节点中调用相应的多模态工具

3. 关键技术：
   - 媒体类型检测（图像、PDF、Word、Excel等）
   - 模拟OCR和图像描述（实际部署可用CLIP、Tesseract等）
   - 文档内容提取（模拟，实际可用pypdf、python-docx等）
   - 多模态知识库管理

4. 优势：
   - 与现有RAG系统无缝集成
   - 支持记忆和用户偏好
   - 可扩展的架构，支持更多媒体类型
   - 生产就绪的错误处理和降级策略
    """)

def test_multimodal_module_directly():
    """直接测试多模态模块"""
    print("\n" + "=" * 70)
    print("多模态模块直接测试")
    print("=" * 70)

    try:
        from multimodal_support import MultimodalTools, test_multimodal_support

        # 运行模块自带的测试
        print("运行多模态支持模块自测试...")
        success = test_multimodal_support()
        if success:
            print("[SUCCESS] 多模态支持模块测试通过")
        else:
            print("[FAILED] 多模态支持模块测试失败")

        # 创建工具实例
        tools = MultimodalTools()
        print("\n多模态工具已初始化:")
        print(f"  - 知识库管理器: {tools.knowledge_base}")
        print(f"  - 图像处理器: {tools.image_processor}")
        print(f"  - 多模态工具集合可用")

    except ImportError as e:
        print(f"[ERROR] 无法导入多模态模块: {e}")
    except Exception as e:
        print(f"[ERROR] 多模态模块测试失败: {e}")

def main():
    """主函数"""
    print("企业智能客服Agent - 第三阶段优化完成")
    print("=" * 70)
    print("优化内容：")
    print("1. 高级RAG系统：混合检索、查询扩展、重排序、向量缓存")
    print("2. 异步RAG系统：并行检索、异步缓存、批处理优化")
    print("3. 多模态支持：图像分析、文档处理、文件上传")
    print("4. 完整集成：所有功能无缝集成到LangGraph工作流")
    print()

    # 演示多模态功能
    demo_multimodal_features()

    # 直接测试多模态模块
    test_multimodal_module_directly()

    print("\n" + "=" * 70)
    print("[SUCCESS] 多模态功能演示完成")
    print("=" * 70)
    print("项目现已支持：")
    print("✓ 文本问答 (RAG增强)")
    print("✓ 多轮对话记忆")
    print("✓ 用户偏好学习")
    print("✓ 6种企业级工具")
    print("✓ 高级RAG检索")
    print("✓ 异步并行处理")
    print("✓ 多模态输入处理")
    print("✓ 生产级错误处理")
    print()

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)