#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试异步RAG系统集成
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_async_rag_with_vectorstore():
    """测试带向量数据库的异步RAG"""
    print("=== 测试异步RAG与向量数据库集成 ===")

    try:
        from langchain_community.embeddings import DashScopeEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document
        from async_rag_system import create_async_advanced_rag_system

        # 检查向量数据库是否存在
        chroma_db_path = "./chroma_db"
        if not os.path.exists(chroma_db_path):
            print("[WARN] 向量数据库不存在，跳过测试")
            return False

        # 加载向量数据库
        print("[INFO] 加载向量数据库...")
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("DASHSCOPE_API_KEY")

        embeddings = DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=api_key
        )
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=chroma_db_path
        )

        # 创建异步RAG系统
        print("[INFO] 创建异步RAG系统...")
        retriever = await create_async_advanced_rag_system(
            vectorstore=vectorstore,
            use_mock=True,
            config={
                "use_cache": True,
                "query_expansion": True,
                "vector_k": 5
            }
        )

        print(f"[INFO] 异步RAG检索器已初始化: {type(retriever)}")

        # 测试异步检索
        test_queries = [
            "如何重置密码",
            "产品价格",
            "技术支持时间",
            "API调用频率限制"
        ]

        for query in test_queries:
            print(f"\n查询: '{query}'")
            start_time = asyncio.get_event_loop().time()

            results = await retriever.retrieve(query, k=3)

            end_time = asyncio.get_event_loop().time()
            elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒

            print(f"  检索时间: {elapsed_time:.1f}ms")
            print(f"  检索到 {len(results)} 个结果")

            if results:
                for i, doc in enumerate(results[:2]):
                    print(f"  结果 {i+1}: {doc.page_content[:80]}...")
                    print(f"    来源: {doc.metadata.get('source', '未知')}")
                    print(f"    检索器: {doc.metadata.get('retriever', '未知')}")
                    print(f"    分数: {doc.metadata.get('score', 0):.3f}")
            else:
                print("  未找到相关结果")

        print("\n[SUCCESS] 异步RAG与向量数据库集成测试完成")
        return True

    except Exception as e:
        print(f"[FAILED] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_concurrent_queries():
    """测试并发查询"""
    print("\n=== 测试并发查询性能 ===")

    try:
        from async_rag_system import create_async_advanced_rag_system
        from langchain_core.documents import Document

        # 创建模拟文档
        test_documents = [
            Document(page_content="如何重置密码？您可以通过登录页面点击'忘记密码'链接重置密码。"),
            Document(page_content="产品价格信息：企业版每年10,000元，包含技术支持。"),
            Document(page_content="技术支持时间：工作日9:00-18:00，电话400-123-4567。"),
            Document(page_content="API调用频率限制：标准版每分钟100次，每日10,000次。"),
            Document(page_content="数据安全措施：采用TLS 1.3加密传输，AES-256加密存储。"),
        ]

        # 创建异步RAG系统
        retriever = await create_async_advanced_rag_system(
            documents=test_documents,
            use_mock=True,
            config={
                "use_cache": True,
                "query_expansion": True,
                "bm25_k": 5
            }
        )

        # 并发查询
        queries = ["密码重置", "产品价格", "技术支持", "API限制", "数据安全"]

        print(f"并发执行 {len(queries)} 个查询...")
        start_time = asyncio.get_event_loop().time()

        tasks = [retriever.retrieve(query, k=2) for query in queries]
        results = await asyncio.gather(*tasks)

        end_time = asyncio.get_event_loop().time()
        total_time = (end_time - start_time) * 1000

        print(f"总执行时间: {total_time:.1f}ms")
        print(f"平均查询时间: {total_time/len(queries):.1f}ms")

        for i, (query, docs) in enumerate(zip(queries, results)):
            print(f"\n 查询 {i+1}: '{query}'")
            print(f"  结果数: {len(docs)}")

        print("\n[SUCCESS] 并发查询测试完成")
        return True

    except Exception as e:
        print(f"[FAILED] 并发查询测试失败: {e}")
        return False

async def main():
    print("异步RAG系统集成测试")
    print("=" * 50)

    # 测试异步RAG与向量数据库
    await test_async_rag_with_vectorstore()

    # 测试并发查询
    await test_concurrent_queries()

    print("\n" + "=" * 50)
    print("所有测试完成")

if __name__ == "__main__":
    asyncio.run(main())