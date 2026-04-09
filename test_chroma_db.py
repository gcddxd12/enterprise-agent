#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Chroma向量数据库
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_chroma_db():
    """测试Chroma向量数据库"""
    print("=== 测试Chroma向量数据库 ===")

    try:
        from langchain_community.embeddings import DashScopeEmbeddings
        from langchain_community.vectorstores import Chroma

        chroma_db_path = "./chroma_db"
        if not os.path.exists(chroma_db_path):
            print("[ERROR] Chroma数据库目录不存在")
            return False

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            print("[WARN] 未找到DASHSCOPE_API_KEY环境变量")
            api_key = "dummy_key"

        print(f"[INFO] 加载向量数据库: {chroma_db_path}")

        embeddings = DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=api_key
        )

        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=chroma_db_path
        )

        # 获取集合信息
        collection = vectorstore._collection
        if collection:
            print(f"[INFO] 集合名称: {collection.name}")

            # 统计文档数量
            count = collection.count()
            print(f"[INFO] 文档数量: {count}")

            # 获取一些文档示例
            if count > 0:
                results = collection.get(limit=min(3, count))
                if results and 'documents' in results:
                    print(f"[INFO] 文档示例 ({len(results['documents'])} 个):")
                    for i, doc in enumerate(results['documents'][:3]):
                        print(f"  文档 {i+1}: {doc[:100]}...")
                        if 'metadatas' in results and i < len(results['metadatas']):
                            metadata = results['metadatas'][i]
                            print(f"    元数据: {metadata}")

            # 测试检索
            test_queries = ["密码", "产品", "技术支持"]
            for query in test_queries:
                print(f"\n[INFO] 测试检索查询: '{query}'")
                try:
                    docs = vectorstore.similarity_search(query, k=2)
                    print(f"  找到 {len(docs)} 个相关文档:")
                    for i, doc in enumerate(docs):
                        print(f"    文档 {i+1}: {doc.page_content[:80]}...")
                        print(f"      元数据: {doc.metadata}")
                except Exception as e:
                    print(f"  检索失败: {e}")

        return True

    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chroma_db()
    sys.exit(0 if success else 1)