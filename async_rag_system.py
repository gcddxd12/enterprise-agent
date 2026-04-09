#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异步RAG系统 - 企业智能客服Agent第二阶段优化

功能：
1. 异步工具调用和检索
2. 并行向量和关键词检索
3. 异步缓存管理
4. 批处理查询优化

作者：gcddxd12
版本：1.0.0
创建日期：2026-04-09
"""

import asyncio
import aiofiles
import json
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseLanguageModel
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"警告：LangChain导入失败，异步RAG功能受限: {e}")
    LANGCHAIN_AVAILABLE = False

# ========== 异步缓存管理器 ==========
class AsyncVectorCache:
    """异步向量缓存管理器"""

    def __init__(self, cache_dir: str = "./async_vector_cache", max_size: int = 1000, ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.embedding_cache = {}  # 文本->嵌入向量缓存
        self.result_cache = {}     # 查询->检索结果缓存
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)

        # 异步加载缓存
        asyncio.create_task(self._async_load_cache())

    async def _async_load_cache(self):
        """异步加载持久化缓存"""
        cache_file = os.path.join(self.cache_dir, "async_vector_cache.pkl")
        if os.path.exists(cache_file):
            try:
                async with aiofiles.open(cache_file, "rb") as f:
                    import pickle
                    data = pickle.loads(await f.read())
                    self.embedding_cache = data.get("embedding_cache", {})
                    self.result_cache = data.get("result_cache", {})
                    print(f"[ASYNC] 向量缓存已加载，条目数: {len(self.embedding_cache)} 个嵌入，{len(self.result_cache)} 个结果")
            except Exception as e:
                print(f"[ASYNC] 加载缓存失败: {e}")

    async def save_cache(self):
        """异步保存缓存到磁盘"""
        cache_file = os.path.join(self.cache_dir, "async_vector_cache.pkl")
        try:
            import pickle
            data = {
                "embedding_cache": self.embedding_cache,
                "result_cache": self.result_cache,
                "timestamp": datetime.now().isoformat()
            }
            async with aiofiles.open(cache_file, "wb") as f:
                await f.write(pickle.dumps(data))
        except Exception as e:
            print(f"[ASYNC] 保存缓存失败: {e}")

    async def get_cached_embedding(self, text: str, embedding_func: Callable) -> List[float]:
        """异步获取缓存的嵌入向量"""
        cache_key = hashlib.md5(text.encode()).hexdigest()

        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        # 异步计算嵌入
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            self.executor, embedding_func, text
        )

        # 缓存结果
        self.embedding_cache[cache_key] = embedding

        # 如果缓存过大，清理旧条目
        if len(self.embedding_cache) > self.max_size:
            self._cleanup_cache()

        return embedding

    async def get_cached_results(self, query: str, search_func: Callable, use_cache: bool = True) -> List[Document]:
        """异步获取缓存的检索结果"""
        if not use_cache:
            return await self._async_search(query, search_func)

        cache_key = hashlib.md5(query.encode()).hexdigest()

        if cache_key in self.result_cache:
            cache_entry = self.result_cache[cache_key]
            # 检查缓存是否过期
            if (datetime.now() - cache_entry["timestamp"]).total_seconds() < self.ttl_hours * 3600:
                return cache_entry["results"]

        # 执行搜索
        results = await self._async_search(query, search_func)

        # 缓存结果
        self.result_cache[cache_key] = {
            "results": results,
            "timestamp": datetime.now(),
            "query": query
        }

        # 如果缓存过大，清理旧条目
        if len(self.result_cache) > self.max_size:
            self._cleanup_cache()

        return results

    async def _async_search(self, query: str, search_func: Callable) -> List[Document]:
        """异步执行搜索"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, search_func, query)

    def _cleanup_cache(self):
        """清理缓存（保留最新的max_size/2个条目）"""
        # 按时间戳排序，保留最新的
        if len(self.embedding_cache) > self.max_size:
            items = list(self.embedding_cache.items())
            items.sort(key=lambda x: hash(x[0]))  # 简单排序
            self.embedding_cache = dict(items[-self.max_size//2:])

        if len(self.result_cache) > self.max_size:
            items = list(self.result_cache.items())
            items.sort(key=lambda x: x[1]["timestamp"], reverse=True)
            self.result_cache = dict(items[:self.max_size//2])

    async def clear_cache(self):
        """清空缓存"""
        self.embedding_cache.clear()
        self.result_cache.clear()
        print("[ASYNC] 向量缓存已清空")

# ========== 异步查询扩展器 ==========
class AsyncQueryExpander:
    """异步查询扩展器"""

    def __init__(self, llm=None, use_mock: bool = True):
        self.llm = llm
        self.use_mock = use_mock
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def expand_query(self, query: str) -> List[str]:
        """异步扩展查询"""
        if self.use_mock or not self.llm:
            return await self._mock_expand_query(query)

        try:
            # 异步调用LLM进行查询扩展
            prompt = f"""请为以下查询生成3个相关的变体或扩展：
            原始查询: {query}

            请生成：
            1. 同义查询
            2. 更详细的查询
            3. 更具体的查询

            返回格式：JSON列表，例如 ["查询1", "查询2", "查询3"]"""

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor, self.llm.invoke, {"query": prompt}
            )

            # 解析响应
            import json
            expanded_queries = json.loads(response.content)
            return [query] + expanded_queries[:3]  # 包含原始查询
        except Exception as e:
            print(f"[ASYNC] 查询扩展失败: {e}")
            return await self._mock_expand_query(query)

    async def _mock_expand_query(self, query: str) -> List[str]:
        """模拟查询扩展"""
        # 简单扩展逻辑
        expansions = [
            query,
            f"{query}的方法",
            f"{query}的步骤",
            f"{query}详细说明"
        ]
        return expansions[:3]  # 最多返回3个扩展

# ========== 异步高级检索器 ==========
class AsyncAdvancedRAGRetriever:
    """异步高级RAG检索器"""

    def __init__(self,
                 vector_retriever: Any = None,
                 keyword_retriever: Any = None,
                 query_expander: AsyncQueryExpander = None,
                 use_cache: bool = True,
                 cache_dir: str = "./async_vector_cache"):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.query_expander = query_expander or AsyncQueryExpander(use_mock=True)
        self.use_cache = use_cache
        self.cache_manager = AsyncVectorCache(cache_dir=cache_dir)
        self.executor = ThreadPoolExecutor(max_workers=4)

        print(f"[ASYNC] 异步高级RAG检索器已初始化: "
              f"向量检索器={vector_retriever is not None}, "
              f"关键词检索器={keyword_retriever is not None}, "
              f"查询扩展={query_expander is not None}, "
              f"缓存={use_cache}")

    async def retrieve(self, query: str, k: int = 5, expand_queries: bool = True) -> List[Document]:
        """异步检索文档"""
        try:
            # 查询扩展
            expanded_queries = [query]
            if expand_queries and self.query_expander:
                expanded_queries = await self.query_expander.expand_query(query)
                if len(expanded_queries) > 1:
                    print(f"[ASYNC] 查询扩展: 原始='{query}'，扩展变体={expanded_queries[1:]}")

            # 并行执行所有查询的检索
            all_results = []
            tasks = []

            for q in expanded_queries:
                task = self._retrieve_single_query_async(q, k)
                tasks.append(task)

            # 等待所有检索完成
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            # 合并结果
            for results in results_list:
                if isinstance(results, Exception):
                    print(f"[ASYNC] 检索失败: {results}")
                    continue
                all_results.extend(results)

            # 重排序和去重
            final_results = await self._rerank_and_deduplicate_async(all_results, query, k)

            return final_results[:k]

        except Exception as e:
            print(f"[ASYNC] 异步检索失败: {e}")
            return []

    async def _retrieve_single_query_async(self, query: str, k: int) -> List[Document]:
        """异步执行单个查询的检索"""
        vector_results = []
        keyword_results = []

        # 并行执行向量和关键词检索
        tasks = []

        if self.vector_retriever:
            tasks.append(self._retrieve_vector_async(query, k))

        if self.keyword_retriever:
            tasks.append(self._retrieve_keyword_async(query, k))

        # 等待所有检索完成
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    print(f"[ASYNC] 子检索失败: {result}")
                elif result:
                    if result and len(result) > 0:
                        if hasattr(result[0], 'metadata') and 'retriever' in result[0].metadata:
                            if result[0].metadata['retriever'] == 'vector':
                                vector_results = result
                            else:
                                keyword_results = result
                        else:
                            # 默认分配
                            vector_results = result

        # 合并结果
        combined_results = vector_results + keyword_results

        # 添加检索器信息
        for i, doc in enumerate(combined_results):
            if 'retriever' not in doc.metadata:
                doc.metadata['retriever'] = 'vector' if i < len(vector_results) else 'keyword'
            if 'score' not in doc.metadata:
                doc.metadata['score'] = 0.8 if i < len(vector_results) else 0.7

        return combined_results

    async def _retrieve_vector_async(self, query: str, k: int) -> List[Document]:
        """异步向量检索"""
        if not self.vector_retriever:
            return []

        try:
            if self.use_cache:
                results = await self.cache_manager.get_cached_results(
                    query,
                    lambda q: self.vector_retriever.invoke(q)[:k*2],
                    use_cache=True
                )
            else:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    self.executor, self.vector_retriever.invoke, query
                )
                results = results[:k*2]

            # 添加权重信息
            for doc in results:
                doc.metadata["weight"] = 0.8
                doc.metadata["retriever"] = "vector"

            return results
        except Exception as e:
            print(f"[ASYNC] 向量检索失败: {e}")
            return []

    async def _retrieve_keyword_async(self, query: str, k: int) -> List[Document]:
        """异步关键词检索"""
        if not self.keyword_retriever:
            return []

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor, self.keyword_retriever.invoke, query
            )
            results = results[:k*2]

            # 添加权重信息
            for doc in results:
                doc.metadata["weight"] = 0.7
                doc.metadata["retriever"] = "keyword"

            return results
        except Exception as e:
            print(f"[ASYNC] 关键词检索失败: {e}")
            return []

    async def _rerank_and_deduplicate_async(self, results: List[Document], original_query: str, k: int) -> List[Document]:
        """异步重排序和去重"""
        if not results:
            return []

        # 简单的去重和排序
        seen_content = set()
        unique_results = []

        for doc in results:
            content_hash = hashlib.md5(doc.page_content[:100].encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)

        # 按分数排序
        unique_results.sort(
            key=lambda x: x.metadata.get("score", 0) * x.metadata.get("weight", 1),
            reverse=True
        )

        return unique_results[:k*2]

    async def get_relevant_documents(self, query: str) -> List[Document]:
        """兼容BaseRetriever接口"""
        return await self.retrieve(query)

# ========== 异步工具函数 ==========
async def create_async_advanced_rag_system(
    vectorstore=None,
    documents: List[Document] = None,
    llm=None,
    use_mock: bool = True,
    config: Dict[str, Any] = None
) -> AsyncAdvancedRAGRetriever:
    """
    创建异步高级RAG系统

    Args:
        vectorstore: 向量数据库
        documents: 文档列表（用于BM25）
        llm: 语言模型
        use_mock: 是否使用模拟模式
        config: 配置字典

    Returns:
        异步高级RAG检索器
    """
    config = config or {}

    # 向量检索器
    vector_retriever = None
    if vectorstore:
        try:
            vector_retriever = vectorstore.as_retriever(
                search_kwargs={"k": config.get("vector_k", 10)}
            )
        except Exception as e:
            print(f"[ASYNC] 创建向量检索器失败: {e}")

    # 关键词检索器（BM25）
    keyword_retriever = None
    if documents and LANGCHAIN_AVAILABLE:
        try:
            from langchain_community.retrievers import BM25Retriever
            keyword_retriever = BM25Retriever.from_documents(documents)
            keyword_retriever.k = config.get("bm25_k", 10)
        except Exception as e:
            print(f"[ASYNC] 创建BM25检索器失败: {e}")

    # 查询扩展器
    query_expander = AsyncQueryExpander(llm=llm, use_mock=use_mock)

    # 创建异步高级检索器
    retriever = AsyncAdvancedRAGRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        query_expander=query_expander,
        use_cache=config.get("use_cache", True),
        cache_dir=config.get("cache_dir", "./async_vector_cache")
    )

    return retriever

# ========== 测试函数 ==========
async def test_async_rag():
    """测试异步RAG系统"""
    print("=== 测试异步高级RAG系统 ===")

    try:
        # 创建模拟文档
        from langchain_core.documents import Document
        test_documents = [
            Document(
                page_content="如何重置密码？您可以通过登录页面点击'忘记密码'链接重置密码。",
                metadata={"source": "faq", "topic": "password"}
            ),
            Document(
                page_content="产品价格信息：企业版每年10,000元，包含技术支持。",
                metadata={"source": "pricing", "topic": "price"}
            ),
            Document(
                page_content="技术支持时间：工作日9:00-18:00，电话400-123-4567。",
                metadata={"source": "support", "topic": "support"}
            ),
        ]

        # 创建异步RAG系统
        retriever = await create_async_advanced_rag_system(
            documents=test_documents,
            use_mock=True,
            config={
                "use_cache": True,
                "query_expansion": True,
                "vector_k": 10,
                "bm25_k": 10
            }
        )

        print(f"异步RAG检索器已初始化: {type(retriever)}")

        # 测试异步检索
        test_queries = ["如何重置密码", "产品价格", "技术支持"]

        for query in test_queries:
            print(f"\n测试查询: '{query}'")
            results = await retriever.retrieve(query, k=2)

            if results:
                for i, doc in enumerate(results):
                    print(f"  结果 {i+1}: {doc.page_content[:60]}...")
                    print(f"    来源: {doc.metadata.get('source', '未知')}")
                    print(f"    检索器: {doc.metadata.get('retriever', '未知')}")
                    print(f"    分数: {doc.metadata.get('score', 0):.3f}")
            else:
                print("  未找到相关结果")

        print("\n[SUCCESS] 异步RAG系统测试完成")
        return True

    except Exception as e:
        print(f"[FAILED] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# ========== 主入口 ==========
if __name__ == "__main__":
    print("异步RAG系统模块")
    print("功能: 异步混合检索、查询扩展、重排序、向量缓存")

    # 运行异步测试
    import asyncio
    asyncio.run(test_async_rag())