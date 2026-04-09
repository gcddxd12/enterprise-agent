"""
高级RAG系统 - 企业智能客服Agent第二阶段优化

功能：
1. 混合检索：关键词匹配 + 向量相似度
2. 查询扩展：基于LLM生成查询变体和相关术语
3. 重排序：对检索结果进行重新排序，提升相关性
4. 向量缓存：常用查询结果的嵌入向量缓存

作者：gcddxd12
版本：1.0.0
创建日期：2026-04-09
"""

__all__ = [
    'VectorCache',
    'QueryExpander',
    'AdvancedRAGRetriever',
    'create_advanced_rag_system',
    'test_advanced_rag'
]

import os
import json
import pickle
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Callable
from datetime import datetime, timedelta
from dotenv import load_dotenv

# LangChain相关导入
try:
    # 注意：LangChain 1.2.15中，EnsembleRetriever和ContextualCompressionRetriever不可用
    # 我们将实现自己的混合检索逻辑
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseLanguageModel

    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"警告：LangChain导入失败，部分高级功能不可用: {e}")
    LANGCHAIN_AVAILABLE = False

# 加载环境变量
load_dotenv()

# ========== 缓存管理器 ==========
class VectorCache:
    """向量和检索结果缓存管理器"""

    def __init__(self, cache_dir: str = "./vector_cache", max_size: int = 1000, ttl_hours: int = 24):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录路径
            max_size: 最大缓存条目数
            ttl_hours: 缓存有效期（小时）
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.embedding_cache = {}  # 文本->嵌入向量缓存
        self.result_cache = {}     # 查询->检索结果缓存

        # 缓存统计
        self.stats = {
            "embedding_hits": 0,
            "embedding_misses": 0,
            "result_hits": 0,
            "result_misses": 0,
            "total_saved_time": 0.0,  # 估计节省的时间（秒）
            "last_reset": datetime.now().isoformat()
        }

        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)

        # 加载持久化缓存和统计
        self._load_cache()

    def _load_cache(self):
        """加载持久化缓存"""
        cache_file = os.path.join(self.cache_dir, "vector_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                    self.embedding_cache = data.get("embedding_cache", {})
                    self.result_cache = data.get("result_cache", {})
                    self.stats = data.get("stats", self.stats)
                print(f"向量缓存已加载，条目数: {len(self.embedding_cache)} 个嵌入，{len(self.result_cache)} 个结果")
            except Exception as e:
                print(f"加载缓存失败: {e}")

    def _save_cache(self):
        """保存缓存到文件"""
        cache_file = os.path.join(self.cache_dir, "vector_cache.pkl")
        try:
            data = {
                "embedding_cache": self.embedding_cache,
                "result_cache": self.result_cache,
                "stats": self.stats,
                "timestamp": datetime.now().isoformat()
            }
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"保存缓存失败: {e}")

    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get_embedding(self, text: str, embedding_func: Callable) -> List[float]:
        """
        获取或计算文本的嵌入向量

        Args:
            text: 文本内容
            embedding_func: 嵌入函数

        Returns:
            嵌入向量
        """
        cache_key = self._get_cache_key(text)

        # 检查缓存
        if cache_key in self.embedding_cache:
            cached_data = self.embedding_cache[cache_key]
            cached_time = datetime.fromisoformat(cached_data["timestamp"])
            if datetime.now() - cached_time < timedelta(hours=self.ttl_hours):
                # 缓存命中
                self.stats["embedding_hits"] += 1
                # 假设节省了0.1秒（嵌入计算时间）
                self.stats["total_saved_time"] += 0.1
                return cached_data["embedding"]

        # 缓存未命中
        self.stats["embedding_misses"] += 1

        # 计算新嵌入
        embedding = embedding_func(text)

        # 更新缓存
        self.embedding_cache[cache_key] = {
            "embedding": embedding,
            "text": text[:100],  # 只存储前100个字符用于调试
            "timestamp": datetime.now().isoformat(),
            "access_count": 1
        }

        # 限制缓存大小
        if len(self.embedding_cache) > self.max_size:
            # 删除访问次数最少的条目（如果可用），否则删除最旧的
            if all("access_count" in data for data in self.embedding_cache.values()):
                # 基于访问次数删除
                oldest_key = min(self.embedding_cache.keys(),
                               key=lambda k: self.embedding_cache[k].get("access_count", 0))
            else:
                # 基于时间删除
                oldest_key = min(self.embedding_cache.keys(),
                               key=lambda k: self.embedding_cache[k]["timestamp"])
            del self.embedding_cache[oldest_key]

        self._save_cache()
        return embedding

    def get_search_results(self, query: str, search_func: Callable, use_cache: bool = True) -> List[Document]:
        """
        获取或计算检索结果

        Args:
            query: 查询文本
            search_func: 检索函数
            use_cache: 是否使用缓存

        Returns:
            检索结果文档列表
        """
        cache_key = self._get_cache_key(query)

        # 检查缓存
        if use_cache and cache_key in self.result_cache:
            cached_data = self.result_cache[cache_key]
            cached_time = datetime.fromisoformat(cached_data["timestamp"])
            if datetime.now() - cached_time < timedelta(hours=self.ttl_hours):
                # 缓存命中
                self.stats["result_hits"] += 1
                # 假设节省了0.5秒（检索时间）
                self.stats["total_saved_time"] += 0.5

                # 更新访问计数
                if "access_count" in cached_data:
                    cached_data["access_count"] += 1
                else:
                    cached_data["access_count"] = 1
                self.result_cache[cache_key] = cached_data

                return cached_data["results"]

        # 缓存未命中
        self.stats["result_misses"] += 1

        # 执行检索
        results = search_func(query)

        # 更新缓存
        self.result_cache[cache_key] = {
            "results": results,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "access_count": 1
        }

        # 限制缓存大小
        if len(self.result_cache) > self.max_size:
            # 删除访问次数最少的条目（如果可用），否则删除最旧的
            if all("access_count" in data for data in self.result_cache.values()):
                # 基于访问次数删除
                oldest_key = min(self.result_cache.keys(),
                               key=lambda k: self.result_cache[k].get("access_count", 0))
            else:
                # 基于时间删除
                oldest_key = min(self.result_cache.keys(),
                               key=lambda k: self.result_cache[k]["timestamp"])
            del self.result_cache[oldest_key]

        self._save_cache()
        return results

    def clear_cache(self):
        """清空缓存"""
        self.embedding_cache = {}
        self.result_cache = {}
        self.stats = {
            "embedding_hits": 0,
            "embedding_misses": 0,
            "result_hits": 0,
            "result_misses": 0,
            "total_saved_time": 0.0,
            "last_reset": datetime.now().isoformat()
        }
        self._save_cache()
        print("向量缓存已清空")

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        embedding_total = self.stats["embedding_hits"] + self.stats["embedding_misses"]
        result_total = self.stats["result_hits"] + self.stats["result_misses"]

        embedding_hit_rate = 0.0
        if embedding_total > 0:
            embedding_hit_rate = self.stats["embedding_hits"] / embedding_total

        result_hit_rate = 0.0
        if result_total > 0:
            result_hit_rate = self.stats["result_hits"] / result_total

        return {
            "embedding_cache_size": len(self.embedding_cache),
            "result_cache_size": len(self.result_cache),
            "embedding_hits": self.stats["embedding_hits"],
            "embedding_misses": self.stats["embedding_misses"],
            "embedding_hit_rate": embedding_hit_rate,
            "result_hits": self.stats["result_hits"],
            "result_misses": self.stats["result_misses"],
            "result_hit_rate": result_hit_rate,
            "total_saved_time_seconds": self.stats["total_saved_time"],
            "last_reset": self.stats["last_reset"]
        }

    def print_stats(self):
        """打印缓存统计信息"""
        stats = self.get_stats()
        print("=== 向量缓存统计 ===")
        print(f"嵌入缓存大小: {stats['embedding_cache_size']} 个条目")
        print(f"结果缓存大小: {stats['result_cache_size']} 个条目")
        print(f"嵌入缓存命中率: {stats['embedding_hit_rate']:.1%} ({stats['embedding_hits']}/{stats['embedding_hits']+stats['embedding_misses']})")
        print(f"结果缓存命中率: {stats['result_hit_rate']:.1%} ({stats['result_hits']}/{stats['result_hits']+stats['result_misses']})")
        print(f"估计节省时间: {stats['total_saved_time_seconds']:.2f} 秒")
        print(f"统计重置时间: {stats['last_reset']}")

# ========== 查询扩展器 ==========
class QueryExpander:
    """查询扩展器：生成查询变体和相关术语"""

    def __init__(self, llm=None, use_mock: bool = True):
        """
        初始化查询扩展器

        Args:
            llm: 语言模型实例
            use_mock: 是否使用模拟模式
        """
        self.llm = llm
        self.use_mock = use_mock

        # 预定义的查询扩展模板
        self.expansion_templates = [
            "原始查询: {query}",
            "相关问题: 如何{query}",
            "详细说明: 请详细解释{query}",
            "步骤指南: {query}的具体步骤",
            "常见问题: 关于{query}的常见问题",
            "相关概念: {query}涉及的关键概念"
        ]

        # 企业客服领域同义词词典
        self.synonyms = {
            "密码": ["口令", "登录密码", "账户密码", "用户密码"],
            "重置": ["修改", "更改", "恢复", "找回"],
            "产品": ["服务", "解决方案", "软件", "系统"],
            "价格": ["费用", "成本", "收费标准", "定价"],
            "技术支持": ["客服支持", "客户服务", "技术支持", "帮助中心"],
            "工单": ["服务请求", "支持工单", "问题工单", "客服工单"],
            "登录": ["登入", "进入系统", "访问账户"],
            "注册": ["开户", "创建账户", "账号注册"],
            "付款": ["支付", "缴费", "结算", "账单支付"],
            "发票": ["收据", "账单", "费用凭证"]
        }

        # 常见问题模式
        self.question_patterns = [
            ("如何(.+)", ["怎样{0}", "{0}的方法", "{0}的步骤"]),
            ("什么(.+)", ["哪些{0}", "关于{0}的信息", "{0}是什么"]),
            ("为什么(.+)", ["为何{0}", "{0}的原因", "{0}的缘由"]),
            ("怎么(.+)", ["如何{0}", "{0}的操作", "{0}的方式"])
        ]

    def expand_query(self, query: str, max_variants: int = 3) -> List[str]:
        """
        扩展查询，生成多个变体

        Args:
            query: 原始查询
            max_variants: 最大变体数量

        Returns:
            查询变体列表
        """
        if self.use_mock or self.llm is None:
            # 模拟模式：使用预定义模板
            return self._mock_expand_query(query, max_variants)

        # 实际LLM模式
        try:
            prompt = f"""请为以下查询生成{max_variants}个相关的查询变体，用于改进检索效果。

原始查询: "{query}"

请生成：
1. 同义词替换
2. 问题重述
3. 相关概念
4. 具体示例
5. 上下文扩展

请以JSON格式返回，包含"variants"字段，值为字符串列表。"""

            response = self.llm.invoke(prompt)

            # 解析响应
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)

            # 尝试解析JSON
            import json
            try:
                data = json.loads(content)
                variants = data.get("variants", [])
            except:
                # 如果JSON解析失败，使用简单规则
                variants = self._extract_variants_from_text(content)

            # 确保包含原始查询
            variants = [query] + [v for v in variants if v != query][:max_variants-1]
            return variants[:max_variants]

        except Exception as e:
            print(f"查询扩展失败，使用模拟模式: {e}")
            return self._mock_expand_query(query, max_variants)

    def _mock_expand_query(self, query: str, max_variants: int) -> List[str]:
        """模拟查询扩展（改进版）"""
        variants = [query]

        # 1. 同义词替换扩展
        for word, synonyms in self.synonyms.items():
            if word in query:
                for synonym in synonyms[:2]:  # 每个词最多使用2个同义词
                    variant = query.replace(word, synonym)
                    if variant != query:
                        variants.append(variant)

        # 2. 问题模式扩展
        for pattern, templates in self.question_patterns:
            import re
            match = re.search(pattern, query)
            if match:
                content = match.group(1)
                for template in templates:
                    variant = template.format(content)
                    if variant not in variants:
                        variants.append(variant)

        # 3. 通用扩展规则
        generic_expansions = [
            f"{query}的详细说明",
            f"{query}的具体步骤",
            f"{query}的常见问题",
            f"{query}的相关信息",
            f"{query}的操作指南",
            f"如何解决{query}问题",
            f"{query}的最佳实践"
        ]

        for expansion in generic_expansions:
            if expansion not in variants:
                variants.append(expansion)

        # 4. 语句转换（疑问句↔陈述句）
        if query.endswith("？") or query.endswith("?"):
            # 如果是疑问句，转换为陈述句
            statement = query.rstrip("？?").rstrip("?")
            variants.append(f"关于{statement}的说明")
            variants.append(f"{statement}相关信息")
        else:
            # 如果是陈述句，转换为疑问句
            variants.append(f"如何{query}？")
            variants.append(f"什么是{query}？")
            variants.append(f"为什么需要{query}？")

        # 去重并限制数量
        seen = set()
        unique_variants = []
        for v in variants:
            if v not in seen and len(v) > 2:  # 过滤掉太短的变体
                seen.add(v)
                unique_variants.append(v)

        return unique_variants[:max_variants]

    def _extract_variants_from_text(self, text: str) -> List[str]:
        """从文本中提取查询变体"""
        # 简单规则：提取带数字或项目符号的行
        lines = text.split('\n')
        variants = []

        for line in lines:
            line = line.strip()
            # 移除编号和项目符号
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '- ', '* ', '• ')):
                line = line[2:].strip()

            if line and len(line) > 5 and not line.startswith('原始查询'):
                variants.append(line)

        return variants

# ========== 高级检索器 ==========
class AdvancedRAGRetriever:
    """高级RAG检索器：集成混合检索、查询扩展和重排序"""

    def __init__(self,
                 vector_retriever: Any = None,
                 keyword_retriever: Any = None,
                 query_expander: QueryExpander = None,
                 use_cache: bool = True,
                 cache_dir: str = "./vector_cache"):
        """
        初始化高级检索器

        Args:
            vector_retriever: 向量检索器
            keyword_retriever: 关键词检索器（如BM25）
            query_expander: 查询扩展器
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
        """
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.query_expander = query_expander or QueryExpander(use_mock=True)
        self.use_cache = use_cache

        # 初始化缓存
        self.cache_manager = VectorCache(cache_dir=cache_dir)

        # 权重配置
        self.vector_weight = 0.7
        self.keyword_weight = 0.3

        print(f"高级RAG检索器已初始化: 向量检索器={vector_retriever is not None}, "
              f"关键词检索器={keyword_retriever is not None}, 查询扩展={query_expander is not None}")

    def retrieve(self, query: str, k: int = 5, expand_queries: bool = True) -> List[Document]:
        """
        执行高级检索

        Args:
            query: 查询文本
            k: 返回结果数量
            expand_queries: 是否使用查询扩展

        Returns:
            检索结果文档列表
        """
        # 查询扩展
        if expand_queries:
            expanded_queries = self.query_expander.expand_query(query, max_variants=3)
            print(f"查询扩展: 原始='{query}'，扩展变体={expanded_queries}")
        else:
            expanded_queries = [query]

        all_results = []

        # 对每个查询变体执行检索
        for q in expanded_queries:
            results = self._retrieve_single_query(q, k * 2)  # 获取更多结果用于重排序
            all_results.extend(results)

        # 去重和重排序
        final_results = self._rerank_and_deduplicate(all_results, query, k)

        return final_results

    def _retrieve_single_query(self, query: str, k: int) -> List[Document]:
        """执行单个查询的检索"""
        results = []

        # 向量检索
        if self.vector_retriever:
            try:
                if self.use_cache:
                    vector_results = self.cache_manager.get_search_results(
                        query,
                        lambda q: self.vector_retriever.invoke(q)[:k*2],
                        use_cache=True
                    )
                else:
                    vector_results = self.vector_retriever.invoke(query)[:k*2]

                # 添加权重信息
                for doc in vector_results:
                    doc.metadata["score"] = doc.metadata.get("score", 0.8) * self.vector_weight
                    doc.metadata["retriever"] = "vector"

                results.extend(vector_results)
            except Exception as e:
                print(f"向量检索失败: {e}")

        # 关键词检索
        if self.keyword_retriever:
            try:
                keyword_results = self.keyword_retriever.invoke(query)[:k*2]

                # 添加权重信息
                for doc in keyword_results:
                    doc.metadata["score"] = doc.metadata.get("score", 0.7) * self.keyword_weight
                    doc.metadata["retriever"] = "keyword"

                results.extend(keyword_results)
            except Exception as e:
                print(f"关键词检索失败: {e}")

        return results

    def _rerank_and_deduplicate(self, results: List[Document], original_query: str, k: int) -> List[Document]:
        """重排序和去重"""
        if not results:
            return []

        # 去重：基于内容哈希
        seen_content = set()
        unique_results = []

        for doc in results:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)

        # 改进的重排序算法
        for doc in unique_results:
            base_score = doc.metadata.get("score", 0.5)

            # 计算查询相关性（使用改进的算法）
            relevance_score = self._calculate_relevance_score(doc.page_content, original_query)

            # 考虑文档长度惩罚（避免过长文档占据优势）
            content_length = len(doc.page_content)
            length_penalty = 1.0
            if content_length > 1000:  # 如果文档过长，适当惩罚
                length_penalty = 0.8
            elif content_length < 50:  # 如果文档过短，也适当惩罚
                length_penalty = 0.9

            # 考虑元数据信息
            metadata_boost = 1.0
            metadata = doc.metadata
            if metadata:
                # 如果文档有明确的来源信息，给予轻微提升
                if metadata.get("source"):
                    metadata_boost = 1.05
                # 如果文档有主题信息且与查询相关，给予更大提升
                if metadata.get("topic") and metadata["topic"] in original_query.lower():
                    metadata_boost = 1.1

            # 计算最终分数
            final_score = base_score * 0.6 + relevance_score * 0.4
            final_score = final_score * length_penalty * metadata_boost

            # 更新元数据
            doc.metadata["final_score"] = final_score
            doc.metadata["relevance_score"] = relevance_score
            doc.metadata["length_penalty"] = length_penalty
            doc.metadata["metadata_boost"] = metadata_boost

        # 按最终分数排序
        unique_results.sort(key=lambda x: x.metadata.get("final_score", 0), reverse=True)

        return unique_results[:k]

    def _calculate_relevance_score(self, content: str, query: str) -> float:
        """计算查询和内容的相关性分数（改进版）"""
        # 转换为小写并分词
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())

        if not query_terms:
            return 0.0

        # 计算精确匹配
        exact_matches = len(query_terms.intersection(content_terms))

        # 计算部分匹配（子字符串）
        partial_matches = 0
        for q_term in query_terms:
            for c_term in content_terms:
                if len(q_term) > 2 and q_term in c_term:
                    partial_matches += 0.5
                elif len(c_term) > 2 and c_term in q_term:
                    partial_matches += 0.5

        # 计算查询术语覆盖率
        coverage = (exact_matches + partial_matches) / len(query_terms)

        # 限制在0-1范围内
        return min(coverage, 1.0)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """兼容BaseRetriever接口"""
        return self.retrieve(query)

# ========== 配置和工具函数 ==========
def create_advanced_rag_system(
    vectorstore=None,
    documents: List[Document] = None,
    llm=None,
    use_mock: bool = True,
    config: Dict[str, Any] = None
) -> AdvancedRAGRetriever:
    """
    创建高级RAG系统

    Args:
        vectorstore: 向量数据库
        documents: 文档列表（用于BM25）
        llm: 语言模型
        use_mock: 是否使用模拟模式
        config: 配置字典

    Returns:
        高级RAG检索器
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
            print(f"创建向量检索器失败: {e}")

    # 关键词检索器（BM25）
    keyword_retriever = None
    if documents and LANGCHAIN_AVAILABLE:
        try:
            keyword_retriever = BM25Retriever.from_documents(documents)
            keyword_retriever.k = config.get("bm25_k", 10)
        except Exception as e:
            print(f"创建BM25检索器失败: {e}")

    # 查询扩展器
    query_expander = QueryExpander(llm=llm, use_mock=use_mock)

    # 创建高级检索器
    retriever = AdvancedRAGRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        query_expander=query_expander,
        use_cache=config.get("use_cache", True),
        cache_dir=config.get("cache_dir", "./vector_cache")
    )

    return retriever

# ========== 测试函数 ==========
def test_advanced_rag():
    """测试高级RAG系统"""
    print("=== 测试高级RAG系统 ===")

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
                metadata={"source": "support", "topic": "contact"}
            )
        ]

        # 创建高级RAG检索器（模拟模式）
        retriever = create_advanced_rag_system(
            documents=test_documents,
            use_mock=True,
            config={"use_cache": False}
        )

        # 测试查询
        test_queries = [
            "如何重置密码",
            "产品价格",
            "技术支持"
        ]

        for query in test_queries:
            print(f"\n测试查询: '{query}'")
            results = retriever.retrieve(query, k=2)

            for i, doc in enumerate(results):
                print(f"  结果 {i+1}: {doc.page_content[:60]}...")
                print(f"    来源: {doc.metadata.get('source', '未知')}")
                print(f"    分数: {doc.metadata.get('final_score', 0):.3f}")

        print("\n[SUCCESS] 高级RAG系统测试完成")
        return True

    except Exception as e:
        print(f"[FAILED] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# ========== 主入口 ==========
if __name__ == "__main__":
    print("高级RAG系统模块")
    print("功能: 混合检索、查询扩展、重排序、向量缓存")

    # 运行测试
    test_advanced_rag()