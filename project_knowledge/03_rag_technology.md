# RAG（检索增强生成）技术

## 概述
RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术，通过从知识库中检索相关信息来增强大语言模型的生成能力。本项目实现了从基础RAG到高级优化RAG的完整演进。

## 1. RAG基础架构

### 1.1 离线构建阶段

**流程**:
```
原始文本 → 文本清洗 → 文本分割 → 向量化 → 向量存储
```

**代码示例**:
```python
# build_vector_store.py 第1-50行（概念代码）：向量库构建
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 加载文档
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

# 2. 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 分块大小
    chunk_overlap=50,    # 重叠大小
    separators=["\n\n", "\n", "。", "，", " ", ""]  # 中文友好分隔符
)
texts = text_splitter.split_documents(documents)

# 3. 创建嵌入
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=api_key
)

# 4. 存储到向量数据库
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"  # 持久化存储
)
```

### 1.2 在线查询阶段

**流程**:
```
用户查询 → 向量化 → 相似度检索 → 上下文拼接 → LLM生成 → 最终答案
```

**代码示例**:
```python
# rag_agent.py 第15-32行：基础RAG实现
# 1. 加载向量数据库
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=api_key
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 2. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 返回top-3文档

# 3. 创建检索问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 将所有文档拼接后传入
    retriever=retriever,
    return_source_documents=True  # 返回源文档用于调试
)

# 4. 查询
def retrieve_and_answer(query: str) -> str:
    """从知识库检索并回答问题"""
    result = qa_chain.invoke({"query": query})
    return result["result"]
```

## 2. 文本分割策略

### 2.1 分块参数设计

**代码示例**:
```python
# 文本分割器配置
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 平衡信息密度和上下文长度
    chunk_overlap=50,    # 避免信息割裂
    separators=["\n\n", "\n", "。", "，", " ", ""]  # 中文友好分隔符
)
```

**参数选择依据**:
1. **chunk_size=500**: 适合大多数LLM的上下文窗口
2. **chunk_overlap=50**: 确保关键信息不丢失
3. **中文分隔符**: 针对中文文本特点优化

### 2.2 分块策略比较

**固定大小分块**:
- 优点: 简单统一
- 缺点: 可能切断完整语义单元

**语义分块**:
- 优点: 保持语义完整性
- 缺点: 实现复杂，需要额外模型

**递归分块**（本项目采用）:
- 优点: 平衡简单性和效果
- 缺点: 依赖分隔符质量

## 3. 嵌入模型技术

### 3.1 DashScope嵌入模型

**代码示例**:
```python
# enterprise_agent.py 第33-37行：嵌入模型初始化
from langchain_community.embeddings import DashScopeEmbeddings

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=api_key
        )
    return _embeddings
```

**模型特点**:
- **text-embedding-v4**: 阿里云百炼的嵌入模型
- **支持中文**: 针对中文文本优化
- **高维向量**: 通常1536维，表示能力强

### 3.2 指令感知嵌入

**概念**: 为嵌入模型提供任务上下文，提升检索相关性

**代码示例**:
```python
# 指令感知嵌入格式（概念代码）
def format_text_for_embedding(text: str, task: str = "retrieval") -> str:
    """为嵌入模型格式化文本"""
    if task == "retrieval":
        return f"Instruct: 检索相关信息\nQuery: {text}"
    elif task == "classification":
        return f"Instruct: 分类文本\nText: {text}"
    else:
        return text

# 使用格式化文本进行嵌入
formatted_text = format_text_for_embedding("如何重置密码", "retrieval")
embedding = embeddings.embed_query(formatted_text)
```

### 3.3 维度选择与优化

**代码示例**:
```python
# 维度配置（概念代码，Fireworks API示例）
payload = {
    "input": text,
    "model": "fireworks/qwen3-embedding-8b",
    "dimensions": 1024  # 可选：降低维度节省存储
}
```

**维度权衡**:
1. **高维度（1536）**: 表示能力更强，准确性更高
2. **低维度（1024）**: 存储和计算成本更低
3. **选择标准**: 根据精度需求和资源约束平衡

## 4. 向量数据库（Chroma）

### 4.1 数据库初始化

**代码示例**:
```python
# enterprise_agent.py 第40-48行：向量数据库初始化
def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        # 检查是否在 CI 环境中，如果是则返回 None 或 mock
        if os.getenv("CI"):
            return None
        embeddings = get_embeddings()
        _vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    return _vectorstore
```

**设计考虑**:
1. **环境感知**: CI环境返回None，便于测试
2. **延迟初始化**: 按需加载，提升启动速度
3. **持久化**: 数据保存到磁盘，避免重复计算

### 4.2 检索配置

**代码示例**:
```python
# rag_agent.py 第21行：检索器配置
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 高级检索配置（概念代码）
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 相似度检索
    search_kwargs={
        "k": 5,  # 返回文档数量
        "score_threshold": 0.7,  # 相似度阈值
        "filter": {"category": "technical"}  # 元数据过滤
    }
)
```

**检索策略**:
1. **相似度检索**: 基于向量相似度
2. **MMR检索**: 最大边际相关性，平衡相关性和多样性
3. **混合检索**: 结合多个检索策略

## 5. 高级RAG优化

### 5.1 混合检索器

**文件**: [advanced_rag_system.py](e:\my_multi_agent\advanced_rag_system.py)

**代码示例**:
```python
# advanced_rag_system.py 第50-120行：高级RAG检索器
class AdvancedRAGRetriever:
    def __init__(self, vector_retriever, keyword_retriever=None):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.query_expander = QueryExpander()
        self.cache_manager = VectorCache()
        
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """混合检索：向量相似度 + BM25关键词检索"""
        
        # 1. 查询扩展
        expanded_queries = self.query_expander.expand_query(query, max_variants=3)
        
        # 2. 并行检索
        vector_results = self.vector_retriever.invoke(expanded_queries[0], k=k*2)
        
        keyword_results = []
        if self.keyword_retriever:
            keyword_results = self.keyword_retriever.invoke(query, k=k*2)
        
        # 3. 结果融合
        all_results = self._merge_results(vector_results, keyword_results)
        
        # 4. 重排序
        reranked_results = self._rerank_and_deduplicate(all_results, query)
        
        return reranked_results[:k]
```

**混合检索优势**:
1. **召回率提升**: 向量检索+关键词检索互补
2. **准确性平衡**: 结合语义相似度和字面匹配
3. **鲁棒性增强**: 对查询表述变化更稳健

### 5.2 查询扩展技术

#### 5.2.1 同义词扩展

**代码示例**:
```python
# advanced_rag_system.py 第170-210行：查询扩展器
class QueryExpander:
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.enterprise_synonyms = {
            "密码": ["密码重置", "忘记密码", "修改密码", "密码找回"],
            "产品": ["商品", "服务", "方案", "产品信息"],
            "价格": ["费用", "成本", "报价", "收费标准"],
            "技术支持": ["客服", "帮助", "支持", "问题解答"],
            "工单": ["票据", "单据", "服务单", "问题单"]
        }
    
    def expand_query(self, query: str, max_variants: int = 3) -> List[str]:
        """扩展查询：生成相关查询变体"""
        if self.use_mock:
            return self._mock_expansion(query, max_variants)
        else:
            return self._llm_expansion(query, max_variants)
    
    def _mock_expansion(self, query: str, max_variants: int) -> List[str]:
        """模拟查询扩展：基于同义词词典"""
        variants = [query]
        
        for original, synonyms in self.enterprise_synonyms.items():
            if original in query:
                for synonym in synonyms[:max_variants-1]:
                    variant = query.replace(original, synonym)
                    variants.append(variant)
                    if len(variants) >= max_variants:
                        break
        
        return variants[:max_variants]
```

#### 5.2.2 LLM智能扩展

**代码示例**:
```python
# advanced_rag_system.py 第212-240行：LLM查询扩展
def _llm_expansion(self, query: str, max_variants: int) -> List[str]:
    """使用LLM生成查询变体"""
    prompt = f"""
    你是一个查询扩展专家。请为以下用户查询生成{max_variants-1}个相关变体。
    变体应该：
    1. 使用同义词替换关键词
    2. 从不同角度表达相同问题
    3. 包含更具体或更一般的表述
    
    原始查询：{query}
    
    请输出JSON格式的列表，例如：["变体1", "变体2", ...]
    
    查询变体：
    """
    
    try:
        # 调用LLM生成变体
        response = llm.invoke(prompt)
        variants = json.loads(response.content)
        return [query] + variants[:max_variants-1]
    except Exception as e:
        print(f"[WARN] LLM查询扩展失败，使用模拟模式: {e}")
        return self._mock_expansion(query, max_variants)
```

### 5.3 结果重排序算法

**代码示例**:
```python
# advanced_rag_system.py 第122-168行：重排序算法
def _rerank_and_deduplicate(self, documents: List[Document], query: str) -> List[Document]:
    """重排序和去重：综合多维度评分"""
    scored_docs = []
    
    for doc in documents:
        score = 0.0
        
        # 1. 向量相似度分数（基础）
        if hasattr(doc, 'metadata') and 'similarity_score' in doc.metadata:
            score += doc.metadata['similarity_score'] * 0.6
        
        # 2. 关键词匹配分数
        keyword_score = self._calculate_keyword_score(doc.page_content, query)
        score += keyword_score * 0.2
        
        # 3. 长度惩罚（避免过长或过短）
        length = len(doc.page_content)
        if length < 50:
            score *= 0.7  # 过短文档惩罚
        elif length > 1000:
            score *= 0.8  # 过长文档惩罚
        
        # 4. 元数据增强（来源可信度、新鲜度等）
        if hasattr(doc, 'metadata'):
            if doc.metadata.get('source') in ['official', 'knowledge_base']:
                score *= 1.2  # 官方来源加分
            
            # 新鲜度加分（如果有时间信息）
            if 'timestamp' in doc.metadata:
                days_old = (datetime.now() - doc.metadata['timestamp']).days
                if days_old < 30:
                    score *= 1.1  # 新鲜内容加分
        
        scored_docs.append((score, doc))
    
    # 按分数排序
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # 去重：基于内容相似度
    unique_docs = []
    seen_contents = set()
    
    for score, doc in scored_docs:
        # 简单去重：基于内容哈希
        content_hash = hash(doc.page_content[:200])  # 取前200字符的哈希
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            # 更新文档元数据中的最终分数
            doc.metadata['final_score'] = score
            unique_docs.append(doc)
    
    return unique_docs
```

**重排序维度**:
1. **向量相似度**: 基础相关性
2. **关键词匹配**: 字面相关性
3. **内容质量**: 长度、完整性等
4. **元数据信息**: 来源可信度、新鲜度等

### 5.4 向量缓存系统

**代码示例**:
```python
# advanced_rag_system.py 第242-350行：向量缓存
class VectorCache:
    def __init__(self, cache_dir: str = "./vector_cache", max_size: int = 1000, ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.cache: Dict[str, Dict] = {}
        self.access_count: Dict[str, int] = {}
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "total_saved_time_seconds": 0.0
        }
    
    def get_embedding(self, text: str, embedding_func: Callable) -> List[float]:
        """获取嵌入向量，优先从缓存读取"""
        self.stats["total_requests"] += 1
        cache_key = self._generate_cache_key(text)
        
        # 检查缓存
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            
            # 检查TTL（生存时间）
            if time.time() - cache_entry["timestamp"] < self.ttl_hours * 3600:
                self.stats["cache_hits"] += 1
                self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
                
                # 估计节省的时间（假设每次嵌入计算需要0.1秒）
                self.stats["total_saved_time_seconds"] += 0.1
                
                return cache_entry["embedding"]
            else:
                # 缓存过期，删除
                del self.cache[cache_key]
                if cache_key in self.access_count:
                    del self.access_count[cache_key]
        
        # 缓存未命中，计算并缓存
        start_time = time.time()
        embedding = embedding_func(text)
        calculation_time = time.time() - start_time
        
        # 添加新缓存（实施LRU淘汰策略）
        self._add_to_cache(cache_key, text, embedding, calculation_time)
        
        return embedding
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = self.stats.copy()
        stats["current_cache_size"] = len(self.cache)
        
        # 计算命中率
        if stats["total_requests"] > 0:
            stats["embedding_hit_rate"] = stats["cache_hits"] / stats["total_requests"]
        else:
            stats["embedding_hit_rate"] = 0.0
        
        # 计算平均访问次数
        if self.access_count:
            stats["avg_access_count"] = sum(self.access_count.values()) / len(self.access_count)
        else:
            stats["avg_access_count"] = 0.0
        
        return stats
```

**缓存策略**:
1. **LRU淘汰**: 基于最近使用频率管理缓存大小
2. **TTL管理**: 基于时间失效机制
3. **性能统计**: 跟踪命中率和节省时间
4. **磁盘持久化**: 可选地将缓存保存到磁盘

## 6. RAG性能优化

### 6.1 检索参数调优

**代码示例**:
```python
# 检索参数优化（概念代码）
def optimize_retrieval_parameters():
    """优化检索参数"""
    param_grid = {
        'k': [3, 5, 10],  # 返回文档数量
        'chunk_size': [300, 500, 800],  # 分块大小
        'similarity_threshold': [0.6, 0.7, 0.8]  # 相似度阈值
    }
    
    best_params = {}
    best_score = 0
    
    # 网格搜索寻找最优参数
    for params in itertools.product(*param_grid.values()):
        current_params = dict(zip(param_grid.keys(), params))
        
        # 评估参数效果
        score = evaluate_retrieval_quality(current_params)
        
        if score > best_score:
            best_score = score
            best_params = current_params
    
    return best_params, best_score
```

### 6.2 异步RAG检索

**文件**: [async_rag_system.py](e:\my_multi_agent\async_rag_system.py)（如果存在）

**概念代码**:
```python
# 异步RAG检索（概念代码）
import asyncio
from typing import List
from langchain_core.documents import Document

class AsyncRAGSystem:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    async def async_retrieve(self, query: str, k: int = 5) -> List[Document]:
        """异步检索文档"""
        # 异步执行检索
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(
            None, self.retriever.invoke, query, k
        )
        return documents
    
    async def async_generate(self, query: str, documents: List[Document]) -> str:
        """异步生成答案"""
        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # 异步调用LLM
        prompt = f"基于以下上下文回答问题：\n\n{context}\n\n问题：{query}\n答案："
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, self.llm.invoke, prompt
        )
        
        return response.content
    
    async def end_to_end_async(self, query: str) -> str:
        """端到端异步RAG流程"""
        # 并行执行检索和查询分析
        retrieve_task = self.async_retrieve(query)
        query_analysis_task = self.analyze_query_async(query)
        
        documents, query_analysis = await asyncio.gather(
            retrieve_task, query_analysis_task
        )
        
        # 基于查询分析优化检索结果
        optimized_docs = self.optimize_results(documents, query_analysis)
        
        # 生成最终答案
        answer = await self.async_generate(query, optimized_docs)
        
        return answer
```

## 7. RAG评估与测试

### 7.1 检索质量评估

**文件**: [test_rag_optimization.py](e:\my_multi_agent\test_rag_optimization.py)

**代码示例**:
```python
# test_rag_optimization.py 第64-117行：检索功能测试
def test_retrieval_functionality():
    """测试检索功能"""
    from langgraph_agent_with_memory import advanced_rag_retriever
    
    if not advanced_rag_retriever:
        print("[WARN] 高级RAG检索器未初始化，跳过检索测试")
        return False
    
    # 测试查询
    test_queries = [
        "如何重置密码",
        "产品价格信息", 
        "技术支持联系方式",
        "密码找回方法"
    ]
    
    all_passed = True
    for query in test_queries:
        print(f"\n[INFO] 测试查询: '{query}'")
        try:
            start_time = time.time()
            results = advanced_rag_retriever.retrieve(query, k=3)
            duration = time.time() - start_time
            
            if results:
                print(f"  检索成功: 找到 {len(results)} 个结果 (耗时: {duration:.3f}s)")
                for i, doc in enumerate(results[:2]):
                    score = doc.metadata.get('final_score', 0)
                    source = doc.metadata.get('source', '未知')
                    print(f"    结果 {i+1}: 分数={score:.3f}, 来源={source}")
                    print(f"      内容: {doc.page_content[:80]}...")
            else:
                print(f"  检索成功: 0 个结果 (耗时: {duration:.3f}s)")
                all_passed = False
                
        except Exception as e:
            print(f"  检索失败: {e}")
            all_passed = False
    
    return all_passed
```

### 7.2 端到端测试

**代码示例**:
```python
# test_rag_optimization.py 第206-236行：工作流集成测试
def test_integration_with_workflow():
    """测试与工作流的集成"""
    from langgraph_agent_with_memory import knowledge_search
    
    test_queries = [
        "重置密码",
        "产品价格", 
        "技术支持"
    ]
    
    print("[INFO] 测试knowledge_search工具...")
    for query in test_queries:
        print(f"\n[INFO] 测试查询: '{query}'")
        try:
            response = knowledge_search.run(query)
            print(f"  响应: {response[:100]}...")
            print("  状态: 成功")
        except Exception as e:
            print(f"  响应失败: {e}")
    
    return True
```

## 8. 常见问题与解决方案

### 8.1 检索结果不相关

**可能原因**:
1. 嵌入模型不适合领域
2. 分块策略不合理
3. 查询表述不清晰

**解决方案**:
```python
# 查询重写优化（概念代码）
def optimize_query(query: str) -> str:
    """优化查询表述"""
    # 1. 纠正拼写错误
    query = spell_check(query)
    
    # 2. 扩展缩写
    query = expand_abbreviations(query)
    
    # 3. 添加领域上下文
    if is_technical_query(query):
        query = f"技术问题: {query}"
    elif is_business_query(query):
        query = f"业务咨询: {query}"
    
    return query
```

### 8.2 上下文过长

**解决方案**:
```python
# 上下文截断策略（概念代码）
def truncate_context(documents: List[Document], max_tokens: int = 4000) -> str:
    """智能截断上下文"""
    context_parts = []
    total_tokens = 0
    
    for doc in documents:
        doc_tokens = estimate_tokens(doc.page_content)
        
        if total_tokens + doc_tokens <= max_tokens:
            context_parts.append(doc.page_content)
            total_tokens += doc_tokens
        else:
            # 部分截断最后一个文档
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 100:  # 至少保留100个token
                truncated_content = truncate_text(doc.page_content, remaining_tokens)
                context_parts.append(truncated_content)
            break
    
    return "\n\n".join(context_parts)
```

### 8.3 幻觉问题

**缓解策略**:
1. **引用源文档**: 在答案中注明信息来源
2. **置信度阈值**: 低于阈值的答案标记为不确定
3. **多源验证**: 交叉验证多个检索结果

## 9. 学习总结

### 关键技术要点
1. **分块策略**: 影响检索质量的关键因素
2. **嵌入模型**: 决定语义理解能力
3. **混合检索**: 提升召回率和鲁棒性
4. **查询扩展**: 解决表述差异问题
5. **结果重排序**: 优化最终结果质量
6. **向量缓存**: 提升性能的重要手段

### 最佳实践
1. **参数调优**: 根据数据特点优化分块和检索参数
2. **质量评估**: 建立评估体系监控检索效果
3. **渐进优化**: 从基础RAG开始，逐步添加高级功能
4. **监控统计**: 跟踪命中率、响应时间等关键指标

### 进阶方向
1. **自适应性RAG**: 根据查询动态调整检索策略
2. **多跳检索**: 支持复杂问题的多步检索
3. **实时更新**: 支持知识库的实时增量更新
4. **个性化检索**: 基于用户历史优化检索结果

---

**相关文件**:
- [rag_agent.py](e:\my_multi_agent\rag_agent.py) - 基础RAG实现
- [advanced_rag_system.py](e:\my_multi_agent\advanced_rag_system.py) - 高级RAG优化
- [test_rag_optimization.py](e:\my_multi_agent\test_rag_optimization.py) - RAG测试用例
- [test_chroma_db.py](e:\my_multi_agent\test_chroma_db.py) - 向量数据库测试

**下一步学习**: LangGraph工作流 →