# RAG（检索增强生成）技术

## 概述
RAG（Retrieval-Augmented Generation）通过从知识库中检索相关信息来增强大语言模型的生成能力。本项目实现了从基础RAG到高级优化RAG的完整演进，当前使用 `advanced_rag_system.py` 中的混合检索器作为知识检索的核心引擎。

## 1. RAG基础架构

### 1.1 离线构建阶段

**流程**:
```
原始文本 → 文本清洗 → 文本分割 → 向量化 → 向量存储
```

```python
# build_vector_store.py：向量库构建
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

loader = TextLoader("knowledge_base.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""]  # 中文友好分隔符
)
texts = text_splitter.split_documents(documents)

embeddings = DashScopeEmbeddings(model="text-embedding-v4", dashscope_api_key=api_key)
vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")
```

### 1.2 在线查询阶段（集成在主Agent中）

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第232-324行

```python
# RAG初始化（在模块加载时自动执行）
def init_advanced_rag():
    """初始化高级RAG检索器"""
    embeddings = DashScopeEmbeddings(model="text-embedding-v4", dashscope_api_key=api_key)
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=chroma_db_path)

    # 从向量数据库加载文档用于BM25关键词检索
    documents_for_bm25 = []
    results = vectorstore._collection.get(limit=100)
    for doc_text in results['documents']:
        documents_for_bm25.append(Document(page_content=doc_text, metadata=...))

    # 创建高级RAG系统（向量 + BM25 + 重排序）
    advanced_rag_retriever = create_advanced_rag_system(
        vectorstore=vectorstore,
        documents=documents_for_bm25,
        llm=get_llm(),
        config={"use_cache": True, "query_expansion": True, "vector_k": 10, "bm25_k": 10}
    )
```

`knowledge_search` 工具使用该检索器，不可用时回退到模拟响应。

## 2. 文本分割策略

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 平衡信息密度和上下文长度
    chunk_overlap=50,    # 避免信息割裂
    separators=["\n\n", "\n", "。", "，", " ", ""]  # 中文友好
)
```

**参数选择**: chunk_size=500适合大多数LLM的上下文窗口，overlap=50确保关键信息不丢失，中文分隔符针对中文文本优化。

## 3. 嵌入模型

```python
# langgraph_agent_with_memory.py 第238-254行
from langchain_community.embeddings import DashScopeEmbeddings

embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=api_key
)
```

使用阿里云百炼的 `text-embedding-v4` 模型，针对中文文本优化，通常1536维。

## 4. 向量数据库（Chroma）

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第244-258行

```python
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
```

持久化存储到 `./chroma_db`，启动时加载，CI环境返回None便于测试。

## 5. 高级RAG优化

### 5.1 混合检索器

**文件**: [advanced_rag_system.py](e:\my_multi_agent\advanced_rag_system.py)

```python
class AdvancedRAGRetriever:
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """混合检索：向量相似度 + BM25关键词检索"""
        # 1. 查询扩展
        expanded_queries = self.query_expander.expand_query(query, max_variants=3)

        # 2. 并行检索
        vector_results = self.vector_retriever.invoke(expanded_queries[0], k=k*2)
        keyword_results = self.keyword_retriever.invoke(query, k=k*2)

        # 3. 结果融合
        all_results = self._merge_results(vector_results, keyword_results)

        # 4. 重排序
        reranked_results = self._rerank_and_deduplicate(all_results, query)
        return reranked_results[:k]
```

**混合检索优势**: 向量检索（语义相似度）+ BM25（关键词匹配）互补，提升召回率和鲁棒性。

### 5.2 查询扩展

**文件**: [advanced_rag_system.py](e:\my_multi_agent\advanced_rag_system.py)

```python
class QueryExpander:
    def expand_query(self, query: str, max_variants: int = 3) -> List[str]:
        """基于同义词词典或LLM生成查询变体"""
```

两种模式：基于同义词词典的模拟扩展（快速）和LLM智能扩展（更准确）。

### 5.3 结果重排序

综合多维度评分：向量相似度(×0.6) + 关键词匹配(×0.2) + 长度惩罚 + 元数据增强（来源可信度、新鲜度）。

### 5.4 向量缓存

**文件**: [advanced_rag_system.py](e:\my_multi_agent\advanced_rag_system.py)

```python
class VectorCache:
    def get_embedding(self, text: str, embedding_func: Callable) -> List[float]:
        """获取嵌入向量，优先从缓存读取"""
```

LRU淘汰策略 + TTL时效管理，缓存命中可节省 ~0.1秒/次嵌入计算。

## 6. 在主Agent中的使用

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第328-425行

`knowledge_search` 工具中调用 `advanced_rag_retriever.retrieve(query)`，取top 3结果拼接后返回。LLM在ReAct循环中自主决定何时调用该工具。

```python
@tool
def knowledge_search(query: str) -> str:
    if advanced_rag_retriever:
        results = advanced_rag_retriever.retrieve(query)
        if results:
            top_results = results[:3]
            return "\n\n".join([doc.page_content for doc in top_results])
    # 回退到模拟模式
    return mock_responses.get(key, mock_responses["默认"])
```

## 7. RAG测试

**文件**: [test_rag_optimization.py](e:\my_multi_agent\test_rag_optimization.py) 和 [test_rag_integration.py](e:\my_multi_agent\test_rag_integration.py)

测试覆盖检索功能、工作流集成、缓存命中率等方面。

## 8. 学习总结

### 关键技术要点
1. **分块策略**: chunk_size=500 + overlap=50，中文友好分隔符
2. **嵌入模型**: DashScope text-embedding-v4，1536维
3. **混合检索**: 向量检索 + BM25 + 重排序
4. **查询扩展**: 同义词词典 + LLM智能变体
5. **向量缓存**: LRU + TTL，减少重复计算

### 最佳实践
1. **从基础到高级**: 先用基础RAG验证可行性，再逐步添加优化
2. **优雅降级**: 高级RAG不可用时回退到模拟模式
3. **参数调优**: 根据数据特点优化chunk_size、k值、相似度阈值
4. **缓存策略**: 对频繁查询使用缓存减少API调用

### 进阶方向
1. **自适应性RAG**: 根据查询复杂度动态调整检索策略
2. **多跳检索**: 支持复杂问题的多步检索
3. **实时更新**: 支持知识库增量更新

---

**相关文件**:
- [advanced_rag_system.py](e:\my_multi_agent\advanced_rag_system.py) — 高级RAG优化（混合检索/查询扩展/重排序/缓存）
- [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) — 主Agent中的RAG集成（knowledge_search工具）
- [build_vector_store.py](e:\my_multi_agent\build_vector_store.py) — 向量库构建脚本
- [async_rag_system.py](e:\my_multi_agent\async_rag_system.py) — 异步RAG支持
- [test_rag_optimization.py](e:\my_multi_agent\test_rag_optimization.py) — RAG测试
- [test_rag_integration.py](e:\my_multi_agent\test_rag_integration.py) — RAG集成测试

**下一步学习**: LangGraph工作流 →
