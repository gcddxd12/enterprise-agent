"""
中国移动智能客服 - 向量知识库增量构建脚本
对新增 300 条 CMCC 数据做切分+向量化，增量添加到现有 ChromaDB
旧数据不动、不删、不重建
"""
import os
import json
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

# ========== 初始化 Embedding（与旧数据使用相同的模型） ==========
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=api_key
)

# ========== 加载新增 CMCC 数据 ==========
labeled_path = os.path.join("knowledge_data", "cmcc_knowledge_labeled.json")
raw_path = os.path.join("data", "cmcc_300_knowledge.txt")

if os.path.exists(labeled_path):
    with open(labeled_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"从 {labeled_path} 加载 {len(items)} 条已分类知识")
else:
    with open(raw_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    items = [{"id": i + 1, "category": "综合业务", "content": line} for i, line in enumerate(lines)]
    print(f"从 {raw_path} 加载 {len(items)} 条知识（无分类标签）")

# ========== 拼接为全文后，用与旧数据相同的规则切分 ==========
# 每条内容前加分类标签，提高语义匹配度
full_text_parts = []
for item in items:
    full_text_parts.append(f"[{item['category']}] {item['content']}。")

full_text = "\n".join(full_text_parts)

# 与旧数据完全一致的切分规则
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)
chunks = text_splitter.split_text(full_text)
print(f"切分完成: {len(items)} 条原始数据 → {len(chunks)} 个 chunk")

# ========== 构建 Document ==========
documents = [
    Document(
        page_content=chunk,
        metadata={"source": "cmcc_300_knowledge"}
    )
    for chunk in chunks
]

# ========== 增量添加到现有向量库（不删旧数据） ==========
persist_dir = "./chroma_db"

if os.path.exists(persist_dir):
    # 检测旧库使用的 collection 名称
    import chromadb
    client = chromadb.PersistentClient(path=persist_dir)
    existing_collections = client.list_collections()
    if existing_collections:
        # 使用已存在的 collection（可能是旧 build_vector_store.py 所建）
        collection_name = existing_collections[0].name
    else:
        collection_name = "cmcc_knowledge"

    print(f"使用已有 collection: {collection_name}")

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name
    )
    before_count = vectorstore._collection.count()
    vectorstore.add_documents(documents)
    after_count = vectorstore._collection.count()
    print(f"增量添加完成: {before_count} 条 → {after_count} 条 (+{after_count - before_count})")
else:
    # 首次创建
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="cmcc_knowledge"
    )
    print(f"首次创建向量库: {vectorstore._collection.count()} 条文档")

# ========== 快速验证 ==========
print("\n=== 检索验证 ===")
test_queries = [
    "5G套餐有什么特点",
    "如何保障网络安全",
    "物联网平台支持哪些设备",
    "算力网络是什么",
    "如何重置密码",  # 旧数据
]
for q in test_queries:
    results = vectorstore.similarity_search(q, k=2)
    print(f"\n查询: {q}")
    for r in results:
        print(f"  [{r.metadata.get('source', 'old_kb')}] {r.page_content[:80]}...")
