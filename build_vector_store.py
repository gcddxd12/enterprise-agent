"""
中国移动智能客服 - 向量知识库增量构建脚本
对新增 CMCC 数据（txt/json/pdf）做切分+向量化，增量添加到现有 ChromaDB
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

# ========== 加载数据源 ==========
all_documents = []

# ---------- 数据源1: txt/json 知识库 ----------
labeled_path = os.path.join("knowledge_data", "cmcc_knowledge_labeled.json")
raw_path = os.path.join("data", "cmcc_300_knowledge.txt")

if os.path.exists(labeled_path):
    with open(labeled_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"[txt] 从 {labeled_path} 加载 {len(items)} 条已分类知识")
else:
    with open(raw_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    items = [{"id": i + 1, "category": "综合业务", "content": line} for i, line in enumerate(lines)]
    print(f"[txt] 从 {raw_path} 加载 {len(items)} 条知识（无分类标签）")

txt_text_parts = []
for item in items:
    txt_text_parts.append(f"[{item['category']}] {item['content']}。")
txt_full = "\n".join(txt_text_parts)

# ---------- 数据源2: PDF 知识库 ----------
pdf_path = os.path.join("data", "中国移动智能客服+营业厅300条超长完整版专业RAG知识库（适配AI客服问答专用）.pdf")
pdf_extracted_path = os.path.join("data", "pdf_extracted.txt")

pdf_text = ""
if os.path.exists(pdf_path):
    try:
        from pdfplumber import open as pdf_open
        print(f"[pdf] 正在解析 PDF: {pdf_path}")
        pdf = pdf_open(pdf_path)
        page_texts = []
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                page_texts.append(t)
        pdf.close()
        pdf_text = "\n".join(page_texts)
        # 缓存提取结果，避免重复解析
        with open(pdf_extracted_path, "w", encoding="utf-8") as f:
            f.write(pdf_text)
        print(f"[pdf] 提取完成: {len(pdf.pages)} 页, {len(pdf_text)} 字符")
    except Exception as e:
        print(f"[pdf] pdfplumber 解析失败: {e}，尝试 pypdf...")
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            page_texts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    page_texts.append(t)
            pdf_text = "\n".join(page_texts)
            with open(pdf_extracted_path, "w", encoding="utf-8") as f:
                f.write(pdf_text)
            print(f"[pdf] 提取完成(pypdf): {len(reader.pages)} 页, {len(pdf_text)} 字符")
        except Exception as e2:
            print(f"[pdf] pypdf 也失败: {e2}，跳过 PDF")
elif os.path.exists(pdf_extracted_path):
    with open(pdf_extracted_path, "r", encoding="utf-8") as f:
        pdf_text = f.read()
    print(f"[pdf] 从缓存加载: {len(pdf_text)} 字符")

# ---------- 合并所有文本 + 统一切分 ----------
# PDF 内容按编号条目切分为独立段落（每条是一个完整知识点）
pdf_chunks = []
if pdf_text:
    import re
    # 去掉标题行
    pdf_body = pdf_text
    # 按 "换行+数字+." 拆分（PDF中条目格式: \n1.xxx \n2.xxx）
    entries = re.split(r'\n(?=\d{1,3}\.)', pdf_body)
    for entry in entries:
        # 清理条目：去掉编号、控制字符（如 \x01）
        entry = re.sub(r'^\d{1,3}\.\s*', '', entry.strip())
        entry = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f]', '', entry)
        # 过滤标题行（含"知识库""超长完整版"等字样）和太短的碎片
        is_title = any(kw in entry for kw in ["完整版专业RAG", "超长完整版", "知识库（适配"])
        if entry and len(entry) > 80 and not is_title:
            pdf_chunks.append(entry)
    print(f"[pdf] 按条目切分: {len(pdf_chunks)} 个知识条目")

# 合并 txt 文本和 PDF 条目
if pdf_chunks:
    full_text = txt_full  # txt 保持原样走统一切分
else:
    full_text = txt_full

# txt 部分用原规则切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)
txt_chunks = text_splitter.split_text(full_text)
print(f"[txt] 切分: {len(txt_chunks)} 个 chunk")

# PDF 条目太长的也切一下
all_chunks = txt_chunks.copy()
for entry in pdf_chunks:
    if len(entry) <= 600:
        all_chunks.append(entry)
    else:
        sub_chunks = text_splitter.split_text(entry)
        all_chunks.extend(sub_chunks)

print(f"总计: {len(all_chunks)} 个 chunk (txt: {len(txt_chunks)}, pdf: {len(all_chunks) - len(txt_chunks)})")

# ========== 构建 Document ==========
documents = [
    Document(
        page_content=chunk,
        metadata={"source": "cmcc_combined_knowledge"}
    )
    for chunk in all_chunks
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
    "异地补卡怎么办理",
    "话费余额包含哪些",
    "定向流量和通用流量的区别",
    "营业厅能办哪些高风险业务",
]
for q in test_queries:
    results = vectorstore.similarity_search(q, k=2)
    print(f"\n查询: {q}")
    for r in results:
        preview = r.page_content[:80].replace("\n", " ")
        print(f"  [{r.metadata.get('source', 'old_kb')}] {preview}...")
