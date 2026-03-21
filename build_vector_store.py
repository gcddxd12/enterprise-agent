import os
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document  # 使用 langchain_core 中的 Document

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

# 初始化嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=api_key
)

# 读取知识库文件
with open("knowledge_base.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 文本切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)
chunks = text_splitter.split_text(text)

# 转换为 Document 对象
docs = [Document(page_content=chunk) for chunk in chunks]
print(f"共切分成 {len(docs)} 个文档块")

# 创建向量数据库并持久化
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
vectorstore.persist()
print("向量数据库已创建并保存到 ./chroma_db")