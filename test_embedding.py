import os
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings

# 加载环境变量
load_dotenv()

# 从环境变量读取API Key
api_key = os.getenv("DASHSCOPE_API_KEY")
print(f"读取到的API Key: {api_key[:10]}...{api_key[-4:] if api_key else 'None'}")  # 只显示前后几位，安全

if not api_key:
    print("❌ 没有读取到 DASHSCOPE_API_KEY，请检查 .env 文件")
    exit(1)

# 初始化DashScope嵌入模型
# 注意：Qwen3-Embedding-8B 在DashScope上的模型名称是 "qwen3-embedding-8b"
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=api_key
)

# 测试文本 - 使用指令感知格式能提升效果
# Qwen3-Embedding支持"指令感知"，可以告诉模型你的任务类型
task_description = "为编程问题检索相关的技术文档"
query = "Python中列表和元组的区别是什么？"

# 格式化输入：把任务描述和查询组合起来
formatted_text = f"Instruct: {task_description}\nQuery: {query}"

try:
    print("正在生成向量...")
    # 生成向量
    vector = embeddings.embed_query(formatted_text)

    print(f"✅ 成功生成向量！")
    print(f"向量维度: {len(vector)}")
    print(f"向量前10个值: {[round(x, 4) for x in vector[:10]]}")

    # 测试一下不加指令的效果（对比用）
    vector_simple = embeddings.embed_query(query)
    print(f"\n不加指令的向量维度: {len(vector_simple)}")

except Exception as e:
    print(f"❌ 调用失败: {e}")