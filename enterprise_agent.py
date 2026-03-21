import os
import json
import re
from datetime import date
from dotenv import load_dotenv
from langchain.tools import tool, Tool
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatTongyi
from langchain.prompts import ChatPromptTemplate

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

# ========== 共享状态 ==========
shared_state = {
    "user_query": "",
    "plan": None,
    "retrieved_docs": [],
    "tool_results": {},
    "final_answer": None
}

# ========== 初始化 LLM 和检索器 ==========
embeddings = DashScopeEmbeddings(model="text-embedding-v4", dashscope_api_key=api_key)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatTongyi(model="qwen-plus", temperature=0)

# 检索问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


# ========== 工具定义（企业业务场景） ==========
@tool
def knowledge_search(query: str) -> str:
    """从企业知识库中检索信息，返回答案。适用于产品使用、技术支持、销售政策等问题。"""
    result = qa_chain.invoke({"query": query})
    return result["result"]


@tool
def query_ticket_status(ticket_id: str) -> str:
    """模拟查询工单状态。工单号格式为 TK-xxxxxx。"""
    # 模拟数据
    mock_status = {
        "TK-123456": "您的工单 TK-123456 已受理，正在处理中，预计48小时内完成。",
        "TK-789012": "工单 TK-789012 已处理完毕，请登录系统查看结果。",
        "default": "未找到工单信息，请确认工单号是否正确。"
    }
    return mock_status.get(ticket_id, mock_status["default"])


@tool
def escalate_to_human(query: str) -> str:
    """当无法回答或用户情绪激动时，转人工处理。"""
    return "感谢您的耐心，我已将您的问题转接给人工客服，他们将尽快与您联系（预计5分钟内）。"


@tool
def get_current_date(query: str) -> str:
    """返回今天的日期。"""
    return str(date.today())


# ========== 规划 Agent ==========
def planning_agent(query: str) -> list:
    """调用 LLM 拆解任务，输出 JSON 格式的任务列表"""
    prompt = ChatPromptTemplate.from_template("""
你是一个企业客服系统的任务规划专家。请将用户的问题拆解为一系列子任务，每个子任务必须是以下之一：
- "knowledge_search: 具体查询内容"   # 从知识库检索信息
- "ticket_query: 工单号"           # 查询工单状态
- "escalate"                        # 转人工
- "date_query"                      # 查询日期

以下是几个示例：
用户问“如何重置密码？” -> ["knowledge_search: 重置密码"]
用户问“查询工单 TK-123456” -> ["ticket_query: TK-123456"]
用户问“今天几号？” -> ["date_query"]
用户问“帮我转人工” -> ["escalate"]

只输出 JSON 格式的任务列表，不要输出其他内容。

用户问题：{query}
任务列表（JSON）：
""")
    chain = prompt | llm
    response = chain.invoke({"query": query}).content
    try:
        tasks = json.loads(response)
    except:
        # 如果模型输出不是纯 JSON，尝试提取
        match = re.search(r'\[.*\]', response, re.DOTALL)
        tasks = json.loads(match.group()) if match else []
    return tasks


# ========== 执行 Agent ==========
def execution_agent(tasks: list) -> dict:
    """依次执行任务，返回结果字典"""
    results = {}
    for task in tasks:
        if task.startswith("knowledge_search:"):
            query = task.replace("knowledge_search:", "").strip()
            results[task] = knowledge_search.run(query)
        elif task.startswith("ticket_query:"):
            ticket_id = task.replace("ticket_query:", "").strip()
            results[task] = query_ticket_status.run(ticket_id)
        elif task == "escalate":
            results[task] = escalate_to_human.run("")
        elif task == "date_query":
            results[task] = get_current_date.run("")
        else:
            results[task] = "未知任务"
    return results


# ========== 验证 Agent（简化版，基于关键词） ==========
def validation_agent(user_query: str, final_answer: str) -> str:
    """
    对执行结果进行验证。
    - 如果答案包含工单信息、日期信息或转人工提示，直接返回（信任确定性工具）
    - 如果答案太短或包含“无法确定”，返回通用拒答
    - 否则原样返回（假设是 RAG 结果，需要进一步验证可在此扩展）
    """
    # 定义确定性结果的关键词
    trust_keywords = ["工单", "受理", "处理中", "已完成", "今天", "年-月-日", "转人工"]
    if any(kw in final_answer for kw in trust_keywords):
        return final_answer
    # 如果答案过短或明显表示失败
    if len(final_answer) < 5 or "无法确定" in final_answer:
        return "抱歉，我无法确定准确答案。建议您转人工客服。"
    # 对于其他情况（如 RAG 结果），直接返回（可后续加强验证）
    return final_answer


# ========== 主流程 ==========
def run_multi_agent(user_query: str) -> dict:
    shared_state["user_query"] = user_query

    tasks = planning_agent(user_query)
    shared_state["plan"] = tasks
    exec_results = execution_agent(tasks)
    shared_state["tool_results"] = exec_results
    preliminary = "\n".join(exec_results.values()) if exec_results else "无结果"
    final = validation_agent(user_query, preliminary)
    shared_state["final_answer"] = final

    return {
        "tasks": tasks,
        "exec_results": exec_results,
        "final_answer": final
    }


if __name__ == "__main__":
    print("企业智能客服 Agent 已启动（多 Agent 模式）")
    while True:
        q = input("\n用户: ")
        if q.lower() == 'quit':
            break
        ans = run_multi_agent(q)
        print(f"助手: {ans}")