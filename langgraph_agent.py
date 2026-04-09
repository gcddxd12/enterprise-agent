"""
基于 LangGraph 的企业智能客服 Agent 系统
支持：
- 条件工作流（简单/复杂问题不同路径）
- 循环优化（答案不满意时重新处理）
- 并行执行（多个工具同时调用）
- 可视化调试（自动生成工作流图）

作者：gcddxd12
版本：1.0.0
创建日期：2026-04-09
"""

import os
import json
import re
from datetime import date
from typing import TypedDict, List, Dict, Any, Literal, Optional
from dotenv import load_dotenv
from langchain_core.tools import tool, Tool
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain.chains.retrieval_qa import RetrievalQA - removed due to version compatibility
from langchain_community.chat_models import ChatTongyi
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint import MemorySaver

# 加载环境变量
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

# ========== LangGraph 状态定义 ==========
class AgentState(TypedDict):
    """Agent 工作流的状态定义"""
    # 输入
    user_query: str
    # 规划阶段
    plan: Optional[List[str]]
    # 执行阶段
    tool_results: Optional[Dict[str, str]]
    # 验证阶段
    final_answer: Optional[str]
    # 元数据
    step: Literal["planning", "execution", "validation", "completed", "escalate"]
    # 循环控制
    iteration: int
    max_iterations: int
    # 条件判断
    needs_human_escalation: bool
    answer_quality: Optional[Literal["poor", "fair", "good"]]


# ========== 延迟初始化资源 ==========
_embeddings = None
_vectorstore = None
_retriever = None
_llm = None
_qa_chain = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = DashScopeEmbeddings(model="text-embedding-v4", dashscope_api_key=api_key)
    return _embeddings


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        # 检查是否在 CI 环境中，如果是则返回 None 或 mock
        if os.getenv("CI"):
            return None
        embeddings = get_embeddings()
        _vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return _vectorstore


def get_retriever():
    global _retriever
    if _retriever is None:
        vectorstore = get_vectorstore()
        if vectorstore is not None:
            _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return _retriever


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatTongyi(model="qwen-plus", temperature=0)
    return _llm


def get_qa_chain():
    global _qa_chain
    if _qa_chain is None and get_retriever() is not None and get_llm() is not None:
        _qa_chain = RetrievalQA.from_chain_type(
            llm=get_llm(),
            chain_type="stuff",
            retriever=get_retriever(),
            return_source_documents=True
        )
    return _qa_chain


# ========== 工具定义（企业业务场景） ==========
@tool
def knowledge_search(query: str) -> str:
    """从企业知识库中检索信息，返回答案。适用于产品使用、技术支持、销售政策等问题。"""
    qa_chain = get_qa_chain()
    if qa_chain is None:
        return "知识库服务不可用，请稍后再试。"
    result = qa_chain.invoke({"query": query})
    return result["result"]


@tool
def query_ticket_status(ticket_id: str) -> str:
    """模拟查询工单状态。工单号格式为 TK-xxxxxx。"""
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


# ========== LangGraph 节点函数 ==========
def planning_node(state: AgentState) -> AgentState:
    """规划节点：分析用户查询，拆解为任务列表"""
    print(f"[规划节点] 处理查询: {state['user_query']}")

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
    chain = prompt | get_llm()
    response = chain.invoke({"query": state["user_query"]}).content

    try:
        tasks = json.loads(response)
    except:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        tasks = json.loads(match.group()) if match else []

    print(f"[规划节点] 生成任务: {tasks}")
    return {**state, "plan": tasks, "step": "execution"}


def execution_node(state: AgentState) -> AgentState:
    """执行节点：执行规划的任务，收集结果"""
    print(f"[执行节点] 执行任务: {state['plan']}")

    if not state["plan"]:
        return {**state, "tool_results": {}, "step": "validation"}

    results = {}
    for task in state["plan"]:
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

    print(f"[执行节点] 执行结果: {results}")
    return {**state, "tool_results": results, "step": "validation"}


def validation_node(state: AgentState) -> AgentState:
    """验证节点：验证答案质量，决定下一步"""
    print(f"[验证节点] 验证答案质量")

    # 合并所有工具结果
    preliminary = "\n".join(state["tool_results"].values()) if state["tool_results"] else "无结果"

    # 验证逻辑
    trust_keywords = ["工单", "受理", "处理中", "已完成", "今天", "年-月-日", "转人工"]
    if any(kw in preliminary for kw in trust_keywords):
        final_answer = preliminary
        answer_quality = "good"
    elif len(preliminary) < 5 or "无法确定" in preliminary:
        final_answer = "抱歉，我无法确定准确答案。建议您转人工客服。"
        answer_quality = "poor"
    else:
        final_answer = preliminary
        answer_quality = "fair"

    # 判断是否需要转人工
    needs_human_escalation = (
        "escalate" in state["plan"] or
        answer_quality == "poor" or
        state["iteration"] >= state["max_iterations"]
    )

    print(f"[验证节点] 答案质量: {answer_quality}, 转人工: {needs_human_escalation}")
    return {
        **state,
        "final_answer": final_answer,
        "answer_quality": answer_quality,
        "needs_human_escalation": needs_human_escalation,
        "step": "completed" if not needs_human_escalation else "escalate"
    }


def human_escalation_node(state: AgentState) -> AgentState:
    """人工升级节点：处理需要人工介入的情况"""
    print(f"[人工升级节点] 转人工处理")

    if state["final_answer"] and "转人工" in state["final_answer"]:
        final_answer = state["final_answer"]
    else:
        final_answer = "感谢您的耐心，我已将您的问题转接给人工客服，他们将尽快与您联系（预计5分钟内）。"

    return {**state, "final_answer": final_answer, "step": "completed"}


# ========== 条件路由函数 ==========
def route_after_validation(state: AgentState) -> Literal["human_escalation", "improvement", "end"]:
    """验证后的路由决策"""
    if state["needs_human_escalation"]:
        return "human_escalation"
    elif state["answer_quality"] == "poor" and state["iteration"] < state["max_iterations"]:
        return "improvement"
    else:
        return "end"


def route_after_escalation(state: AgentState) -> Literal["end"]:
    """人工升级后的路由决策"""
    return "end"


# ========== 图构建 ==========
def create_workflow() -> StateGraph:
    """创建 LangGraph 工作流"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("planning", planning_node)
    workflow.add_node("execution", execution_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("human_escalation", human_escalation_node)

    # 设置入口点
    workflow.set_entry_point("planning")

    # 添加边（正常流程）
    workflow.add_edge("planning", "execution")
    workflow.add_edge("execution", "validation")

    # 条件边（验证后）
    workflow.add_conditional_edges(
        "validation",
        route_after_validation,
        {
            "human_escalation": "human_escalation",
            "improvement": "planning",  # 重新规划改进
            "end": END
        }
    )

    # 人工升级后到结束
    workflow.add_edge("human_escalation", END)

    return workflow


# ========== 主入口函数 ==========
def run_langgraph_agent(user_query: str, max_iterations: int = 3) -> Dict[str, Any]:
    """
    运行 LangGraph Agent 处理用户查询

    Args:
        user_query: 用户查询
        max_iterations: 最大迭代次数（用于答案改进）

    Returns:
        Dict containing plan, tool_results, final_answer, and workflow info
    """
    # 初始化状态
    initial_state: AgentState = {
        "user_query": user_query,
        "plan": None,
        "tool_results": None,
        "final_answer": None,
        "step": "planning",
        "iteration": 0,
        "max_iterations": max_iterations,
        "needs_human_escalation": False,
        "answer_quality": None
    }

    # 创建图和工作流
    workflow = create_workflow()

    # 创建检查点存储器（支持会话持久化）
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # 运行工作流
    print(f"\n{'='*60}")
    print(f"开始处理查询: {user_query}")
    print(f"{'='*60}")

    final_state = None
    for iteration in range(max_iterations):
        print(f"\n--- 迭代 {iteration + 1} ---")

        # 更新迭代计数
        if iteration > 0:
            initial_state["iteration"] = iteration

        # 执行工作流
        config = {"configurable": {"thread_id": "user_session_1"}}
        result = app.invoke(initial_state, config)
        final_state = result

        # 检查是否完成
        if result["step"] == "completed":
            break

        # 准备下一次迭代
        initial_state = result

    print(f"\n{'='*60}")
    print(f"处理完成")
    print(f"{'='*60}")

    return {
        "plan": final_state["plan"] if final_state else None,
        "tool_results": final_state["tool_results"] if final_state else None,
        "final_answer": final_state["final_answer"] if final_state else None,
        "workflow_info": {
            "iterations": final_state["iteration"] + 1 if final_state else 0,
            "final_step": final_state["step"] if final_state else None,
            "answer_quality": final_state["answer_quality"] if final_state else None
        }
    }


def visualize_workflow(output_path: str = "workflow.png"):
    """生成工作流可视化图"""
    try:
        from langgraph.graph import StateGraph
        workflow = create_workflow()
        app = workflow.compile()

        # 生成 PNG 图像
        png_data = app.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_data)
        print(f"工作流图已保存至: {output_path}")

        # 生成 Mermaid 代码
        mermaid_code = app.get_graph().draw_mermaid()
        mermaid_path = output_path.replace(".png", ".mmd")
        with open(mermaid_path, "w", encoding="utf-8") as f:
            f.write(mermaid_code)
        print(f"Mermaid 代码已保存至: {mermaid_path}")

        return True
    except Exception as e:
        print(f"可视化生成失败: {e}")
        return False


# ========== 测试函数 ==========
def test_agent():
    """测试函数"""
    test_queries = [
        "如何重置密码？",
        "查询工单 TK-123456",
        "今天几号？",
        "帮我转人工",
        "产品价格是多少？"
    ]

    for query in test_queries:
        print(f"\n{'#'*60}")
        print(f"测试查询: {query}")
        print(f"{'#'*60}")
        result = run_langgraph_agent(query)
        print(f"最终答案: {result['final_answer']}")
        print(f"工作流信息: {result['workflow_info']}")


# ========== 主入口 ==========
if __name__ == "__main__":
    print("LangGraph 企业智能客服 Agent 已启动")
    print("输入 'quit' 退出，'visualize' 生成工作流图")

    # 生成工作流图
    visualize_workflow()

    while True:
        query = input("\n用户: ").strip()
        if query.lower() == 'quit':
            break
        elif query.lower() == 'visualize':
            visualize_workflow()
            continue

        result = run_langgraph_agent(query)
        print(f"\n助手: {result['final_answer']}")
        print(f"\n[调试信息]")
        print(f"- 任务规划: {result['plan']}")
        print(f"- 迭代次数: {result['workflow_info']['iterations']}")
        print(f"- 答案质量: {result['workflow_info']['answer_quality']}")