"""
LangGraph 企业智能客服 Agent - 简化版
专注于展示 LangGraph 工作流架构
使用模拟工具，避免依赖兼容性问题

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
from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi
# from langchain.prompts import ChatPromptTemplate - removed due to version compatibility
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

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


# ========== 模拟 LLM（用于演示） ==========
def get_llm():
    """获取 LLM 实例"""
    # 使用模拟 LLM 避免 API 调用
    class MockLLM:
        def invoke(self, input_dict):
            class MockResponse:
                def __init__(self, content):
                    self.content = content
            # 模拟响应
            query = input_dict.get("query", "")
            if "重置密码" in query:
                return MockResponse('["knowledge_search: 重置密码"]')
            elif "TK-123456" in query:
                return MockResponse('["ticket_query: TK-123456"]')
            elif "今天" in query:
                return MockResponse('["date_query"]')
            elif "转人工" in query:
                return MockResponse('["escalate"]')
            elif "天气" in query:
                # 提取城市名，默认为北京
                city = "北京"
                if "上海" in query:
                    city = "上海"
                elif "广州" in query:
                    city = "广州"
                elif "深圳" in query:
                    city = "深圳"
                return MockResponse(f'["weather_query: {city}"]')
            elif "股票" in query:
                # 提取股票代码，默认为AAPL
                symbol = "AAPL"
                if "谷歌" in query or "GOOGL" in query:
                    symbol = "GOOGL"
                elif "特斯拉" in query or "TSLA" in query:
                    symbol = "TSLA"
                return MockResponse(f'["stock_query: {symbol}"]')
            else:
                return MockResponse('["knowledge_search: ' + query + '"]')

    return MockLLM()


# ========== 模拟工具 ==========
@tool
def knowledge_search(query: str) -> str:
    """模拟知识库检索"""
    mock_responses = {
        "重置密码": "您可以通过登录页面点击'忘记密码'链接重置密码。系统将发送重置邮件到您的注册邮箱。",
        "产品价格": "企业版产品价格为每年 10,000 元，包含所有功能和技术支持。",
        "技术支持": "技术支持时间为工作日 9:00-18:00，电话：400-123-4567。",
        "默认": "根据知识库信息，您的问题已记录，我们会尽快为您提供详细解答。"
    }
    for key, response in mock_responses.items():
        if key in query:
            return response
    return mock_responses["默认"]


@tool
def query_ticket_status(ticket_id: str) -> str:
    """模拟查询工单状态"""
    mock_status = {
        "TK-123456": "您的工单 TK-123456 已受理，正在处理中，预计48小时内完成。",
        "TK-789012": "工单 TK-789012 已处理完毕，请登录系统查看结果。",
        "default": "未找到工单信息，请确认工单号是否正确。"
    }
    return mock_status.get(ticket_id, mock_status["default"])


@tool
def escalate_to_human(query: str) -> str:
    """模拟转人工处理"""
    return "感谢您的耐心，我已将您的问题转接给人工客服，他们将尽快与您联系（预计5分钟内）。"


@tool
def get_current_date(query: str) -> str:
    """返回今天的日期"""
    return str(date.today())


@tool
def weather_query(city: str) -> str:
    """查询城市天气信息。支持北京、上海、广州、深圳等城市。"""
    weather_data = {
        "北京": "北京今天晴转多云，气温 15-25°C，北风2-3级。",
        "上海": "上海今天多云，气温 18-28°C，东南风3-4级。",
        "广州": "广州今天阵雨，气温 22-30°C，南风2-3级。",
        "深圳": "深圳今天晴，气温 23-32°C，南风2级。",
        "default": "该城市天气信息暂不可用，请稍后再试。"
    }
    return weather_data.get(city, weather_data["default"])


@tool
def stock_query(symbol: str) -> str:
    """查询股票实时价格。支持 AAPL、GOOGL、TSLA 等股票代码。"""
    stock_data = {
        "AAPL": "苹果公司 (AAPL) 当前价格 $175.20，今日上涨 2.3%。",
        "GOOGL": "谷歌 (GOOGL) 当前价格 $155.80，今日下跌 0.5%。",
        "TSLA": "特斯拉 (TSLA) 当前价格 $180.50，今日上涨 5.2%。",
        "default": "该股票代码信息暂不可用，请确认代码是否正确。"
    }
    return stock_data.get(symbol, stock_data["default"])


# ========== LangGraph 节点函数 ==========
def planning_node(state: AgentState) -> AgentState:
    """规划节点：分析用户查询，拆解为任务列表"""
    print(f"[规划节点] 处理查询: {state['user_query']}")

    # 使用模拟 LLM 生成规划
    llm = get_llm()
    response = llm.invoke({"query": state["user_query"]}).content

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
        elif task.startswith("weather_query:"):
            city = task.replace("weather_query:", "").strip()
            results[task] = weather_query.run(city)
        elif task.startswith("stock_query:"):
            symbol = task.replace("stock_query:", "").strip()
            results[task] = stock_query.run(symbol)
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
    trust_keywords = ["工单", "受理", "处理中", "已完成", "今天", "年-月-日", "转人工", "天气", "气温", "股票", "价格", "上涨", "下跌"]
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
    print("LangGraph 企业智能客服 Agent (简化版) 已启动")
    print("输入 'quit' 退出，'visualize' 生成工作流图，'test' 运行测试")

    # 生成工作流图
    visualize_workflow()

    while True:
        query = input("\n用户: ").strip()
        if query.lower() == 'quit':
            break
        elif query.lower() == 'visualize':
            visualize_workflow()
            continue
        elif query.lower() == 'test':
            test_agent()
            continue

        result = run_langgraph_agent(query)
        print(f"\n助手: {result['final_answer']}")
        print(f"\n[调试信息]")
        print(f"- 任务规划: {result['plan']}")
        print(f"- 迭代次数: {result['workflow_info']['iterations']}")
        print(f"- 答案质量: {result['workflow_info']['answer_quality']}")