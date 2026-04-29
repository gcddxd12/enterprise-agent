"""
LangGraph 企业智能客服 Agent - 带记忆版本
在简化版基础上添加对话记忆机制

记忆功能：
1. 短期记忆：最近对话轮次
2. 用户偏好记忆：语言风格、信息详细程度
3. 对话历史持久化：支持跨会话记忆

作者：gcddxd12
版本：1.1.0
创建日期：2026-04-09
"""

import os
import json
from datetime import date, datetime
from typing import TypedDict, List, Dict, Any, Literal, Optional, Annotated
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# 尝试导入高级RAG系统
try:
    from advanced_rag_system import create_advanced_rag_system
    ADVANCED_RAG_AVAILABLE = True
except ImportError as e:
    print(f"警告：高级RAG系统导入失败，将使用模拟模式: {e}")
    ADVANCED_RAG_AVAILABLE = False

# 尝试导入监控系统
try:
    from monitoring_system import get_monitoring_system
    MONITORING_AVAILABLE = True

    # 获取监控系统实例
    monitoring_system = None
    try:
        monitoring_system = get_monitoring_system()
        print("[INFO] 监控系统已初始化")
    except Exception as e:
        print(f"[WARN] 监控系统初始化失败: {e}")
        MONITORING_AVAILABLE = False
except ImportError as e:
    print(f"警告：监控系统导入失败，监控功能将不可用: {e}")
    MONITORING_AVAILABLE = False
    monitoring_system = None

# 加载环境变量
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")


# ========== AgentState 定义（标准 ReAct Agent） ==========
class AgentState(TypedDict):
    """标准 ReAct Agent 的工作流状态"""
    user_query: str
    messages: Annotated[list, add_messages]
    final_answer: Optional[str]
    raw_context: Optional[str]  # RAG 原始检索上下文（展示用）
    tool_results: Optional[Dict[str, str]]  # 工具调用记录
    plan: Optional[List[str]]  # 已执行的工具列表
    step: Literal["agent", "completed"]
    iteration: int
    max_iterations: int
    conversation_summary: Optional[str]
    user_preferences: Dict[str, Any]
    tracking_info: Optional[Dict[str, Any]]
    active_skills: Optional[List[str]]  # 已激活的skill名称列表
    skill_context: Optional[str]  # 合并后的skill指令文本


# ========== 记忆管理器 ==========
class MemoryManager:
    """管理对话记忆"""

    def __init__(self):
        self.conversation_history = []
        self.user_preferences = {
            "language_style": "neutral",  # neutral, formal, casual
            "detail_level": "moderate",   # brief, moderate, detailed
            "frequent_topics": set(),
            "last_interaction": None
        }

    def add_message(self, role: str, content: str):
        """添加消息到对话历史"""
        timestamp = datetime.now().isoformat()
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        self.conversation_history.append(message)

        # 保持最近10轮对话
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    def get_recent_history(self, max_messages: int = 5) -> List[Dict[str, str]]:
        """获取最近的对话历史"""
        return self.conversation_history[-max_messages:] if self.conversation_history else []

    def generate_summary(self) -> str:
        """生成对话摘要"""
        if not self.conversation_history:
            return "暂无对话历史"

        # 简单摘要：统计话题和轮次
        user_messages = [msg for msg in self.conversation_history if msg["role"] == "user"]
        assistant_messages = [msg for msg in self.conversation_history if msg["role"] == "assistant"]

        topics = set()
        for msg in user_messages:
            content = msg["content"].lower()
            if "套餐" in content or "资费" in content:
                topics.add("套餐查询")
            elif "流量" in content:
                topics.add("流量查询")
            elif "话费" in content or "账单" in content:
                topics.add("话费账单")
            elif "宽带" in content or "光纤" in content:
                topics.add("宽带业务")
            elif "工单" in content:
                topics.add("工单查询")
            elif "天气" in content:
                topics.add("天气查询")
            elif "股票" in content:
                topics.add("股票查询")
            elif "5G" in content or "信号" in content:
                topics.add("5G业务")
            elif "物联网" in content or "IoT" in content:
                topics.add("物联网")
            elif "图片" in content or "图像" in content or "照片" in content:
                topics.add("图像处理")
            elif "文档" in content or "pdf" in content or "word" in content or "excel" in content:
                topics.add("文档处理")
            elif "上传" in content or "文件" in content:
                topics.add("文件上传")

        summary = f"对话历史：{len(self.conversation_history)}条消息（用户：{len(user_messages)}，助手：{len(assistant_messages)}）"
        if topics:
            summary += f"。涉及话题：{', '.join(topics)}"

        return summary

    def update_preferences(self, user_query: str, assistant_response: str):
        """基于对话更新用户偏好"""
        # 分析语言风格
        if "请" in user_query or "谢谢" in user_query or "您好" in user_query:
            self.user_preferences["language_style"] = "formal"
        elif "哈喽" in user_query or "哈哈" in user_query:
            self.user_preferences["language_style"] = "casual"

        # 分析详细程度偏好
        response_length = len(assistant_response)
        if response_length < 50:
            self.user_preferences["detail_level"] = "brief"
        elif response_length > 200:
            self.user_preferences["detail_level"] = "detailed"

        # 更新最后交互时间
        self.user_preferences["last_interaction"] = datetime.now().isoformat()

    def adapt_response(self, response: str) -> str:
        """根据用户偏好调整响应"""
        style = self.user_preferences.get("language_style", "neutral")
        detail = self.user_preferences.get("detail_level", "moderate")

        # 调整语言风格
        if style == "formal":
            response = "尊敬的客户，" + response
        elif style == "casual":
            if not response.startswith("哈喽") and not response.startswith("你好"):
                response = "哈喽！" + response

        # 调整详细程度
        if detail == "brief" and len(response) > 100:
            # 简化长响应
            sentences = response.split('。')
            if len(sentences) > 2:
                response = '。'.join(sentences[:2]) + '。'
        elif detail == "detailed" and len(response) < 100:
            # 添加更多细节
            response = response + " 如果您需要更详细的信息，请随时告诉我。"

        return response


# ========== 全局记忆管理器 ==========
_memory_manager = None

def get_memory_manager():
    """获取记忆管理器实例"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


# ========== LLM 实例（真实模型） ==========
_llm = None

def get_llm():
    """获取 LLM 实例（单例模式，使用阿里百炼 qwen-plus 模型）"""
    global _llm
    if _llm is None:
        _llm = ChatTongyi(
            model="qwen-plus",
            temperature=0,
            dashscope_api_key=api_key
        )
    return _llm


# ========== 高级RAG系统初始化 ==========
advanced_rag_retriever = None

def init_advanced_rag():
    """初始化高级RAG检索器"""
    global advanced_rag_retriever

    if not ADVANCED_RAG_AVAILABLE:
        print("[INFO] 高级RAG系统不可用，使用模拟模式")
        return None

    try:
        # 尝试从向量数据库创建检索器
        chroma_db_path = "./chroma_db"
        if os.path.exists(chroma_db_path):
            print("[INFO] 加载向量数据库...")
            from langchain_community.embeddings import DashScopeEmbeddings
            from langchain_community.vectorstores import Chroma
            from langchain_core.documents import Document

            embeddings = DashScopeEmbeddings(
                model="text-embedding-v4",
                dashscope_api_key=api_key
            )
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=chroma_db_path
            )

            # 从向量数据库获取文档用于BM25
            documents_for_bm25 = []
            try:
                # 获取所有文档（最多100个）
                collection = vectorstore._collection
                if collection:
                    results = collection.get(limit=100)
                    if results and 'documents' in results:
                        for i, doc_text in enumerate(results['documents']):
                            metadata = {}
                            if 'metadatas' in results and i < len(results['metadatas']):
                                metadata = results['metadatas'][i] or {}

                            # 创建Document对象
                            document = Document(
                                page_content=doc_text,
                                metadata=metadata
                            )
                            documents_for_bm25.append(document)

                        print(f"[INFO] 从向量数据库加载了 {len(documents_for_bm25)} 个文档用于BM25检索")
            except Exception as e:
                print(f"[WARN] 无法从向量数据库获取文档: {e}")
                # 如果无法获取文档，使用一些默认文档
                documents_for_bm25 = [
                    Document(
                        page_content="中国移动5G-A规模化商用采用低频广覆盖、中频容量承载、高频超高速补充的三层频谱组网，融合超级上行、通感一体、RedCap、无源物联增强技术。",
                        metadata={"source": "cmcc_5g", "topic": "5G"}
                    ),
                    Document(
                        page_content="中国移动移动云构建全域分布式云架构，包含中心云、区域云、边缘云三级节点，提供专属云、私有云、混合云、本地容灾、多云管理全栈能力。",
                        metadata={"source": "cmcc_cloud", "topic": "cloud"}
                    ),
                    Document(
                        page_content="中国移动算力网络以连接+算力+能力三位一体为核心架构，依托全国八大国家级算力枢纽节点布局，实现跨域算力调度与动态负载均衡。",
                        metadata={"source": "cmcc_computing", "topic": "computing"}
                    )
                ]

            # 创建高级RAG系统
            config = {
                "use_cache": True,
                "query_expansion": True,
                "vector_k": 10,
                "bm25_k": 10,
                "cache_dir": "./vector_cache"
            }
            advanced_rag_retriever = create_advanced_rag_system(
                vectorstore=vectorstore,
                documents=documents_for_bm25,  # 提供文档给BM25
                llm=get_llm(),
                use_mock=False,  # 使用真实LLM进行查询扩展
                config=config
            )
            print("[INFO] 高级RAG系统初始化成功")
            return advanced_rag_retriever
        else:
            print("[WARN] 向量数据库不存在，使用模拟模式")
            return None
    except Exception as e:
        print(f"[ERROR] 高级RAG系统初始化失败: {e}")
        return None

# 尝试初始化高级RAG系统
init_advanced_rag()


# ========== 模拟工具 ==========
@tool
def knowledge_search(query: str) -> str:
    """从中国移动知识库中检索信息，返回答案。适用于套餐资费、5G业务、宽带、物联网、
    云计算、算力网络、网络安全、政企服务等中国移动相关业务咨询。"""
    if not query or not isinstance(query, str) or not query.strip():
        return "错误：请提供有效的查询内容"
    query = query.strip()
    import time
    start_time = time.time()

    memory_manager = get_memory_manager()
    global advanced_rag_retriever

    # 监控跟踪：RAG检索开始
    if MONITORING_AVAILABLE and monitoring_system:
        try:
            monitoring_system.track_rag_retrieval(query, 0, 0, success=True, started=True)
        except Exception as e:
            print(f"[WARN] RAG检索监控跟踪失败: {e}")

    # 如果有高级RAG检索器，使用它
    if advanced_rag_retriever:
        try:
            print(f"[INFO] 使用高级RAG检索查询: '{query}'")

            # 使用高级RAG检索器获取文档
            results = advanced_rag_retriever.retrieve(query)
            duration = time.time() - start_time

            if results and len(results) > 0:
                # 取 top 3 条结果（原始内容带元数据，展示用）
                top_results = results[:3]
                display_parts = []
                clean_parts = []
                for doc in top_results:
                    content = doc.page_content
                    source = doc.metadata.get('source', '知识库')
                    score = doc.metadata.get('final_score', 0) or doc.metadata.get('relevance_score', 0)
                    # 显示版本（带元数据）
                    display_parts.append(f"[{source}] (相关度:{score:.2f}) {content}")
                    # 清洁版本（只保留纯文本内容，给 LLM 用）
                    clean_parts.append(content)

                response = "\n\n".join(clean_parts)

                # 统计信息
                if hasattr(advanced_rag_retriever, 'cache_manager'):
                    try:
                        cache_stats = advanced_rag_retriever.cache_manager.get_stats()
                        if cache_stats['result_hits'] > 0 or cache_stats['embedding_hits'] > 0:
                            response += f"\n\n[检索共找到 {len(top_results)} 条相关信息（缓存命中 {cache_stats['result_hits'] + cache_stats['embedding_hits']} 次）]"
                    except Exception:
                        pass

                adapted_response = memory_manager.adapt_response(response)

                # 监控跟踪：RAG检索成功
                if MONITORING_AVAILABLE and monitoring_system:
                    try:
                        monitoring_system.track_rag_retrieval(query, duration, len(results), success=True)
                    except Exception as e:
                        print(f"[WARN] RAG检索成功监控跟踪失败: {e}")

                return adapted_response
            else:
                print("[WARN] 高级RAG检索到0个结果，回退到模拟模式")
                # 监控跟踪：RAG检索无结果
                if MONITORING_AVAILABLE and monitoring_system:
                    try:
                        monitoring_system.track_rag_retrieval(query, duration, 0, success=False, error="No results found")
                    except Exception as e:
                        print(f"[WARN] RAG检索无结果监控跟踪失败: {e}")
        except Exception as e:
            duration = time.time() - start_time
            print(f"[ERROR] 高级RAG检索失败: {e}，回退到模拟模式")
            # 监控跟踪：RAG检索失败
            if MONITORING_AVAILABLE and monitoring_system:
                try:
                    monitoring_system.track_rag_retrieval(query, duration, 0, success=False, error=str(e)[:100])
                except Exception as e2:
                    print(f"[WARN] RAG检索失败监控跟踪失败: {e2}")

    # 回退到内存知识库仓储
    print(f"[INFO] 使用内存知识库检索查询: '{query}'")
    try:
        from repositories import get_knowledge_repo
        repo = get_knowledge_repo()
        results = repo.search(query, top_k=3)
        if results:
            parts = [f"[{r['category']}] {r['content']}" for r in results]
            response = "\n\n".join(parts)
            adapted_response = memory_manager.adapt_response(response)
            return adapted_response
    except Exception as e:
        print(f"[WARN] 知识库仓储检索失败: {e}")

    default_response = "根据中国移动知识库信息，您的问题已记录，我们将尽快为您提供详细解答。如需人工服务，可拨打10086。"
    adapted_response = memory_manager.adapt_response(default_response)
    return adapted_response


@tool
def query_ticket_status(ticket_id: str) -> str:
    """查询工单处理状态。输入工单号（如TK-123456），返回当前处理进度。"""
    if not ticket_id or not isinstance(ticket_id, str) or not ticket_id.strip():
        return "错误：请提供有效的工单号（如 TK-123456）"
    ticket_id = ticket_id.strip()

    try:
        from repositories import get_ticket_repo
        repo = get_ticket_repo()
        ticket = repo.get_status(ticket_id)
        if ticket:
            memory_manager = get_memory_manager()
            response = (
                f"工单 {ticket['ticket_id']}：{ticket['type']}\n"
                f"状态：{ticket['status']}  |  优先级：{ticket['priority']}  |  处理人：{ticket['handler']}\n"
                f"详情：{ticket['detail']}\n"
                f"创建时间：{ticket['created_at']}"
            )
            return memory_manager.adapt_response(response)
        else:
            return f"未找到工单「{ticket_id}」的信息，请确认工单号是否正确。可尝试提供手机号查询关联工单。"
    except Exception as e:
        return f"工单查询服务暂不可用: {e}"


@tool
def escalate_to_human(query: str) -> str:
    """将用户问题转接给人工客服处理。适用于投诉升级、复杂业务办理、敏感问题等场景。"""
    memory_manager = get_memory_manager()
    try:
        from repositories import get_escalation_repo
        repo = get_escalation_repo()
        result = repo.escalate(query, priority="normal")
        response = result.get("message", "已为您转接人工客服，请稍候。")
        return memory_manager.adapt_response(response)
    except Exception:
        response = "感谢您的耐心，我已将您的问题转接给人工客服，他们将尽快与您联系（预计5分钟内）。"
        return memory_manager.adapt_response(response)


@tool
def get_current_date(query: str) -> str:
    """返回今天的日期"""
    memory_manager = get_memory_manager()
    response = f"今天是 {date.today()}。"
    adapted_response = memory_manager.adapt_response(response)
    return adapted_response


AGENT_TOOLS = []  # 延迟初始化

def get_tools():
    """获取 Agent 工具列表（延迟初始化）"""
    global AGENT_TOOLS
    if not AGENT_TOOLS:
        from skill_manager import create_use_skill_tool
        AGENT_TOOLS = [
            knowledge_search,
            query_ticket_status,
            escalate_to_human,
            get_current_date,
            create_use_skill_tool(),
        ]
        # 集成 MCP 外部工具
        try:
            from mcp_client import init_mcp_tools
            mcp_tools = init_mcp_tools()
            if mcp_tools:
                AGENT_TOOLS.extend(mcp_tools)
        except Exception as e:
            print(f"[MCP] 加载 MCP 工具失败: {e}")
    return AGENT_TOOLS


# ========== 系统提示词 ==========
BASE_SYSTEM_PROMPT = """你是一名中国移动智能客服助手。你的职责是帮助用户解答中国移动业务相关问题。

## 你的工具
- **knowledge_search**: 从中国移动知识库检索信息。参数 query 请使用简洁关键词（如"5G套餐资费"、"宽带故障报修"、"流量包订购"），不要传完整长句。
  适用场景：套餐资费、5G业务、宽带、流量、话费、信号、携号转网、国际漫游、积分、营业厅等业务咨询。
- **query_ticket_status**: 查询工单状态。工单号格式为 TK-xxxxxx。
- **escalate_to_human**: 当你无法回答用户问题、用户情绪激动、或用户明确要求转人工时，调用此工具。
- **get_current_date**: 查询今天的日期。
- **use_skill**: 当你需要特定领域的专业知识或处理流程时，加载对应的技能指令。可用技能请参考系统提示中"当前激活的专业技能"部分。

## 行为准则
1. 对于业务咨询，先调用 knowledge_search 检索（用关键词），再基于检索结果回答
2. 如果用户问题涉及到你已激活的专业技能领域，严格遵守技能中定义的处理流程和话术
3. 用亲切、专业的语气回复，称呼用户为"您"
4. 只基于检索结果回答，不要编造没有的信息（套餐价格、流量额度等必须来源于检索结果）
5. 如果检索结果包含了用户问的信息，整理成清晰的结构回复
6. 如果检索结果不够或无法回答，告知用户并建议拨打10086或转人工
7. 结尾可以引导用户进一步提问"""


def build_system_prompt(active_skills: Optional[List[str]] = None) -> str:
    """构建系统提示：基础提示 + 可用skill摘要 + 已激活skill的完整指令"""
    prompt = BASE_SYSTEM_PROMPT

    # 添加可用skill摘要（供LLM参考，知道何时调用 use_skill）
    try:
        from skill_manager import get_skill_manager
        skill_manager = get_skill_manager()
        all_skills = skill_manager.list_skills()
        if all_skills:
            prompt += "\n\n## 可用的专业技能（通过 use_skill 加载）\n"
            for s in all_skills:
                triggers_str = "、".join(s["triggers"][:5])
                prompt += f"- **{s['name']}**: {s['description']}（触发词: {triggers_str}）\n"
    except Exception:
        pass

    # 添加已激活skill的完整指令
    if active_skills:
        prompt += "\n\n## 当前激活的专业技能（请严格遵守以下指令）\n"
        for skill_name in active_skills:
            skill = skill_manager.get_skill(skill_name)
            if skill:
                prompt += f"\n### {skill.name}\n{skill.content}\n"

    # 添加 MCP 工具描述
    try:
        from mcp_client import get_mcp_manager
        mcp_mgr = get_mcp_manager()
        if mcp_mgr:
            mcp_desc = mcp_mgr.get_tool_descriptions()
            if mcp_desc:
                prompt += mcp_desc
    except Exception:
        pass

    return prompt

MAX_AGENT_STEPS = 5  # ReAct 循环最大步数


def _use_skill_executor(tool_args) -> str:
    """use_skill 工具的执行器包装"""
    from skill_manager import get_skill_manager
    skill_manager = get_skill_manager()
    skill_name = tool_args.get("skill_name", "") if isinstance(tool_args, dict) else str(tool_args)
    skill = skill_manager.get_skill(skill_name)
    if skill:
        return f"已加载技能「{skill.name}」:\n\n{skill.content}"
    available = [s["name"] for s in skill_manager.list_skills()]
    return f"未找到技能「{skill_name}」。可用技能: {', '.join(available)}"


# ========== Agent 节点（标准 ReAct 模式） ==========
def agent_node(state: AgentState) -> AgentState:
    """标准 ReAct Agent 节点：LLM 自主决定工具调用，观察结果，迭代推理"""
    import time
    start_time = time.time()
    print("[Agent] 开始 ReAct 推理循环")

    llm = get_llm()
    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)

    # 构建消息列表：动态系统提示 + 对话历史 + 用户问题
    active_skills = state.get("active_skills", [])
    system_prompt = build_system_prompt(active_skills)
    messages = [SystemMessage(content=system_prompt)]

    # 注入记忆上下文（如有）
    conversation_summary = state.get("conversation_summary", "")
    if conversation_summary:
        messages.append(SystemMessage(content=f"对话历史摘要：{conversation_summary}"))

    # 注入skill上下文（如有）
    skill_context = state.get("skill_context", "")
    if skill_context:
        messages.append(SystemMessage(content=f"专业技能参考：\n{skill_context}"))

    messages.append(HumanMessage(content=state["user_query"]))
    print(f"[Agent] 初始消息数: {len(messages)}")

    tool_results = {}
    plan = []
    raw_context = ""

    # ReAct 循环
    for step_idx in range(MAX_AGENT_STEPS):
        print(f"[Agent] ReAct 第 {step_idx + 1} 步...")
        step_start = time.time()

        response = llm_with_tools.invoke(messages)
        messages.append(response)
        step_duration = time.time() - step_start

        if not response.tool_calls:
            # LLM 认为不需要更多工具 → 这是最终回答
            final_answer = response.content or ""
            print(f"[Agent] LLM 完成（无工具调用），答案长度: {len(final_answer)}, 耗时: {step_duration:.2f}s")

            duration = time.time() - start_time
            print(f"[Agent] ReAct 完成，共 {step_idx + 1} 步，总耗时: {duration:.2f}s")
            print(f"[Agent] 调用工具: {plan}")
            return {
                **state,
                "final_answer": final_answer,
                "raw_context": raw_context,
                "tool_results": tool_results,
                "plan": plan,
                "step": "completed"
            }

        # 执行工具调用
        tool_names = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_id = tool_call["id"]
            tool_names.append(tool_name)

            print(f"[Agent] 调用工具: {tool_name}, 参数: {str(tool_args)[:80]}")

            # 提取查询参数
            query_str = str(tool_args.get("query", list(tool_args.values())[0] if tool_args.values() else ""))
            plan.append(f"{tool_name}: {query_str}")

            # 执行工具（字典分发：本地工具 + MCP 工具）
            tool_executors = {
                "knowledge_search": lambda a: knowledge_search.run(a if isinstance(a, str) else str(a.get("query", list(a.values())[0] if a.values() else ""))),
                "query_ticket_status": lambda a: query_ticket_status.run(a.get("ticket_id", query_str) if isinstance(a, dict) else str(a)),
                "escalate_to_human": lambda a: escalate_to_human.run(query_str or ""),
                "get_current_date": lambda a: get_current_date.run(query_str or ""),
                "use_skill": lambda a: _use_skill_executor(a),
            }
            # 动态添加 MCP 工具执行器
            try:
                from mcp_client import get_mcp_manager
                mcp_mgr = get_mcp_manager()
                if mcp_mgr:
                    for info in mcp_mgr.list_tools():
                        # MCP call_tool 接受 dict，处理 string 回退
                        tool_executors[info.name] = (
                            lambda a, name=info.name: mcp_mgr.call_tool(
                                name, a if isinstance(a, dict) else {"query": str(a)}
                            )
                        )
            except Exception:
                pass

            try:
                executor = tool_executors.get(tool_name)
                if executor:
                    result = executor(tool_args)
                    if tool_name == "knowledge_search" and raw_context == "":
                        raw_context = result  # 保留第一次检索结果
                else:
                    result = f"未知工具: {tool_name}"
            except Exception as e:
                result = f"工具执行失败: {e}"
                print(f"[Agent] 工具 {tool_name} 执行异常: {e}")

            tool_results[f"{tool_name}: {query_str[:50]}"] = result[:500]
            messages.append(ToolMessage(content=result, tool_call_id=tool_id))

        print(f"[Agent] 第 {step_idx + 1} 步完成，调用: {tool_names}，耗时: {step_duration:.2f}s")

    # 达到最大步数，强制输出
    print(f"[Agent] 达到最大步数 {MAX_AGENT_STEPS}，强制输出最终回复")
    llm_direct = get_llm()
    force_prompt = f"""你是一名中国移动智能客服。已完成知识检索，请根据以下信息为用户问题生成回复。

用户问题：{state['user_query']}

检索结果：
{raw_context if raw_context else '无检索结果'}

请直接输出客服回复："""
    try:
        final_answer = llm_direct.invoke([HumanMessage(content=force_prompt)]).content
    except Exception:
        final_answer = "抱歉，处理您的问题需要更多时间，建议您拨打10086热线或前往就近营业厅咨询。"

    duration = time.time() - start_time
    print(f"[Agent] ReAct 完成（达最大步数），总耗时: {duration:.2f}s")
    return {
        **state,
        "final_answer": final_answer,
        "raw_context": raw_context,
        "tool_results": tool_results,
        "plan": plan,
        "step": "completed"
    }


# ========== LangGraph 节点函数 ==========
def preprocess_node(state: AgentState) -> AgentState:
    """预处理节点：处理用户输入，更新记忆"""
    print(f"[预处理节点] 处理查询: {state['user_query']}")

    memory_manager = get_memory_manager()

    # 添加用户消息到记忆
    memory_manager.add_message("user", state["user_query"])

    # 生成对话摘要
    conversation_summary = memory_manager.generate_summary()

    # 监控跟踪：工作流开始
    tracking_info = {}
    if MONITORING_AVAILABLE and monitoring_system:
        try:
            tracking_info = monitoring_system.track_workflow_start(
                workflow_name="enterprise_agent_workflow",
                inputs={
                    "user_query": state['user_query'],
                    "conversation_summary": conversation_summary
                },
                metadata={
                    "iteration": state.get('iteration', 0),
                    "max_iterations": state.get('max_iterations', 3)
                }
            )
        except Exception as e:
            print(f"[WARN] 监控跟踪失败: {e}")

    print(f"[预处理节点] 对话摘要: {conversation_summary}")

    # Skill匹配：基于关键词自动激活相关领域技能
    active_skills = []
    skill_context = None
    try:
        from skill_manager import get_skill_manager
        skill_manager = get_skill_manager()
        matching_skills = skill_manager.find_matching_skills(state["user_query"], max_skills=2)
        if matching_skills:
            active_skills = [s.name for s in matching_skills]
            skill_context = "\n\n".join([s.content for s in matching_skills])
            print(f"[预处理节点] 激活Skill: {active_skills}")
    except Exception as e:
        print(f"[预处理节点] Skill匹配失败: {e}")

    return {
        **state,
        "conversation_summary": conversation_summary,
        "active_skills": active_skills,
        "skill_context": skill_context,
        "step": "planning",
        "tracking_info": tracking_info,
    }


def postprocess_node(state: AgentState) -> AgentState:
    """后处理节点：LLM整理工具结果 + 更新记忆 + 生成最终响应"""
    print("[后处理节点] 处理最终答案")

    memory_manager = get_memory_manager()
    raw_answer = state.get("final_answer") or "抱歉，处理您的问题时遇到错误，请稍后再试。"
    tool_results = state.get("tool_results") or {}
    raw_context = state.get("raw_context") or ""

    # LLM 整理：有工具调用结果时，让 LLM 整理为自然客服回复
    if tool_results and raw_answer:
        try:
            synthesis_prompt = f"""你是一名中国移动智能客服。请根据检索结果和工具调用结果，生成一段自然友好的客服回复。

用户问题：{state['user_query']}

检索上下文：
{raw_context[:1000] if raw_context else '无'}

工具结果摘要：
{chr(10).join(f'- {k}: {str(v)[:200]}' for k, v in list(tool_results.items())[:5])}

请直接输出面向用户的客服回复（自然语言，语气友好专业，不超过300字）："""
            llm = get_llm()
            synthesized = llm.invoke([HumanMessage(content=synthesis_prompt)]).content
            if synthesized and len(synthesized.strip()) > 10:
                final_answer = synthesized.strip()
                print(f"[后处理节点] LLM已整理答案（原{len(raw_answer)}字 → {len(final_answer)}字）")
            else:
                final_answer = raw_answer
        except Exception as e:
            print(f"[后处理节点] LLM整理失败，使用原始答案: {e}")
            final_answer = raw_answer
    else:
        final_answer = raw_answer

    # 添加助手消息到记忆
    memory_manager.add_message("assistant", final_answer)

    # 更新用户偏好
    memory_manager.update_preferences(state["user_query"], final_answer)

    # 根据偏好调整最终答案
    adapted_answer = memory_manager.adapt_response(final_answer)
    conversation_summary = memory_manager.generate_summary()

    # 安全打印
    try:
        print(f"[后处理节点] 最终答案预览: {adapted_answer[:100]}...")
    except UnicodeEncodeError:
        safe_answer = adapted_answer.encode('gbk', errors='replace').decode('gbk', errors='replace')
        print(f"[后处理节点] 最终答案预览: {safe_answer[:100]}...")

    return {**state, "final_answer": adapted_answer, "conversation_summary": conversation_summary}


def create_workflow() -> StateGraph:
    """创建标准 ReAct Agent 工作流

    工作流结构：
    preprocess -> agent (ReAct loop: LLM思考 → 工具调用 → 观察 → ... → 最终回答) -> postprocess
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("postprocess", postprocess_node)

    workflow.set_entry_point("preprocess")
    workflow.add_edge("preprocess", "agent")
    workflow.add_edge("agent", "postprocess")
    workflow.add_edge("postprocess", END)

    return workflow


# ========== 主入口函数 ==========
def run_langgraph_agent_with_memory(user_query: str, max_iterations: int = 3) -> Dict[str, Any]:
    """
    运行带记忆的 LangGraph Agent 处理用户查询

    Args:
        user_query: 用户查询
        max_iterations: 最大迭代次数（用于答案改进）

    Returns:
        Dict containing plan, tool_results, final_answer, workflow info, and memory info
    """
    if not user_query or not isinstance(user_query, str) or not user_query.strip():
        return {
            "plan": None, "tool_results": None,
            "final_answer": "请输入有效的查询内容",
            "raw_context": "", "active_skills": [],
            "mcp_status": {}, "workflow_info": {"iterations": 0, "final_step": None, "answer_quality": "N/A"},
            "memory_info": {"conversation_length": 0, "user_preferences": {}, "recent_topics": [], "conversation_summary": ""}
        }
    user_query = user_query.strip()
    memory_manager = get_memory_manager()

    # 初始化状态
    initial_state: AgentState = {
        "user_query": user_query,
        "messages": [],
        "user_preferences": memory_manager.user_preferences,
        "plan": None,
        "tool_results": None,
        "final_answer": None,
        "raw_context": None,
        "step": "agent",
        "iteration": 0,
        "max_iterations": max_iterations,
        "conversation_summary": memory_manager.generate_summary(),
        "tracking_info": None,
        "active_skills": None,
        "skill_context": None,
    }

    try:
        workflow = create_workflow()
    except Exception as e:
        return {
            "plan": None, "tool_results": None,
            "final_answer": f"工作流创建失败: {e}",
            "raw_context": "", "active_skills": [],
            "mcp_status": {}, "workflow_info": {"iterations": 0, "final_step": None, "answer_quality": "N/A"},
            "memory_info": {"conversation_length": 0, "user_preferences": {}, "recent_topics": [], "conversation_summary": ""}
        }

    try:
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)

        print(f"\n{'='*60}")
        print(f"开始处理查询: {user_query}")
        print(f"{'='*60}")

        config = {"configurable": {"thread_id": "user_session_1"}}
        final_state = app.invoke(initial_state, config)

        print(f"\n{'='*60}")
        print("处理完成")
        print(f"{'='*60}")
    except Exception as e:
        print(f"Agent 执行失败: {e}")
        final_state = None

    memory_info = {
        "conversation_length": len(memory_manager.conversation_history),
        "user_preferences": {
            k: (list(v) if isinstance(v, set) else v)
            for k, v in memory_manager.user_preferences.items()
        },
        "recent_topics": list(memory_manager.user_preferences.get("frequent_topics", set())),
        "conversation_summary": memory_manager.generate_summary()
    }

    # 获取 MCP 状态
    mcp_status = {}
    try:
        from mcp_client import get_mcp_manager
        mcp_mgr = get_mcp_manager()
        if mcp_mgr:
            mcp_status = mcp_mgr.get_status()
    except Exception:
        pass

    return {
        "plan": final_state.get("plan") if final_state else None,
        "tool_results": final_state.get("tool_results") if final_state else None,
        "final_answer": final_state.get("final_answer") if final_state else "处理失败，请稍后重试",
        "raw_context": final_state.get("raw_context", "") if final_state else "",
        "active_skills": final_state.get("active_skills", []) if final_state else [],
        "mcp_status": mcp_status,
        "workflow_info": {
            "iterations": 1,
            "final_step": final_state.get("step") if final_state else None,
            "answer_quality": "N/A"
        },
        "memory_info": memory_info
    }


def visualize_workflow(output_path: str = "workflow_with_memory.png"):
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


# ========== 记忆管理功能 ==========
def get_conversation_history() -> List[Dict[str, str]]:
    """获取完整对话历史"""
    memory_manager = get_memory_manager()
    return memory_manager.conversation_history


def clear_memory():
    """清空对话记忆"""
    memory_manager = get_memory_manager()
    memory_manager.conversation_history = []
    memory_manager.user_preferences = {
        "language_style": "neutral",
        "detail_level": "moderate",
        "frequent_topics": set(),
        "last_interaction": None
    }
    print("记忆已清空")


def export_memory_to_file(filepath: str = "conversation_memory.json"):
    """导出记忆到文件"""
    memory_manager = get_memory_manager()
    data = {
        "conversation_history": memory_manager.conversation_history,
        "user_preferences": {
            "language_style": memory_manager.user_preferences["language_style"],
            "detail_level": memory_manager.user_preferences["detail_level"],
            "frequent_topics": list(memory_manager.user_preferences.get("frequent_topics", set())),
            "last_interaction": memory_manager.user_preferences["last_interaction"]
        },
        "export_timestamp": datetime.now().isoformat()
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"记忆已导出到: {filepath}")
    return filepath


# ========== 测试函数 ==========
async def run_agent_stream(user_query: str, max_iterations: int = 3):
    """流式运行 Agent，逐步产出事件供前端实时展示。

    使用 LangGraph astream_events 捕获每个节点完成后的状态变化，
    让 Streamlit 等前端可以实时展示思考过程。

    Yields:
        {"type": "node_start", "node": str, "message": str}
        {"type": "node_done", "node": str, "state": dict}
        {"type": "final_answer", "answer": str, "result": dict}
        {"type": "error", "message": str}
    """
    if not user_query or not user_query.strip():
        yield {"type": "error", "message": "请输入有效的查询内容"}
        return

    workflow = create_workflow()
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    initial_state: AgentState = {
        "user_query": user_query.strip(),
        "messages": [],
        "user_preferences": get_memory_manager().user_preferences,
        "plan": None,
        "tool_results": None,
        "final_answer": None,
        "raw_context": None,
        "step": "agent",
        "iteration": 0,
        "max_iterations": max_iterations,
        "conversation_summary": get_memory_manager().generate_summary(),
        "tracking_info": None,
        "active_skills": None,
        "skill_context": None,
    }

    config = {"configurable": {"thread_id": "user_session_1"}}

    try:
        yield {"type": "node_start", "node": "preprocess", "message": "正在分析问题..."}
        async for event in app.astream_events(initial_state, config, version="v2"):
            kind = event.get("event", "")

            if kind == "on_chain_start":
                node_name = event.get("name", "")
                if node_name in ("preprocess", "agent", "postprocess"):
                    yield {"type": "node_start", "node": node_name, "message": _node_status_message(node_name)}

            elif kind == "on_chain_end":
                node_name = event.get("name", "")
                output = event.get("data", {}).get("output", {})
                if node_name == "agent" and isinstance(output, dict):
                    # agent 节点完成，传递工具调用信息
                    tool_results = output.get("tool_results") or {}
                    plan = output.get("plan") or []
                    yield {
                        "type": "node_done", "node": node_name,
                        "tool_calls": plan,
                        "tool_results": {k: str(v)[:200] for k, v in tool_results.items()},
                    }
                elif node_name == "postprocess" and isinstance(output, dict):
                    final_answer = output.get("final_answer", "")
                    yield {"type": "final_answer", "answer": final_answer, "result": output}

    except Exception as e:
        yield {"type": "error", "message": f"处理失败: {e}"}


def _node_status_message(node_name: str) -> str:
    """节点状态消息"""
    messages = {
        "preprocess": "正在分析问题...",
        "agent": "正在检索知识库并调用工具...",
        "postprocess": "正在整理答案...",
    }
    return messages.get(node_name, f"正在执行 {node_name}...")


def test_agent_with_memory():
    """测试带记忆的Agent"""
    test_queries = [
        "如何重置密码？",
        "再次问一下重置密码的事情",
        "查询工单 TK-123456",
        "今天几号？",
        "北京天气怎么样？"
    ]

    print("开始测试带记忆的 Agent...")
    for query in test_queries:
        print(f"\n{'#'*60}")
        print(f"测试查询: {query}")
        print(f"{'#'*60}")
        result = run_langgraph_agent_with_memory(query)
        print(f"最终答案: {result['final_answer']}")
        print(f"记忆摘要: {result['memory_info']['conversation_summary']}")


# ========== 主入口 ==========
if __name__ == "__main__":
    print("LangGraph 企业智能客服 Agent (带记忆版本) 已启动")
    print("命令说明:")
    print("  'quit' - 退出")
    print("  'visualize' - 生成工作流图")
    print("  'test' - 运行测试")
    print("  'history' - 查看对话历史")
    print("  'clear' - 清空记忆")
    print("  'export' - 导出记忆")

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
            test_agent_with_memory()
            continue
        elif query.lower() == 'history':
            history = get_conversation_history()
            print(f"\n对话历史 ({len(history)} 条消息):")
            for msg in history:
                print(f"  [{msg['role']}] {msg['content']}")
            continue
        elif query.lower() == 'clear':
            clear_memory()
            continue
        elif query.lower() == 'export':
            export_memory_to_file()
            continue

        result = run_langgraph_agent_with_memory(query)
        print(f"\n助手: {result.get('final_answer', '处理失败')}")
        print("\n[调试信息]")
        print(f"- 任务规划: {result.get('plan')}")
        print(f"- 迭代次数: {result.get('workflow_info', {}).get('iterations', 'N/A')}")
        print(f"- 答案质量: {result.get('workflow_info', {}).get('answer_quality', 'N/A')}")
        print(f"- 对话长度: {result.get('memory_info', {}).get('conversation_length', 0)} 条消息")
        print(f"- 用户偏好: {result.get('memory_info', {}).get('user_preferences', {})}")
