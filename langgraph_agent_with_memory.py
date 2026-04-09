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
import re
from datetime import date, datetime
from typing import TypedDict, List, Dict, Any, Literal, Optional, Annotated
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import operator

# 尝试导入高级RAG系统
try:
    from advanced_rag_system import create_advanced_rag_system, AdvancedRAGRetriever
    from langchain_core.documents import Document
    ADVANCED_RAG_AVAILABLE = True
except ImportError as e:
    print(f"警告：高级RAG系统导入失败，将使用模拟模式: {e}")
    ADVANCED_RAG_AVAILABLE = False

# 尝试导入多模态支持系统
try:
    from multimodal_support import MultimodalTools, MediaDetector, MediaType
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    print(f"警告：多模态支持系统导入失败，将使用模拟模式: {e}")
    MULTIMODAL_AVAILABLE = False

# 尝试导入监控系统
try:
    from monitoring_system import get_monitoring_system, monitor_workflow, monitor_node
    from monitoring_config import get_config_manager
    from monitoring_config import MonitoringConfig  # 配置类
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

# 异步执行器全局变量
async_executor = None
parallel_scheduler = None
streaming_handler = None

# 工作流配置
USE_ASYNC_EXECUTION = True  # 默认使用异步执行

# 尝试导入异步执行器
try:
    from async_executor import (
        AsyncToolExecutor,
        ParallelTaskScheduler,
        StreamingResponseHandler,
        get_async_executor,
        get_parallel_scheduler,
        get_streaming_handler,
        run_tools_parallel,
        run_tools_streaming,
        TaskPriority,
        TaskStatus,
        StreamChunk
    )
    ASYNC_EXECUTOR_AVAILABLE = True
    print("[INFO] 异步执行器已导入")
except ImportError as e:
    print(f"警告：异步执行器导入失败，将使用同步执行模式: {e}")
    ASYNC_EXECUTOR_AVAILABLE = False

# 加载环境变量
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")


# ========== 增强状态定义（带记忆） ==========
class AgentState(TypedDict):
    """Agent 工作流的状态定义（带记忆）"""
    # 输入和上下文
    user_query: str
    # 对话历史
    messages: Annotated[list, add_messages]
    # 用户偏好
    user_preferences: Dict[str, Any]
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
    # 记忆摘要
    conversation_summary: Optional[str]


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
            if "密码" in content:
                topics.add("密码重置")
            elif "工单" in content:
                topics.add("工单查询")
            elif "天气" in content:
                topics.add("天气查询")
            elif "股票" in content:
                topics.add("股票查询")
            elif "转人工" in content:
                topics.add("人工客服")
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


# ========== 模拟 LLM（带记忆增强） ==========
def get_llm():
    """获取 LLM 实例（带记忆上下文）"""
    # 使用模拟 LLM 避免 API 调用
    class MockLLM:
        def invoke(self, input_dict):
            class MockResponse:
                def __init__(self, content):
                    self.content = content

            memory_manager = get_memory_manager()
            query = input_dict.get("query", "")

            # 检查对话历史中是否有相似问题
            recent_history = memory_manager.get_recent_history(3)
            similar_questions = []
            for msg in recent_history:
                if msg["role"] == "user" and msg["content"] != query:
                    similar_questions.append(msg["content"])

            # 模拟响应（带记忆感知）
            if "重置密码" in query:
                if "密码重置" in memory_manager.user_preferences.get("frequent_topics", set()):
                    response = '["knowledge_search: 重置密码（您之前也问过类似问题）"]'
                else:
                    response = '["knowledge_search: 重置密码"]'
                    memory_manager.user_preferences["frequent_topics"].add("密码重置")
            elif "TK-123456" in query:
                response = '["ticket_query: TK-123456"]'
            elif "今天" in query:
                response = '["date_query"]'
            elif "转人工" in query:
                response = '["escalate"]'
            elif "天气" in query:
                city = "北京"
                if "上海" in query:
                    city = "上海"
                elif "广州" in query:
                    city = "广州"
                elif "深圳" in query:
                    city = "深圳"
                response = f'["weather_query: {city}"]'
            elif "股票" in query:
                symbol = "AAPL"
                if "谷歌" in query or "GOOGL" in query:
                    symbol = "GOOGL"
                elif "特斯拉" in query or "TSLA" in query:
                    symbol = "TSLA"
                response = f'["stock_query: {symbol}"]'
            elif "上传" in query or "文件" in query:
                # 多模态：文件上传处理
                response = '["file_upload_processing: uploaded_file.pdf"]'
            elif "图片" in query or "图像" in query or "照片" in query or "截图" in query or "screenshot" in query.lower():
                # 多模态：图像分析
                # 尝试从查询中提取文件路径
                if "screenshot" in query.lower() or "截图" in query:
                    response = '["image_analysis: screenshot.png"]'
                else:
                    response = '["image_analysis: test_image.png"]'
            elif "文档" in query or "pdf" in query.lower() or "word" in query.lower() or "excel" in query.lower():
                # 多模态：文档处理
                if "pdf" in query.lower():
                    response = '["document_processing: test_document.pdf"]'
                elif "word" in query.lower():
                    response = '["document_processing: test_report.docx"]'
                elif "excel" in query.lower():
                    response = '["document_processing: test_data.xlsx"]'
                else:
                    response = '["document_processing: document.pdf"]'
            else:
                # 如果有相似历史问题，可以引用
                if similar_questions:
                    ref = similar_questions[0][:20] + "..."
                    response = f'["knowledge_search: {query}（参考之前问题：{ref}）"]'
                else:
                    response = f'["knowledge_search: {query}"]'

            return MockResponse(response)

    return MockLLM()


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
                        page_content="如何重置密码？您可以通过登录页面点击'忘记密码'链接重置密码。",
                        metadata={"source": "faq", "topic": "password"}
                    ),
                    Document(
                        page_content="产品价格信息：企业版每年10,000元，包含技术支持。",
                        metadata={"source": "pricing", "topic": "price"}
                    ),
                    Document(
                        page_content="技术支持时间：工作日9:00-18:00，电话400-123-4567。",
                        metadata={"source": "support", "topic": "contact"}
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
    """增强知识库检索（带记忆上下文和高级RAG）"""
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
                # 提取最相关的结果
                best_doc = results[0]
                page_content = best_doc.page_content

                # 根据元数据信息格式化响应
                source = best_doc.metadata.get('source', '知识库')
                score = best_doc.metadata.get('final_score', 0)
                confidence_score = score if score else best_doc.metadata.get('relevance_score', 0)

                # 构建响应
                if confidence_score > 0.5:
                    confidence = "高度相关"
                elif confidence_score > 0.2:
                    confidence = "相关"
                else:
                    confidence = "参考信息"

                response = f"[高级RAG检索] {confidence} 信息来自 {source}（相关性: {confidence_score:.2f}）：{page_content}"

                # 如果分数较低，添加提示
                if confidence_score < 0.2:
                    response += "\n\n[注意] 检索到的信息相关性较低，建议转人工客服获取更准确的回答。"

                # 添加检索统计信息（如果可用）
                if hasattr(advanced_rag_retriever, 'cache_manager'):
                    try:
                        cache_stats = advanced_rag_retriever.cache_manager.get_stats()
                        if cache_stats['result_hits'] > 0 or cache_stats['embedding_hits'] > 0:
                            total_hits = cache_stats['result_hits'] + cache_stats['embedding_hits']
                            response += f"\n\n[缓存信息] 本次检索使用了缓存（总缓存命中: {total_hits} 次）"
                    except Exception as e:
                        print(f"[DEBUG] 获取缓存统计失败: {e}")

                adapted_response = memory_manager.adapt_response(response)

                # 监控跟踪：RAG检索成功
                if MONITORING_AVAILABLE and monitoring_system:
                    try:
                        monitoring_system.track_rag_retrieval(query, duration, len(results), success=True)
                    except Exception as e:
                        print(f"[WARN] RAG检索成功监控跟踪失败: {e}")

                return adapted_response
            else:
                print(f"[WARN] 高级RAG检索到0个结果，回退到模拟模式")
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

    # 回退到模拟模式
    print(f"[INFO] 使用模拟模式检索查询: '{query}'")
    mock_responses = {
        "重置密码": "您可以通过登录页面点击'忘记密码'链接重置密码。系统将发送重置邮件到您的注册邮箱。",
        "产品价格": "企业版产品价格为每年 10,000 元，包含所有功能和技术支持。",
        "技术支持": "技术支持时间为工作日 9:00-18:00，电话：400-123-4567。",
        "默认": "根据知识库信息，您的问题已记录，我们会尽快为您提供详细解答。"
    }

    for key, response in mock_responses.items():
        if key in query:
            # 根据用户偏好调整响应
            adapted_response = memory_manager.adapt_response(response)
            return adapted_response

    default_response = mock_responses["默认"]
    adapted_response = memory_manager.adapt_response(default_response)
    return adapted_response


@tool
def query_ticket_status(ticket_id: str) -> str:
    """模拟查询工单状态"""
    mock_status = {
        "TK-123456": "您的工单 TK-123456 已受理，正在处理中，预计48小时内完成。",
        "TK-789012": "工单 TK-789012 已处理完毕，请登录系统查看结果。",
        "default": "未找到工单信息，请确认工单号是否正确。"
    }

    memory_manager = get_memory_manager()
    response = mock_status.get(ticket_id, mock_status["default"])
    adapted_response = memory_manager.adapt_response(response)
    return adapted_response


@tool
def escalate_to_human(query: str) -> str:
    """模拟转人工处理"""
    memory_manager = get_memory_manager()
    response = "感谢您的耐心，我已将您的问题转接给人工客服，他们将尽快与您联系（预计5分钟内）。"
    adapted_response = memory_manager.adapt_response(response)
    return adapted_response


@tool
def get_current_date(query: str) -> str:
    """返回今天的日期"""
    memory_manager = get_memory_manager()
    response = f"今天是 {date.today()}。"
    adapted_response = memory_manager.adapt_response(response)
    return adapted_response


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

    memory_manager = get_memory_manager()
    response = weather_data.get(city, weather_data["default"])
    adapted_response = memory_manager.adapt_response(response)
    return adapted_response


@tool
def stock_query(symbol: str) -> str:
    """查询股票实时价格。支持 AAPL、GOOGL、TSLA 等股票代码。"""
    stock_data = {
        "AAPL": "苹果公司 (AAPL) 当前价格 $175.20，今日上涨 2.3%。",
        "GOOGL": "谷歌 (GOOGL) 当前价格 $155.80，今日下跌 0.5%。",
        "TSLA": "特斯拉 (TSLA) 当前价格 $180.50，今日上涨 5.2%。",
        "default": "该股票代码信息暂不可用，请确认代码是否正确。"
    }

    memory_manager = get_memory_manager()
    response = stock_data.get(symbol, stock_data["default"])
    adapted_response = memory_manager.adapt_response(response)
    return adapted_response


# ========== 多模态工具函数 ==========
@tool
def image_analysis(image_path: str) -> str:
    """分析图像内容：OCR文字识别、物体检测、场景理解"""
    memory_manager = get_memory_manager()

    if not MULTIMODAL_AVAILABLE:
        response = f"多模态支持不可用，无法分析图像：{image_path}"
        adapted_response = memory_manager.adapt_response(response)
        return adapted_response

    try:
        multimodal_tools = MultimodalTools()
        result = multimodal_tools.analyze_image(image_path)
        response = f"图像分析结果：\n{result}"
    except Exception as e:
        response = f"图像分析失败：{e}\n请确认文件路径：{image_path}"

    adapted_response = memory_manager.adapt_response(response)
    return adapted_response


@tool
def document_processing(document_path: str) -> str:
    """处理文档：提取PDF、Word、Excel文件内容"""
    memory_manager = get_memory_manager()

    if not MULTIMODAL_AVAILABLE:
        response = f"多模态支持不可用，无法处理文档：{document_path}"
        adapted_response = memory_manager.adapt_response(response)
        return adapted_response

    try:
        multimodal_tools = MultimodalTools()
        result = multimodal_tools.extract_document_content(document_path)
        response = f"文档处理结果：\n{result}"
    except Exception as e:
        response = f"文档处理失败：{e}\n请确认文件路径：{document_path}"

    adapted_response = memory_manager.adapt_response(response)
    return adapted_response


@tool
def file_upload_processing(file_path: str) -> str:
    """处理上传的文件：自动识别类型并处理"""
    memory_manager = get_memory_manager()

    if not MULTIMODAL_AVAILABLE:
        response = f"多模态支持不可用，无法处理文件：{file_path}"
        adapted_response = memory_manager.adapt_response(response)
        return adapted_response

    try:
        multimodal_tools = MultimodalTools()
        result = multimodal_tools.process_uploaded_file(file_path)
        response = f"文件处理完成：\n{result}"
    except Exception as e:
        response = f"文件处理失败：{e}\n请确认文件路径：{file_path}"

    adapted_response = memory_manager.adapt_response(response)
    return adapted_response


# ========== LangGraph 节点函数（带记忆） ==========
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
    return {**state, "conversation_summary": conversation_summary, "step": "planning", "tracking_info": tracking_info}


def planning_node(state: AgentState) -> AgentState:
    """规划节点：分析用户查询，拆解为任务列表（带记忆上下文）"""
    import time
    start_time = time.time()

    print(f"[规划节点] 处理查询: {state['user_query']}")
    print(f"[规划节点] 对话摘要: {state.get('conversation_summary', '无摘要')}")

    # 使用模拟 LLM 生成规划
    llm = get_llm()

    # 使用原始查询，避免上下文干扰工具选择
    enhanced_query = state['user_query']

    response = llm.invoke({"query": enhanced_query}).content

    try:
        tasks = json.loads(response)
    except:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        tasks = json.loads(match.group()) if match else []

    duration = time.time() - start_time
    print(f"[规划节点] 生成任务: {tasks} (耗时: {duration:.2f}s)")

    # 监控跟踪：节点执行
    if MONITORING_AVAILABLE and monitoring_system and 'tracking_info' in state:
        try:
            monitoring_system.track_node_execution(
                state['tracking_info'],
                node_name="planning_node",
                inputs={"query": enhanced_query, "conversation_summary": state.get('conversation_summary')},
                outputs={"tasks": tasks},
                duration=duration,
                success=True
            )
        except Exception as e:
            print(f"[WARN] 规划节点监控跟踪失败: {e}")

    # 根据配置决定下一步节点名称
    if USE_ASYNC_EXECUTION and ASYNC_EXECUTOR_AVAILABLE:
        next_step = "execution_async"
    else:
        next_step = "execution"

    return {**state, "plan": tasks, "step": next_step}


def execution_node(state: AgentState) -> AgentState:
    """执行节点：执行规划的任务，收集结果"""
    import time
    start_time = time.time()

    print(f"[执行节点] 执行任务: {state['plan']}")

    if not state["plan"]:
        duration = time.time() - start_time
        # 监控跟踪：空执行节点
        if MONITORING_AVAILABLE and monitoring_system and 'tracking_info' in state:
            try:
                monitoring_system.track_node_execution(
                    state['tracking_info'],
                    node_name="execution_node",
                    inputs={"plan": state['plan']},
                    outputs={"results": {}},
                    duration=duration,
                    success=True
                )
            except Exception as e:
                print(f"[WARN] 执行节点监控跟踪失败: {e}")
        return {**state, "tool_results": {}, "step": "validation"}

    results = {}
    # 为每个工具调用添加跟踪
    tool_start_times = {}
    for task in state["plan"]:
        tool_start_time = time.time()

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
        elif task.startswith("image_analysis:"):
            image_path = task.replace("image_analysis:", "").strip()
            results[task] = image_analysis.run(image_path)
        elif task.startswith("document_processing:"):
            doc_path = task.replace("document_processing:", "").strip()
            results[task] = document_processing.run(doc_path)
        elif task.startswith("file_upload_processing:"):
            file_path = task.replace("file_upload_processing:", "").strip()
            results[task] = file_upload_processing.run(file_path)
        else:
            results[task] = "未知任务"

        tool_duration = time.time() - tool_start_time
        # 记录工具调用指标
        if MONITORING_AVAILABLE and monitoring_system:
            try:
                tool_name = task.split(":")[0] if ":" in task else task
                monitoring_system.metrics.record_tool_call(tool_name, tool_duration, success=True)
            except Exception as e:
                print(f"[WARN] 工具调用指标记录失败: {e}")

    duration = time.time() - start_time
    print(f"[执行节点] 执行结果: {results} (总耗时: {duration:.2f}s)")

    # 监控跟踪：执行节点
    if MONITORING_AVAILABLE and monitoring_system and 'tracking_info' in state:
        try:
            monitoring_system.track_node_execution(
                state['tracking_info'],
                node_name="execution_node",
                inputs={"plan": state['plan']},
                outputs={"results": results},
                duration=duration,
                success=True
            )
        except Exception as e:
            print(f"[WARN] 执行节点监控跟踪失败: {e}")

    return {**state, "tool_results": results, "step": "validation"}


def execution_node_async(state: AgentState) -> AgentState:
    """异步执行节点：并行执行规划的任务，收集结果"""
    import time
    start_time = time.time()

    print(f"[异步执行节点] 执行任务: {state['plan']}")

    if not state["plan"]:
        duration = time.time() - start_time
        # 监控跟踪：空执行节点
        if MONITORING_AVAILABLE and monitoring_system and 'tracking_info' in state:
            try:
                monitoring_system.track_node_execution(
                    state['tracking_info'],
                    node_name="execution_node_async",
                    inputs={"plan": state['plan']},
                    outputs={"results": {}},
                    duration=duration,
                    success=True
                )
            except Exception as e:
                print(f"[WARN] 异步执行节点监控跟踪失败: {e}")
        return {**state, "tool_results": {}, "step": "validation"}

    # 初始化异步执行器（如果需要）
    global async_executor, parallel_scheduler
    if ASYNC_EXECUTOR_AVAILABLE:
        try:
            if async_executor is None:
                async_executor = get_async_executor(max_workers=4)
            if parallel_scheduler is None:
                parallel_scheduler = get_parallel_scheduler()
        except Exception as e:
            print(f"[WARN] 异步执行器初始化失败，将回退到同步执行: {e}")
            ASYNC_EXECUTOR_AVAILABLE = False

    if ASYNC_EXECUTOR_AVAILABLE:
        # 使用异步执行器并行执行任务
        results = {}
        tool_calls = []
        task_to_tool_map = {}  # 映射：任务描述 -> 工具函数和参数

        # 解析任务，构建工具调用列表
        for task in state["plan"]:
            if task.startswith("knowledge_search:"):
                query = task.replace("knowledge_search:", "").strip()
                tool_calls.append({
                    "func": knowledge_search.run,
                    "args": [query],
                    "tool_name": "knowledge_search",
                    "timeout": 30.0,
                    "priority": 2  # NORMAL priority
                })
                task_to_tool_map[task] = len(tool_calls) - 1

            elif task.startswith("ticket_query:"):
                ticket_id = task.replace("ticket_query:", "").strip()
                tool_calls.append({
                    "func": query_ticket_status.run,
                    "args": [ticket_id],
                    "tool_name": "ticket_query",
                    "timeout": 10.0,
                    "priority": 2
                })
                task_to_tool_map[task] = len(tool_calls) - 1

            elif task.startswith("weather_query:"):
                city = task.replace("weather_query:", "").strip()
                tool_calls.append({
                    "func": weather_query.run,
                    "args": [city],
                    "tool_name": "weather_query",
                    "timeout": 10.0,
                    "priority": 2
                })
                task_to_tool_map[task] = len(tool_calls) - 1

            elif task.startswith("stock_query:"):
                symbol = task.replace("stock_query:", "").strip()
                tool_calls.append({
                    "func": stock_query.run,
                    "args": [symbol],
                    "tool_name": "stock_query",
                    "timeout": 10.0,
                    "priority": 2
                })
                task_to_tool_map[task] = len(tool_calls) - 1

            elif task == "escalate":
                tool_calls.append({
                    "func": escalate_to_human.run,
                    "args": [""],
                    "tool_name": "escalate_to_human",
                    "timeout": 5.0,
                    "priority": 3  # HIGH priority
                })
                task_to_tool_map[task] = len(tool_calls) - 1

            elif task == "date_query":
                tool_calls.append({
                    "func": get_current_date.run,
                    "args": [""],
                    "tool_name": "get_current_date",
                    "timeout": 5.0,
                    "priority": 2
                })
                task_to_tool_map[task] = len(tool_calls) - 1

            elif task.startswith("image_analysis:"):
                image_path = task.replace("image_analysis:", "").strip()
                tool_calls.append({
                    "func": image_analysis.run,
                    "args": [image_path],
                    "tool_name": "image_analysis",
                    "timeout": 60.0,  # 图像分析可能耗时较长
                    "priority": 2
                })
                task_to_tool_map[task] = len(tool_calls) - 1

            elif task.startswith("document_processing:"):
                doc_path = task.replace("document_processing:", "").strip()
                tool_calls.append({
                    "func": document_processing.run,
                    "args": [doc_path],
                    "tool_name": "document_processing",
                    "timeout": 45.0,  # 文档处理可能耗时较长
                    "priority": 2
                })
                task_to_tool_map[task] = len(tool_calls) - 1

            elif task.startswith("file_upload_processing:"):
                file_path = task.replace("file_upload_processing:", "").strip()
                tool_calls.append({
                    "func": file_upload_processing.run,
                    "args": [file_path],
                    "tool_name": "file_upload_processing",
                    "timeout": 45.0,
                    "priority": 2
                })
                task_to_tool_map[task] = len(tool_calls) - 1

            else:
                # 未知任务，直接记录结果
                results[task] = "未知任务"

        if tool_calls:
            print(f"[异步执行节点] 准备并行执行 {len(tool_calls)} 个工具调用")

            try:
                # 并行执行工具调用
                parallel_results = run_tools_parallel(tool_calls, timeout=60.0)

                # 映射回任务结果
                for task, tool_idx in task_to_tool_map.items():
                    tool_call = tool_calls[tool_idx]
                    tool_name = tool_call["tool_name"]

                    # 查找对应的任务ID（任务ID是工具调用的索引）
                    task_id = None
                    for result_task_id, result in parallel_results.items():
                        if f"tool_task_" in result_task_id and tool_idx == int(result_task_id.split("_")[2]):
                            task_id = result_task_id
                            break

                    if task_id and task_id in parallel_results:
                        result = parallel_results[task_id]
                        if isinstance(result, Exception):
                            results[task] = f"工具调用失败: {result}"
                            # 记录失败指标
                            if MONITORING_AVAILABLE and monitoring_system:
                                try:
                                    monitoring_system.metrics.record_tool_call(tool_name, 0.0, success=False)
                                except Exception as e:
                                    print(f"[WARN] 工具失败指标记录失败: {e}")
                        else:
                            results[task] = result
                            # 记录成功指标（这里使用估计的执行时间）
                            if MONITORING_AVAILABLE and monitoring_system:
                                try:
                                    # 估计执行时间（假设平均0.5秒）
                                    monitoring_system.metrics.record_tool_call(tool_name, 0.5, success=True)
                                except Exception as e:
                                    print(f"[WARN] 工具成功指标记录失败: {e}")
                    else:
                        results[task] = "工具调用结果未找到"
            except Exception as e:
                print(f"[ERROR] 并行工具调用失败，将回退到串行执行: {e}")
                # 回退到同步执行
                return execution_node(state)

        else:
            results = {}
    else:
        # 异步执行器不可用，回退到同步执行
        print("[WARN] 异步执行器不可用，使用同步执行节点")
        return execution_node(state)

    duration = time.time() - start_time
    print(f"[异步执行节点] 执行结果: {results} (总耗时: {duration:.2f}s)")
    print(f"[异步执行节点] 相比串行执行，预计节省时间: {len(state['plan']) * 0.5 - duration:.2f}s (估计)")

    # 监控跟踪：异步执行节点
    if MONITORING_AVAILABLE and monitoring_system and 'tracking_info' in state:
        try:
            monitoring_system.track_node_execution(
                state['tracking_info'],
                node_name="execution_node_async",
                inputs={"plan": state['plan']},
                outputs={"results": results},
                duration=duration,
                success=True
            )
        except Exception as e:
            print(f"[WARN] 异步执行节点监控跟踪失败: {e}")

    return {**state, "tool_results": results, "step": "validation"}


def validation_node(state: AgentState) -> AgentState:
    """验证节点：验证答案质量，决定下一步"""
    import time
    start_time = time.time()

    print(f"[验证节点] 验证答案质量")

    # 合并所有工具结果
    preliminary = "\n".join(state["tool_results"].values()) if state["tool_results"] else "无结果"

    # 验证逻辑
    trust_keywords = ["工单", "受理", "处理中", "已完成", "今天", "年-月-日", "转人工", "天气", "气温", "股票", "价格", "上涨", "下跌",
                     "图像", "图片", "照片", "OCR", "识别", "文档", "PDF", "Word", "Excel", "文件", "上传", "处理", "内容"]
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

    duration = time.time() - start_time
    print(f"[验证节点] 答案质量: {answer_quality}, 转人工: {needs_human_escalation} (耗时: {duration:.2f}s)")

    # 监控跟踪：验证节点
    if MONITORING_AVAILABLE and monitoring_system and 'tracking_info' in state:
        try:
            monitoring_system.track_node_execution(
                state['tracking_info'],
                node_name="validation_node",
                inputs={"preliminary": preliminary[:200]},
                outputs={"final_answer": final_answer[:200], "answer_quality": answer_quality, "needs_human_escalation": needs_human_escalation},
                duration=duration,
                success=True
            )
        except Exception as e:
            print(f"[WARN] 验证节点监控跟踪失败: {e}")

    return {
        **state,
        "final_answer": final_answer,
        "answer_quality": answer_quality,
        "needs_human_escalation": needs_human_escalation,
        "step": "completed" if not needs_human_escalation else "escalate"
    }


def postprocess_node(state: AgentState) -> AgentState:
    """后处理节点：更新记忆，生成最终响应"""
    import time
    start_time = time.time()

    print(f"[后处理节点] 处理最终答案")

    memory_manager = get_memory_manager()

    # 添加助手消息到记忆
    memory_manager.add_message("assistant", state["final_answer"])

    # 更新用户偏好
    memory_manager.update_preferences(state["user_query"], state["final_answer"])

    # 根据偏好调整最终答案
    adapted_answer = memory_manager.adapt_response(state["final_answer"])

    # 添加记忆上下文提示（如果相关）
    conversation_summary = memory_manager.generate_summary()
    if "密码重置" in conversation_summary and "密码" in state["user_query"]:
        adapted_answer = adapted_answer + " （注意：您之前也询问过密码相关问题）"
    elif "图像处理" in conversation_summary and ("图片" in state["user_query"] or "图像" in state["user_query"] or "照片" in state["user_query"]):
        adapted_answer = adapted_answer + " （注意：您之前也处理过图像相关问题）"
    elif "文档处理" in conversation_summary and ("文档" in state["user_query"] or "pdf" in state["user_query"].lower() or "word" in state["user_query"].lower() or "excel" in state["user_query"].lower()):
        adapted_answer = adapted_answer + " （注意：您之前也处理过文档相关问题）"

    print(f"[后处理节点] 调整后答案: {adapted_answer}")

    # 监控跟踪：后处理节点执行和工作流结束
    if MONITORING_AVAILABLE and monitoring_system and 'tracking_info' in state:
        duration = time.time() - start_time

        # 跟踪节点执行
        try:
            monitoring_system.track_node_execution(
                state['tracking_info'],
                node_name="postprocess_node",
                inputs={"final_answer": state.get("final_answer", ""), "query": state.get("user_query", "")},
                outputs={"adapted_answer": adapted_answer, "conversation_summary": conversation_summary},
                duration=duration,
                success=True
            )
        except Exception as e:
            print(f"[WARN] 后处理节点监控跟踪失败: {e}")

        # 跟踪工作流结束（仅在正常结束且不需要人工介入时）
        if not state.get("needs_human_escalation", False):
            try:
                monitoring_system.track_workflow_end(
                    state['tracking_info'],
                    outputs={"final_answer": adapted_answer, "conversation_summary": conversation_summary},
                    success=True
                )
            except Exception as e:
                print(f"[WARN] 工作流结束监控跟踪失败: {e}")

    return {**state, "final_answer": adapted_answer, "conversation_summary": conversation_summary}


def human_escalation_node(state: AgentState) -> AgentState:
    """人工升级节点：处理需要人工介入的情况"""
    import time
    start_time = time.time()

    print(f"[人工升级节点] 转人工处理")

    memory_manager = get_memory_manager()

    if state["final_answer"] and "转人工" in state["final_answer"]:
        final_answer = state["final_answer"]
    else:
        final_answer = "感谢您的耐心，我已将您的问题转接给人工客服，他们将尽快与您联系（预计5分钟内）。"

    # 根据偏好调整
    adapted_answer = memory_manager.adapt_response(final_answer)

    # 添加记忆
    memory_manager.add_message("assistant", adapted_answer)

    # 监控跟踪：人工升级节点执行和工作流结束
    if MONITORING_AVAILABLE and monitoring_system and 'tracking_info' in state:
        duration = time.time() - start_time

        # 跟踪节点执行
        try:
            monitoring_system.track_node_execution(
                state['tracking_info'],
                node_name="human_escalation_node",
                inputs={"final_answer": state.get("final_answer", ""), "query": state.get("user_query", "")},
                outputs={"adapted_answer": adapted_answer},
                duration=duration,
                success=True
            )
        except Exception as e:
            print(f"[WARN] 人工升级节点监控跟踪失败: {e}")

        # 跟踪工作流结束（人工升级的情况）
        try:
            monitoring_system.track_workflow_end(
                state['tracking_info'],
                outputs={"final_answer": adapted_answer, "escalation_reason": "human_assistance_required"},
                success=True
            )
        except Exception as e:
            print(f"[WARN] 人工升级工作流结束监控跟踪失败: {e}")

    return {**state, "final_answer": adapted_answer, "step": "completed"}


# ========== 条件路由函数 ==========
def route_after_validation(state: AgentState) -> Literal["human_escalation", "improvement", "postprocess"]:
    """验证后的路由决策"""
    if state["needs_human_escalation"]:
        return "human_escalation"
    elif state["answer_quality"] == "poor" and state["iteration"] < state["max_iterations"]:
        return "improvement"
    else:
        return "postprocess"


# ========== 图构建 ==========
def create_workflow(use_async_execution: bool = None) -> StateGraph:
    """创建带记忆的 LangGraph 工作流

    Args:
        use_async_execution: 是否使用异步执行节点（默认使用全局配置USE_ASYNC_EXECUTION）
    """
    workflow = StateGraph(AgentState)

    # 确定是否使用异步执行
    if use_async_execution is None:
        use_async_execution = USE_ASYNC_EXECUTION

    # 选择执行节点
    if use_async_execution and ASYNC_EXECUTOR_AVAILABLE:
        print("[INFO] 工作流使用异步执行节点")
        execution_node_func = execution_node_async
        execution_node_name = "execution_async"
    else:
        if use_async_execution and not ASYNC_EXECUTOR_AVAILABLE:
            print("[WARN] 异步执行器不可用，使用同步执行节点")
        else:
            print("[INFO] 工作流使用同步执行节点")
        execution_node_func = execution_node
        execution_node_name = "execution"

    # 添加节点
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node(execution_node_name, execution_node_func)
    workflow.add_node("validation", validation_node)
    workflow.add_node("postprocess", postprocess_node)
    workflow.add_node("human_escalation", human_escalation_node)

    # 设置入口点
    workflow.set_entry_point("preprocess")

    # 添加边（正常流程）
    workflow.add_edge("preprocess", "planning")
    workflow.add_edge("planning", execution_node_name)
    workflow.add_edge(execution_node_name, "validation")

    # 条件边（验证后）
    workflow.add_conditional_edges(
        "validation",
        route_after_validation,
        {
            "human_escalation": "human_escalation",
            "improvement": "planning",  # 重新规划改进
            "postprocess": "postprocess"
        }
    )

    # 后处理到结束
    workflow.add_edge("postprocess", END)

    # 人工升级后到结束
    workflow.add_edge("human_escalation", END)

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
    # 获取记忆管理器
    memory_manager = get_memory_manager()

    # 初始化状态
    initial_state: AgentState = {
        "user_query": user_query,
        "messages": [],
        "user_preferences": memory_manager.user_preferences,
        "plan": None,
        "tool_results": None,
        "final_answer": None,
        "step": "planning",
        "iteration": 0,
        "max_iterations": max_iterations,
        "needs_human_escalation": False,
        "answer_quality": None,
        "conversation_summary": memory_manager.generate_summary()
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

    # 获取当前记忆状态
    memory_info = {
        "conversation_length": len(memory_manager.conversation_history),
        "user_preferences": memory_manager.user_preferences,
        "recent_topics": list(memory_manager.user_preferences.get("frequent_topics", [])),
        "conversation_summary": memory_manager.generate_summary()
    }

    return {
        "plan": final_state["plan"] if final_state else None,
        "tool_results": final_state["tool_results"] if final_state else None,
        "final_answer": final_state["final_answer"] if final_state else None,
        "workflow_info": {
            "iterations": final_state["iteration"] + 1 if final_state else 0,
            "final_step": final_state["step"] if final_state else None,
            "answer_quality": final_state["answer_quality"] if final_state else None
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
            "frequent_topics": list(memory_manager.user_preferences.get("frequent_topics", [])),
            "last_interaction": memory_manager.user_preferences["last_interaction"]
        },
        "export_timestamp": datetime.now().isoformat()
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"记忆已导出到: {filepath}")
    return filepath


# ========== 测试函数 ==========
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
        print(f"\n助手: {result['final_answer']}")
        print(f"\n[调试信息]")
        print(f"- 任务规划: {result['plan']}")
        print(f"- 迭代次数: {result['workflow_info']['iterations']}")
        print(f"- 答案质量: {result['workflow_info']['answer_quality']}")
        print(f"- 对话长度: {result['memory_info']['conversation_length']} 条消息")
        print(f"- 用户偏好: {result['memory_info']['user_preferences']}")