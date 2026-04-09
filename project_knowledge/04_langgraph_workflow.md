# LangGraph工作流系统

## 概述
LangGraph是一个用于构建有状态、多智能体应用的库，特别适合实现复杂的工作流和状态机。本项目使用LangGraph将传统的多Agent系统重构为可视化、可调试的工作流。

## 1. LangGraph核心概念

### 1.1 状态图（StateGraph）

**概念**: 将工作流建模为状态机，包含节点（状态）和边（转移）

**代码示例**:
```python
# langgraph_agent_with_memory.py 第1219-1249行：工作流创建
from langgraph.graph import StateGraph, END

def create_workflow(use_async_execution: bool = None) -> StateGraph:
    """创建带记忆的 LangGraph 工作流"""
    workflow = StateGraph(AgentState)  # 基于状态类型创建图
    
    # 添加节点
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("planning", planning_node)
    
    # 选择执行节点（同步或异步）
    if use_async_execution and ASYNC_EXECUTOR_AVAILABLE:
        execution_node_func = execution_node_async
        execution_node_name = "execution_async"
    else:
        execution_node_func = execution_node
        execution_node_name = "execution"
    
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
    
    # 条件边（基于验证结果路由）
    workflow.add_conditional_edges(
        "validation",
        route_after_validation,
        {
            "human_escalation": "human_escalation",
            "improvement": "planning",  # 重新规划改进
            "postprocess": "postprocess"
        }
    )
    
    # 其他边
    workflow.add_edge("human_escalation", "postprocess")
    workflow.add_edge("postprocess", END)
    
    return workflow
```

### 1.2 状态定义（TypedDict）

**代码示例**:
```python
# langgraph_agent_with_memory.py 第71-113行：状态定义
from typing import TypedDict, List, Dict, Any, Literal, Optional, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Agent 工作流的状态定义（带记忆）"""
    # 输入和上下文
    user_query: str
    
    # 对话历史（LangGraph特殊注解）
    messages: Annotated[list, add_messages]
    
    # 用户偏好
    user_preferences: Dict[str, Any]
    
    # 规划阶段
    plan: Optional[List[str]]  # 任务列表，如 ["knowledge_search: 重置密码", "ticket_query: TK-123456"]
    
    # 执行阶段  
    tool_results: Optional[Dict[str, str]]  # 工具执行结果 {任务: 结果}
    
    # 验证阶段
    final_answer: Optional[str]  # 最终答案
    
    # 工作流控制
    step: Literal["planning", "execution", "validation", "completed", "escalate"]
    
    # 迭代控制
    iteration: int  # 当前迭代次数
    max_iterations: int  # 最大迭代次数
    
    # 质量控制
    needs_human_escalation: bool  # 是否需要人工介入
    answer_quality: Optional[str]  # 答案质量评估
    
    # 记忆管理
    conversation_summary: Optional[str]  # 对话摘要
    
    # 监控跟踪
    tracking_info: Dict[str, Any]  # 跟踪信息，用于监控
```

**状态设计要点**:
1. **类型安全**: 使用TypedDict提供类型提示
2. **完整性**: 记录全流程关键数据
3. **可扩展性**: 易于添加新状态字段
4. **监控友好**: 包含tracking_info用于监控集成

## 2. 工作流节点设计

### 2.1 预处理节点

**功能**: 分析用户查询，初始化跟踪信息

**代码示例**:
```python
# langgraph_agent_with_memory.py 第595-643行：预处理节点
def preprocess_node(state: AgentState) -> AgentState:
    """预处理节点：分析用户查询，初始化跟踪信息"""
    import time
    
    print(f"[预处理节点] 用户查询: {state['user_query']}")
    
    # 初始化tracking_info（如果不存在）
    if "tracking_info" not in state:
        state["tracking_info"] = {}
    
    tracking_info = state["tracking_info"]
    
    # 设置基本跟踪信息
    tracking_info.update({
        "query_id": f"query_{int(time.time())}_{hash(state['user_query']) % 10000}",
        "start_time": time.time(),
        "query_type": "unknown",
        "user_id": "anonymous",  # 实际应用中可以从会话获取
        "session_id": f"session_{int(time.time())}",
        "platform": "web"  # 可以扩展为mobile, api等
    })
    
    # 分析查询类型
    query = state["user_query"].lower()
    
    if any(kw in query for kw in ["工单", "tk-", "票据"]):
        tracking_info["query_type"] = "ticket_query"
    elif any(kw in query for kw in ["密码", "重置", "忘记密码", "修改密码"]):
        tracking_info["query_type"] = "knowledge_search"
    elif any(kw in query for kw in ["转人工", "人工客服", "真人"]):
        tracking_info["query_type"] = "escalation"
    elif any(kw in query for kw in ["天气", "温度", "预报"]):
        tracking_info["query_type"] = "weather_query"
    elif any(kw in query for kw in ["股票", "股价", "行情"]):
        tracking_info["query_type"] = "stock_query"
    elif any(kw in query for kw in ["日期", "今天", "时间"]):
        tracking_info["query_type"] = "date_query"
    else:
        tracking_info["query_type"] = "general_query"
    
    print(f"[预处理节点] 查询类型: {tracking_info['query_type']}")
    
    # 监控跟踪
    if MONITORING_AVAILABLE and monitoring_system and 'tracking_info' in state:
        try:
            monitoring_system.track_node_execution(
                state['tracking_info'],
                node_name="preprocess_node",
                inputs={"user_query": state['user_query']},
                outputs={"query_type": tracking_info["query_type"]},
                duration=0.0,  # 预处理节点耗时很短
                success=True
            )
        except Exception as e:
            print(f"[WARN] 预处理节点监控跟踪失败: {e}")
    
    # 初始化其他状态字段
    state.update({
        "plan": None,
        "tool_results": None,
        "final_answer": None,
        "step": "planning",
        "iteration": 0,
        "max_iterations": 3,
        "needs_human_escalation": False,
        "answer_quality": None,
        "conversation_summary": None
    })
    
    return state
```

### 2.2 规划节点

**功能**: 分析用户查询，拆解为任务列表

**代码示例**:
```python
# langgraph_agent_with_memory.py 第694-732行：规划节点
def planning_node(state: AgentState) -> AgentState:
    """规划节点：分析用户查询，拆解为任务列表（带记忆上下文）"""
    import time
    start_time = time.time()
    
    print(f"[规划节点] 开始规划: {state['user_query']}")
    
    # 构建记忆上下文
    memory_context = ""
    if state.get("conversation_summary"):
        memory_context = f"对话历史摘要: {state['conversation_summary']}\n\n"
    
    if state.get("user_preferences"):
        prefs = state["user_preferences"]
        if "language_style" in prefs:
            memory_context += f"用户偏好语言风格: {prefs['language_style']}\n"
        if "detail_level" in prefs:
            memory_context += f"用户偏好详细程度: {prefs['detail_level']}\n"
    
    # 构建规划提示
    prompt = f"""
    {memory_context}
    你是一个企业客服系统的任务规划专家。请将用户的问题拆解为一系列子任务。
    
    可用的任务类型：
    1. "knowledge_search: 具体查询内容" - 从知识库检索信息
    2. "ticket_query: 工单号" - 查询工单状态
    3. "weather_query: 城市名" - 查询天气信息  
    4. "stock_query: 股票代码" - 查询股票价格
    5. "escalate" - 转人工客服
    6. "date_query" - 查询当前日期
    7. "image_analysis: 图片路径" - 分析图像内容
    8. "document_processing: 文档路径" - 处理文档
    
    注意：如果用户查询中包含明确的工单号（如TK-123456），使用ticket_query。
    如果用户需要人工帮助，使用escalate。
    
    示例：
    - 用户问"如何重置密码？" → ["knowledge_search: 重置密码"]
    - 用户问"查询工单 TK-123456" → ["ticket_query: TK-123456"]
    - 用户问"北京天气怎么样" → ["weather_query: 北京"]
    - 用户问"帮我转人工客服" → ["escalate"]
    
    输出要求：
    只输出 JSON 格式的任务列表，不要输出其他内容。
    
    用户问题：{state['user_query']}
    任务列表（JSON）：
    """
    
    # 调用LLM进行规划
    llm = get_llm()
    try:
        response = llm.invoke(prompt).content
        print(f"[规划节点] LLM响应: {response[:200]}...")
        
        # 容错解析
        try:
            tasks = json.loads(response)
        except json.JSONDecodeError:
            # 尝试提取JSON部分
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                tasks = json.loads(match.group())
            else:
                tasks = []
                print(f"[WARN] 无法解析任务列表: {response}")
    except Exception as e:
        print(f"[ERROR] 规划失败: {e}")
        tasks = []
    
    duration = time.time() - start_time
    print(f"[规划节点] 规划完成: {tasks} (耗时: {duration:.2f}s)")
    
    # 根据配置决定下一步节点名称
    if USE_ASYNC_EXECUTION and ASYNC_EXECUTOR_AVAILABLE:
        next_step = "execution_async"
    else:
        next_step = "execution"
    
    # 监控跟踪
    if MONITORING_AVAILABLE and monitoring_system and 'tracking_info' in state:
        try:
            monitoring_system.track_node_execution(
                state['tracking_info'],
                node_name="planning_node",
                inputs={"user_query": state['user_query']},
                outputs={"plan": tasks},
                duration=duration,
                success=len(tasks) > 0
            )
        except Exception as e:
            print(f"[WARN] 规划节点监控跟踪失败: {e}")
    
    return {**state, "plan": tasks, "step": next_step}
```

### 2.3 执行节点

#### 2.3.1 同步执行节点

**代码示例**:
```python
# langgraph_agent_with_memory.py 第710-794行：同步执行节点
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
```

#### 2.3.2 异步执行节点

**代码示例**:
```python
# langgraph_agent_with_memory.py 第796-1065行：异步执行节点
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
            
            # ... 其他工具类型类似
            
            else:
                # 未知任务，直接记录结果
                results[task] = "未知任务"
        
        if tool_calls:
            print(f"[异步执行节点] 准备并行执行 {len(tool_calls)} 个工具调用")
            
            try:
                # 并行执行工具调用
                from async_executor import run_tools_parallel
                parallel_results = run_tools_parallel(tool_calls, timeout=60.0)
                
                # 映射回任务结果
                for task, tool_idx in task_to_tool_map.items():
                    tool_call = tool_calls[tool_idx]
                    tool_name = tool_call["tool_name"]
                    
                    # 查找对应的任务ID
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
                            # 记录成功指标
                            if MONITORING_AVAILABLE and monitoring_system:
                                try:
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
    
    # 监控跟踪
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
```

### 2.4 验证节点

**功能**: 验证答案质量，决定下一步

**代码示例**:
```python
# langgraph_agent_with_memory.py 第1067-1132行：验证节点
def validation_node(state: AgentState) -> AgentState:
    """验证节点：验证答案质量，决定下一步"""
    import time
    start_time = time.time()
    
    print(f"[验证节点] 验证答案质量")
    
    # 合并工具结果
    preliminary_answer = ""
    if state.get("tool_results"):
        for task, result in state["tool_results"].items():
            preliminary_answer += f"{result}\n"
    
    # 基本验证规则
    needs_improvement = False
    needs_human = False
    answer_quality = "good"
    
    # 规则1: 检查答案是否过短
    if len(preliminary_answer.strip()) < 10:
        needs_improvement = True
        answer_quality = "poor"
        print("[验证节点] 答案过短，需要改进")
    
    # 规则2: 检查是否包含不确定表述
    uncertain_keywords = ["无法确定", "不知道", "不清楚", "抱歉", "无法回答", "暂不可用"]
    if any(kw in preliminary_answer for kw in uncertain_keywords):
        needs_human = True
        answer_quality = "uncertain"
        print("[验证节点] 答案包含不确定表述，需要人工介入")
    
    # 规则3: 检查是否包含可信关键词
    trust_keywords = ["工单", "受理", "处理中", "已完成", "今天", "年-月-日", "转人工"]
    if any(kw in preliminary_answer for kw in trust_keywords):
        answer_quality = "excellent"
        print("[验证节点] 答案包含可信关键词")
    
    # 规则4: 基于迭代次数的判断
    if state["iteration"] >= state["max_iterations"]:
        needs_human = True
        answer_quality = "timeout"
        print(f"[验证节点] 达到最大迭代次数 {state['max_iterations']}，需要人工介入")
    
    # 更新状态
    state["answer_quality"] = answer_quality
    state["needs_human_escalation"] = needs_human
    state["iteration"] += 1
    
    # 生成最终答案
    if needs_human:
        final_answer = "抱歉，我无法完全确定答案。建议您转人工客服获取更准确的帮助。"
    elif needs_improvement and state["iteration"] < state["max_iterations"]:
        final_answer = preliminary_answer  # 暂时使用，下次迭代改进
    else:
        final_answer = preliminary_answer
    
    state["final_answer"] = final_answer
    
    duration = time.time() - start_time
    print(f"[验证节点] 验证完成: 质量={answer_quality}, 人工介入={needs_human} (耗时: {duration:.2f}s)")
    
    # 监控跟踪
    if MONITORING_AVAILABLE and monitoring_system and 'tracking_info' in state:
        try:
            monitoring_system.track_node_execution(
                state['tracking_info'],
                node_name="validation_node",
                inputs={"preliminary_answer": preliminary_answer[:100]},
                outputs={
                    "answer_quality": answer_quality,
                    "needs_human_escalation": needs_human,
                    "final_answer": final_answer[:100]
                },
                duration=duration,
                success=True
            )
        except Exception as e:
            print(f"[WARN] 验证节点监控跟踪失败: {e}")
    
    return state
```

### 2.5 后处理节点

**功能**: 最终处理，更新记忆，准备响应

**代码示例**:
```python
# langgraph_agent_with_memory.py 第1134-1193行：后处理节点
def postprocess_node(state: AgentState) -> AgentState:
    """后处理节点：最终处理，更新记忆，准备响应"""
    import time
    start_time = time.time()
    
    print(f"[后处理节点] 最终处理")
    
    # 更新对话记忆
    if state.get("final_answer"):
        # 创建对话摘要
        conversation_exchange = f"用户: {state['user_query']}\n助手: {state['final_answer'][:200]}..."
        
        # 更新对话摘要（简化实现）
        if state.get("conversation_summary"):
            # 合并到现有摘要
            state["conversation_summary"] = f"{state['conversation_summary']}\n{conversation_exchange}"
        else:
            state["conversation_summary"] = conversation_exchange
        
        # 基于交互更新用户偏好（简化实现）
        if state.get("user_preferences") is None:
            state["user_preferences"] = {}
        
        # 分析答案长度，推断用户偏好
        answer_length = len(state["final_answer"])
        if answer_length > 500:
            state["user_preferences"]["detail_level"] = "detailed"
        elif answer_length > 200:
            state["user_preferences"]["detail_level"] = "moderate"
        else:
            state["user_preferences"]["detail_level"] = "concise"
    
    # 更新工作流状态
    state["step"] = "completed"
    
    # 完成监控跟踪
    if "tracking_info" in state:
        tracking_info = state["tracking_info"]
        tracking_info["end_time"] = time.time()
        tracking_info["total_duration"] = tracking_info["end_time"] - tracking_info["start_time"]
        tracking_info["success"] = not state.get("needs_human_escalation", False)
        tracking_info["answer_quality"] = state.get("answer_quality", "unknown")
        
        print(f"[后处理节点] 工作流完成: 总耗时={tracking_info['total_duration']:.2f}s, 质量={tracking_info['answer_quality']}")
    
    duration = time.time() - start_time
    
    # 监控跟踪
    if MONITORING_AVAILABLE and monitoring_system and 'tracking_info' in state:
        try:
            monitoring_system.track_node_execution(
                state['tracking_info'],
                node_name="postprocess_node",
                inputs={"final_answer": state.get("final_answer", "")[:50]},
                outputs={"conversation_summary": state.get("conversation_summary", "")[:50]},
                duration=duration,
                success=True
            )
        except Exception as e:
            print(f"[WARN] 后处理节点监控跟踪失败: {e}")
    
    return state
```

### 2.6 人工介入节点

**功能**: 处理需要人工介入的情况

**代码示例**:
```python
# langgraph_agent_with_memory.py 第1195-1216行：人工介入节点
def human_escalation_node(state: AgentState) -> AgentState:
    """人工介入节点：处理需要人工介入的情况"""
    import time
    start_time = time.time()
    
    print(f"[人工介入节点] 转人工处理")
    
    # 设置人工介入标记
    state["needs_human_escalation"] = True
    state["step"] = "escalate"
    
    # 生成人工介入提示
    human_escalation_message = """
    系统检测到您的问题需要人工客服协助处理。
    
    已为您创建服务工单，人工客服将在5分钟内与您联系。
    
    您的问题摘要：
    - 用户查询: {query}
    - 系统尝试: {attempts}
    - 失败原因: {reason}
    
    感谢您的耐心等待。
    """.format(
        query=state["user_query"],
        attempts=f"{state['iteration']}次尝试",
        reason=state.get("answer_quality", "答案质量不足")
    )
    
    state["final_answer"] = human_escalation_message
    
    duration = time.time() - start_time
    
    # 监控跟踪
    if MONITORING_AVAILABLE and monitoring_system and 'tracking_info' in state:
        try:
            monitoring_system.track_node_execution(
                state['tracking_info'],
                node_name="human_escalation_node",
                inputs={"user_query": state["user_query"]},
                outputs={"escalation_message": human_escalation_message[:100]},
                duration=duration,
                success=True
            )
        except Exception as e:
            print(f"[WARN] 人工介入节点监控跟踪失败: {e}")
    
    return state
```

## 3. 条件路由函数

### 3.1 验证后路由

**代码示例**:
```python
# langgraph_agent_with_memory.py 第1218-1232行：条件路由函数
def route_after_validation(state: AgentState) -> str:
    """验证后的条件路由：决定下一步"""
    
    if state.get("needs_human_escalation", False):
        print("[路由] 需要人工介入 -> human_escalation")
        return "human_escalation"
    
    # 检查是否需要改进
    answer_quality = state.get("answer_quality", "unknown")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    
    if answer_quality in ["poor", "uncertain"] and iteration < max_iterations:
        print(f"[路由] 答案质量{answer_quality}，第{iteration}次迭代，重新规划 -> improvement")
        return "improvement"
    else:
        print("[路由] 答案质量可接受或达到最大迭代次数 -> postprocess")
        return "postprocess"
```

## 4. 工作流编译与执行

### 4.1 工作流编译

**代码示例**:
```python
# 工作流编译示例
from langgraph.checkpoint.memory import MemorySaver

def create_and_compile_workflow(use_async: bool = True):
    """创建并编译工作流"""
    # 创建图
    workflow = create_workflow(use_async_execution=use_async)
    
    # 创建记忆检查点
    memory = MemorySaver()
    
    # 编译为可执行应用
    app = workflow.compile(checkpointer=memory)
    
    return app

# 使用示例
app = create_and_compile_workflow(use_async=True)
```

### 4.2 工作流执行

**代码示例**:
```python
# 工作流执行示例
def execute_workflow(user_query: str, thread_id: str = "default"):
    """执行工作流"""
    # 初始状态
    initial_state = {
        "user_query": user_query,
        "messages": [],
        "user_preferences": {},
        "plan": None,
        "tool_results": None,
        "final_answer": None,
        "step": "planning",
        "iteration": 0,
        "max_iterations": 3,
        "needs_human_escalation": False,
        "answer_quality": None,
        "conversation_summary": None,
        "tracking_info": {}  # 初始化为空
    }
    
    # 配置
    config = {"configurable": {"thread_id": thread_id}}
    
    # 执行
    try:
        final_state = app.invoke(initial_state, config=config)
        return final_state
    except Exception as e:
        print(f"[ERROR] 工作流执行失败: {e}")
        return None
```

## 5. 记忆系统集成

### 5.1 记忆管理器

**代码示例**:
```python
# langgraph_agent_with_memory.py 第291-348行：记忆管理器
class MemoryManager:
    def __init__(self):
        self.short_term_memory = []  # 短期对话记忆（最近N轮）
        self.user_preferences = {}   # 用户偏好记忆
        self.topic_tracker = {}      # 话题追踪
        
    def adapt_response(self, response: str) -> str:
        """根据用户偏好和对话历史调整响应"""
        adapted_response = response
        
        # 基于语言风格偏好调整
        style = self.user_preferences.get("language_style", "professional")
        if style == "casual":
            adapted_response = self._make_casual(adapted_response)
        elif style == "detailed":
            adapted_response = self._add_details(adapted_response)
        
        # 基于详细程度偏好调整
        detail_level = self.user_preferences.get("detail_level", "moderate")
        if detail_level == "concise":
            adapted_response = self._make_concise(adapted_response)
        elif detail_level == "detailed":
            adapted_response = self._add_examples(adapted_response)
        
        # 基于历史对话避免重复
        adapted_response = self._avoid_repetition(adapted_response)
        
        return adapted_response
    
    def update_from_conversation(self, user_query: str, assistant_response: str):
        """从对话中更新记忆"""
        # 更新短期记忆
        self.short_term_memory.append({
            "user": user_query,
            "assistant": assistant_response,
            "timestamp": time.time()
        })
        
        # 保持短期记忆大小
        if len(self.short_term_memory) > 10:
            self.short_term_memory.pop(0)
        
        # 分析并更新用户偏好
        self._analyze_user_preferences(user_query, assistant_response)
        
        # 更新话题追踪
        self._update_topic_tracker(user_query)
```

## 6. 可视化与调试

### 6.1 工作流可视化

**代码示例**:
```python
# 工作流可视化（概念代码）
def visualize_workflow():
    """可视化工作流"""
    workflow = create_workflow()
    
    # 导出为Mermaid图表
    mermaid_code = workflow.get_graph().draw_mermaid()
    
    # 或者导出为图片
    workflow.get_graph().draw_png("workflow.png")
    
    return mermaid_code

# 在Web界面中显示
# app_simple.py 中集成了工作流可视化
```

### 6.2 调试信息收集

**代码示例**:
```python
# 调试信息收集
def collect_debug_info(state: AgentState) -> Dict[str, Any]:
    """收集调试信息"""
    debug_info = {
        "current_step": state.get("step", "unknown"),
        "iteration": state.get("iteration", 0),
        "plan": state.get("plan", []),
        "tool_results": state.get("tool_results", {}),
        "answer_quality": state.get("answer_quality", "unknown"),
        "needs_human": state.get("needs_human_escalation", False),
        "conversation_summary": state.get("conversation_summary", "")[:100],
        "user_preferences": state.get("user_preferences", {}),
        "tracking_info": state.get("tracking_info", {}).copy() if state.get("tracking_info") else {}
    }
    
    # 匿名化敏感信息
    if "user_query" in state:
        debug_info["user_query_hash"] = hash(state["user_query"])
    
    return debug_info
```

## 7. 学习总结

### 关键优势
1. **可视化流程**: 清晰的工作流定义和可视化
2. **状态管理**: 类型安全的状态管理和传递
3. **条件分支**: 灵活的路由和循环控制
4. **记忆集成**: 内置的记忆和检查点机制
5. **监控友好**: 易于集成监控和调试

### 设计模式
1. **节点化设计**: 每个节点单一职责
2. **状态驱动**: 状态字典传递所有信息
3. **配置驱动**: 通过配置控制行为
4. **错误隔离**: 节点间错误不传播

### 最佳实践
1. **状态设计**: 完整的类型定义，包含所有必要字段
2. **节点职责**: 每个节点只做一件事
3. **错误处理**: 节点内部处理错误，不中断工作流
4. **监控集成**: 每个节点添加监控跟踪

### 常见陷阱
1. **状态过于复杂**: 状态字典过于庞大难以管理
2. **节点耦合**: 节点间依赖过强
3. **缺少超时**: 工作流可能卡住
4. **记忆泄漏**: 长期运行的内存问题

---

**相关文件**:
- [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) - 完整LangGraph实现
- [langgraph_agent.py](e:\my_multi_agent\langgraph_agent.py) - 简化版LangGraph
- [langgraph_agent_simple.py](e:\my_multi_agent\langgraph_agent_simple.py) - 最简版LangGraph
- [app_simple.py](e:\my_multi_agent\app_simple.py) - Web界面集成

**下一步学习**: 异步与性能优化 →