# LangGraph工作流系统

## 概述
LangGraph是一个用于构建有状态、多智能体应用的库。本项目使用LangGraph构建了一个**标准ReAct Agent工作流**，LLM自主决定工具调用，观察结果后迭代推理，最终生成答案。

## 1. 工作流架构（3节点线性）

**当前架构**:

```
用户输入 → preprocess → agent (ReAct循环) → postprocess → 输出
```

- **preprocess**: 处理用户输入，更新记忆，生成对话摘要
- **agent**: 标准ReAct循环节点，LLM自主推理 + 工具调用
- **postprocess**: 更新记忆，调整响应风格，监控跟踪

### 架构演进说明
v1.0采用6节点流水线（preprocess → planning → execution → validation → postprocess），v2.0简化为3节点标准ReAct架构。工具调用决策由LLM自主完成，不再需要独立的规划/执行/验证节点。

## 2. 核心组件

### 2.1 AgentState（工作流状态）

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第72-85行

```python
class AgentState(TypedDict):
    """标准 ReAct Agent 的工作流状态"""
    user_query: str
    messages: Annotated[list, add_messages]
    final_answer: Optional[str]
    raw_context: Optional[str]       # RAG原始检索上下文（展示用）
    tool_results: Optional[Dict[str, str]]  # 工具调用记录
    plan: Optional[List[str]]        # 已执行的工具列表
    step: Literal["agent", "completed"]
    iteration: int
    max_iterations: int
    conversation_summary: Optional[str]
    user_preferences: Dict[str, Any]
    tracking_info: Optional[Dict[str, Any]]
```

**设计要点**:
- 移除了旧版的 `needs_human_escalation` 和 `answer_quality` 字段（质量判断由LLM在ReAct循环中自主完成）
- `messages` 用于LangGraph消息累积（支持多轮对话）
- `raw_context` 保留RAG检索原始结果，用于前端展示

### 2.2 工作流创建

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第703-720行

```python
def create_workflow() -> StateGraph:
    """创建标准 ReAct Agent 工作流"""
    workflow = StateGraph(AgentState)

    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("postprocess", postprocess_node)

    workflow.set_entry_point("preprocess")
    workflow.add_edge("preprocess", "agent")
    workflow.add_edge("agent", "postprocess")
    workflow.add_edge("postprocess", END)

    return workflow
```

**关键特点**:
- 3节点线性图，无条件分支（v2.0简化）
- ReAct逻辑全部封装在 `agent_node` 内部，不暴露为独立图节点
- 编译时注入 `MemorySaver` 检查点，支持会话持久化

## 3. 节点详解

### 3.1 预处理节点（preprocess_node）

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第616-647行

**功能**: 接收用户输入，更新记忆管理器，生成对话摘要，初始化监控跟踪

```python
def preprocess_node(state: AgentState) -> AgentState:
    memory_manager = get_memory_manager()
    memory_manager.add_message("user", state["user_query"])
    conversation_summary = memory_manager.generate_summary()
    # 监控跟踪...
    return {**state, "conversation_summary": conversation_summary, "step": "planning", ...}
```

### 3.2 Agent节点（agent_node）—— 核心ReAct循环

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第497-612行

**这是整个系统的核心**，实现标准ReAct（Reasoning + Acting）循环：

```python
def agent_node(state: AgentState) -> AgentState:
    llm = get_llm()
    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)  # 关键：绑定工具让LLM自主决策

    # 构建消息：系统提示 + 对话历史 + 用户问题
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if conversation_summary:
        messages.append(SystemMessage(content=f"对话历史摘要：{conversation_summary}"))
    messages.append(HumanMessage(content=state["user_query"]))

    # ReAct迭代循环（最多MAX_AGENT_STEPS=5轮）
    for step_idx in range(MAX_AGENT_STEPS):
        response = llm_with_tools.invoke(messages)  # LLM推理
        messages.append(response)

        if not response.tool_calls:
            # LLM认为不需要更多工具 → 这是最终回答
            return {**state, "final_answer": response.content, ...}

        # 执行工具调用，收集观察结果
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            result = execute_tool_by_name(tool_name, tool_args)
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    # 达到最大步数 → 强制输出
    ...
```

**ReAct循环流程**:
1. LLM接收系统提示 + 用户问题 → 推理是否需要工具
2. 如果需要工具 → `response.tool_calls` 包含工具名和参数
3. 执行工具 → 将结果作为 `ToolMessage` 追加到消息列表
4. LLM重新推理（观察工具结果后可能继续调用工具或给出最终答案）
5. 循环直到LLM输出无 `tool_calls` 的回复，或达到最大步数

**与v1.0（旧版）的关键差异**:
| 维度 | v1.0 (旧版) | v2.0 (当前) |
|------|-----------|-----------|
| 工具决策 | planning_node解析JSON → execution_node手动if-else分发 | LLM通过 `bind_tools` 自主决定 |
| 工具执行 | 硬编码的 `if task.startswith("xxx")` 分支 | 标准 `response.tool_calls` 协议 |
| 质量验证 | 独立 validation_node | LLM在ReAct循环中自主判断 |
| 条件路由 | `route_after_validation` 条件边 | 无需条件路由（线性图） |
| 节点数量 | 6个节点 | 3个节点 |

### 3.3 后处理节点（postprocess_node）

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第650-700行

**功能**: 将助手回答写入记忆，根据用户偏好调整响应风格（正式/休闲、简洁/详细），完成监控跟踪

```python
def postprocess_node(state: AgentState) -> AgentState:
    memory_manager.add_message("assistant", final_answer)
    memory_manager.update_preferences(state["user_query"], final_answer)
    adapted_answer = memory_manager.adapt_response(final_answer)
    # 监控跟踪...
    return {**state, "final_answer": adapted_answer, ...}
```

## 4. 记忆系统

### 4.1 MemoryManager

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第89-203行

- **短期记忆**: 最近10轮对话历史 (`conversation_history`)
- **用户偏好**: 语言风格（正式/休闲/中性）、详细程度（简洁/适中/详细）
- **对话摘要**: 统计话题分布和消息数量
- **响应适配**: 根据偏好调整Agent回复的措辞和长度

## 5. 工作流编译与执行

### 5.1 编译

```python
workflow = create_workflow()
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

### 5.2 执行入口

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第724-789行

```python
def run_langgraph_agent_with_memory(user_query: str, max_iterations: int = 3):
    initial_state: AgentState = {
        "user_query": user_query,
        "messages": [],
        "user_preferences": memory_manager.user_preferences,
        ...
    }
    final_state = app.invoke(initial_state, config)
    return {
        "plan": final_state.get("plan"),          # 已执行的工具列表
        "tool_results": final_state.get("tool_results"),
        "final_answer": final_state.get("final_answer"),
        "raw_context": final_state.get("raw_context"),
        "memory_info": memory_info
    }
```

## 6. 可视化与调试

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第791-813行

```python
def visualize_workflow(output_path: str = "workflow_with_memory.png"):
    app = workflow.compile()
    png_data = app.get_graph().draw_mermaid_png()
    mermaid_code = app.get_graph().draw_mermaid()
```

## 7. 学习总结

### 架构优势
1. **LLM自主决策**: 工具调用不再需要硬编码的JSON解析和if-else分发
2. **标准化协议**: 使用LangChain标准的 `bind_tools` + `ToolMessage` 协议
3. **简洁可维护**: 3节点代替旧版6节点，代码量减少51%（1581→949行）
4. **扩展性**: 新增工具只需 `@tool` + 加入 `AGENT_TOOLS` 列表，无需修改任何工作流节点

### 核心流程总结
```
用户问题 → preprocess(记忆上下文) → agent(LlM + bind_tools → ReAct循环 → 最终回答) → postprocess(风格适配) → 返回
```

### 最佳实践
1. **工具设计**: 清晰的docstring帮助LLM正确选择工具
2. **系统提示**: 明确的行为准则和工具使用指导
3. **步数限制**: `MAX_AGENT_STEPS=5` 防止无限循环
4. **强制输出**: 达到最大步数时用简单LLM调用生成兜底回复

---

**相关文件**:
- [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) — 完整的3节点LangGraph工作流实现
- [app.py](e:\my_multi_agent\app.py) — Streamlit Web界面

**下一步学习**: 异步与性能优化 →
