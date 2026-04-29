# Agent基础概念

## 概述
Agent（智能体）是一个能够感知环境、做出决策并执行动作以实现目标的系统。本项目采用**标准ReAct Agent架构**，LLM通过 `bind_tools` 绑定工具后自主完成推理（Reasoning）和行动（Acting）的循环。

## 1. Agent架构：标准ReAct

### 1.1 核心思想

ReAct = **Rea**soning + **A**cting：LLM自主决定"是否需要工具"、"调用哪个工具"、"何时给用户最终回答"。

**核心文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py)

**ReAct循环流程**:
```
用户输入 → System提示 → LLM推理
                         ↓
                  需要工具？ ──是──→ 调用工具 → 观察结果 → 回到LLM推理
                         ↓ 否
                      最终回答
```

### 1.2 代码实现

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第497-612行

```python
def agent_node(state: AgentState) -> AgentState:
    """标准 ReAct Agent 节点"""
    llm = get_llm()
    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)  # 关键：让LLM知道有哪些工具可用

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.append(HumanMessage(content=state["user_query"]))

    for step_idx in range(MAX_AGENT_STEPS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # LLM决定直接回复 → 最终答案
            return {**state, "final_answer": response.content, ...}

        # 执行LLM请求的工具调用，将结果作为ToolMessage返回
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call)
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
```

**关键机制**:
1. `llm.bind_tools(tools)`: 将工具列表注册给LLM，LLM自动理解每个工具的用途和参数
2. `response.tool_calls`: LLM返回工具调用请求（工具名 + 参数），由代码执行
3. `ToolMessage`: 工具执行结果反馈给LLM，LLM据此继续推理
4. 循环结束条件：LLM返回无 `tool_calls` 的普通文本回复

### 1.3 工具注册（新增工具只需两步）

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第328-473行

```python
# 步骤1：定义 @tool 函数
@tool
def knowledge_search(query: str) -> str:
    """从中国移动知识库中检索信息..."""
    # 实现...

# 步骤2：加入 AGENT_TOOLS 列表
AGENT_TOOLS = [knowledge_search, query_ticket_status, escalate_to_human, get_current_date]
```

LLM会自动根据docstring理解工具用途，无需编写JSON解析或if-else分发逻辑。

## 2. Agent的核心组件

### 2.1 感知层（输入处理）

**preprocess_node** 负责接收用户输入，加载对话记忆上下文：

```python
# langgraph_agent_with_memory.py 第616-647行
def preprocess_node(state: AgentState) -> AgentState:
    memory_manager.add_message("user", state["user_query"])
    conversation_summary = memory_manager.generate_summary()
    return {**state, "conversation_summary": conversation_summary, ...}
```

### 2.2 推理+行动层（ReAct核心）

**agent_node** 内部完成全部推理和工具调用（见1.2节）。与旧版不同，规划和执行不再分离——LLM在一个循环中自主完成。

### 2.3 响应层（后处理）

**postprocess_node** 负责记忆更新和风格适配：

```python
# langgraph_agent_with_memory.py 第650-700行
def postprocess_node(state: AgentState) -> AgentState:
    memory_manager.add_message("assistant", final_answer)
    memory_manager.update_preferences(user_query, final_answer)
    adapted_answer = memory_manager.adapt_response(final_answer)
    return {**state, "final_answer": adapted_answer, ...}
```

## 3. 系统提示词设计

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第477-492行

```python
SYSTEM_PROMPT = """你是一名中国移动智能客服助手。

## 你的工具
- knowledge_search: 从中国移动知识库检索信息。参数query使用简洁关键词。
- query_ticket_status: 查询工单状态。工单号格式为TK-xxxxxx。
- escalate_to_human: 转人工客服。
- get_current_date: 查询今天的日期。

## 行为准则
1. 对于业务咨询，先调用knowledge_search检索，再基于检索结果回答
2. 用亲切、专业的语气回复，称呼用户为"您"
3. 只基于检索结果回答，不要编造没有的信息
4. 结尾可以引导用户进一步提问
"""
```

**设计要点**:
1. 清晰列出每个工具的用途和调用时机
2. 明确行为约束（不要编造信息、使用检索结果）
3. 规定回复风格（亲切专业、称呼"您"）

## 4. 工作流图（LangGraph）

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第703-720行

```
preprocess → agent (ReAct循环) → postprocess → END
```

3节点线性图，无条件分支。完整的ReAct逻辑封装在 `agent_node` 内部。

## 5. 与v1.0（旧版多Agent流水线）的对比

| 维度 | v1.0 (旧版，已废弃) | v2.0 (当前标准ReAct) |
|------|-------------------|---------------------|
| 架构 | 规划Agent → 执行Agent → 验证Agent | 单一ReAct Agent (agent_node) |
| 工具决策 | 人工编写JSON提示 → 手动解析 → if-else分发 | `llm.bind_tools(tools)` 自动决策 |
| 工具执行 | `if task.startswith("knowledge_search"): ...` | 标准 `response.tool_calls` 协议 |
| 新增工具 | 需修改planning提示 + execution分支 + 验证规则 | 只需 `@tool` + 加入 `AGENT_TOOLS` |
| 质量验证 | 独立validation_node + 硬编码规则 | LLM在ReAct循环中自主判断 |
| 节点数 | 6个节点 + 条件路由 | 3个节点，无分支 |
| 文件 | rag_agent.py, enterprise_agent.py（已删除） | langgraph_agent_with_memory.py |

## 6. 关键技术要点

### 6.1 bind_tools机制
`llm.bind_tools(tools)` 将工具的name、description、参数schema注入LLM的上下文，LLM据此生成标准的 `tool_calls`。这是LangChain标准化的Function Calling协议。

### 6.2 ToolMessage协议
工具执行结果通过 `ToolMessage(content=result, tool_call_id=id)` 返回给LLM。LLM通过 `tool_call_id` 关联请求和响应。

### 6.3 ReAct循环控制
- `MAX_AGENT_STEPS = 5`：防止无限循环
- 达到最大步数时：使用简单LLM调用生成兜底回复
- LLM返回无 `tool_calls` 的回复时：循环正常结束

## 7. 学习总结

### 关键收获
1. **标准ReAct优于自定义流水线**: LLM自主决策比硬编码的JSON解析更灵活、更准确
2. **bind_tools是关键**: 让LLM理解工具并自主决策调用时机
3. **简化即优化**: 3节点替代6节点，代码量减半，可维护性大幅提升
4. **扩展成本极低**: 新增工具只需定义函数+注册列表

### 实践建议
1. **写好docstring**: LLM根据docstring理解工具用途，这是最关键的文档
2. **系统提示明确**: 告诉LLM什么时候该用哪个工具
3. **兜底策略**: 设置最大步数和超时机制
4. **记忆管理**: 保持对话上下文，避免重复提问

### 进阶方向
1. **流式响应**: 在ReAct循环中支持token级别的流式输出
2. **多轮自主修正**: LLM发现检索结果不够时自主重新检索
3. **工具组合**: LLM在一次推理中调用多个工具并整合结果

---

**相关文件**:
- [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) — 核心Agent实现（唯一的主Agent文件）
- [app.py](e:\my_multi_agent\app.py) — Streamlit Web界面

**下一步学习**: LangChain工具系统 →
