# LangChain与工具系统

## 概述
本项目使用LangChain标准化的工具调用协议（`bind_tools` + `ToolMessage`），LLM自主理解工具用途并决定调用时机。工具注册遵循"定义 + 列表注册"两步模式，扩展成本极低。

## 1. LLM集成

### 1.1 ChatTongyi（阿里百炼）

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第217-228行

```python
from langchain_community.chat_models import ChatTongyi

_llm = None

def get_llm():
    """获取 LLM 实例（单例模式）"""
    global _llm
    if _llm is None:
        _llm = ChatTongyi(
            model="qwen-plus",
            temperature=0,
            dashscope_api_key=api_key
        )
    return _llm
```

**关键参数**:
- `model="qwen-plus"`: 通义千问模型
- `temperature=0`: 确定性输出（客服场景需要一致性）
- 单例模式：全局共享一个LLM实例，按需初始化

## 2. 标准工具调用协议

### 2.1 bind_tools —— LLM自主决策

**这是v2.0的核心机制**，替代了旧版的JSON手动规划+if-else分发。

```python
# langgraph_agent_with_memory.py 第505-506行
llm = get_llm()
tools = get_tools()
llm_with_tools = llm.bind_tools(tools)  # 将工具Schema注入LLM
```

**`bind_tools` 做了什么**:
- 将每个 `@tool` 函数的name、description、参数类型注入LLM的上下文
- LLM根据用户问题自主判断：是否需要工具、调用哪个工具、传什么参数
- LLM返回标准化的 `response.tool_calls` 列表

### 2.2 ReAct循环中的工具调用

```python
# langgraph_agent_with_memory.py 第524-584行
for step_idx in range(MAX_AGENT_STEPS):
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        # LLM认为不需要工具 → 这就是最终回答
        final_answer = response.content
        return {**state, "final_answer": final_answer, ...}

    # 执行LLM请求的工具
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})

        # 执行工具
        if tool_name == "knowledge_search":
            result = knowledge_search.run(tool_args.get("query", ""))
        elif tool_name == "query_ticket_status":
            result = query_ticket_status.run(tool_args.get("ticket_id", ""))
        # ...

        # 将观察结果反馈给LLM
        messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
```

**消息流转**:
```
SystemMessage → HumanMessage → AIMessage(tool_calls=[...]) → ToolMessage(result) → AIMessage(tool_calls=[]) → 最终答案
```

### 2.3 对比：v1.0手动分发 vs v2.0标准协议

**v1.0（旧版，已废弃）**:
```python
# 人工写JSON提示词 → LLM输出JSON → 手动解析 → if-else分支
tasks = planning_agent(query)  # 返回 ["knowledge_search: 重置密码"]
for task in tasks:
    if task.startswith("knowledge_search:"):
        result = knowledge_search.run(query)
    elif task.startswith("ticket_query:"):
        result = query_ticket_status.run(ticket_id)
    # ... 每加一个新工具就要加elif分支
```

**v2.0（当前）**:
```python
# LLM通过bind_tools自主决策 → 标准tool_calls协议 → 统一执行
response = llm_with_tools.invoke(messages)
for tool_call in response.tool_calls:
    result = execute_tool_by_name(tool_call["name"], tool_call["args"])
    messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
```

## 3. 工具定义方式

### 3.1 @tool装饰器（推荐）

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第328-458行

```python
from langchain_core.tools import tool

@tool
def knowledge_search(query: str) -> str:
    """从中国移动知识库中检索信息，返回答案。适用于套餐资费、5G业务、宽带、物联网、
    云计算、算力网络、网络安全、政企服务等中国移动相关业务咨询。"""
    # 实现...

@tool
def query_ticket_status(ticket_id: str) -> str:
    """模拟查询工单状态"""
    # 实现...

@tool
def escalate_to_human(query: str) -> str:
    """模拟转人工处理"""
    # 实现...

@tool
def get_current_date(query: str) -> str:
    """返回今天的日期"""
    # 实现...
```

### 3.2 工具注册（扩展入口）

```python
# langgraph_agent_with_memory.py 第461-473行
AGENT_TOOLS = []

def get_tools():
    """获取 Agent 工具列表（延迟初始化）"""
    global AGENT_TOOLS
    if not AGENT_TOOLS:
        AGENT_TOOLS = [
            knowledge_search,
            query_ticket_status,
            escalate_to_human,
            get_current_date,
        ]
    return AGENT_TOOLS
```

**新增工具只需两步**:
1. 定义 `@tool` 函数（含清晰的docstring）
2. 在 `AGENT_TOOLS` 列表中添加函数引用

LLM会自动理解新工具的用途并在合适的时机调用它。

## 4. 工具分类

### 4.1 检索类工具

```python
@tool
def knowledge_search(query: str) -> str:
    """从中国移动知识库中检索信息..."""
    # 高级RAG检索 → 如果不可用则回退到模拟模式
```

使用 `advanced_rag_retriever`（向量+B M25混合检索+重排序），不可用时回退到模拟响应。

### 4.2 查询类工具

```python
@tool
def query_ticket_status(ticket_id: str) -> str:
    """模拟查询工单状态"""
    mock_status = {
        "TK-123456": "您的工单 TK-123456 已受理...",
        "TK-789012": "工单 TK-789012 已处理完毕...",
        "default": "未找到工单信息..."
    }
```

### 4.3 服务类工具

```python
@tool
def escalate_to_human(query: str) -> str:
    """模拟转人工处理"""
```

### 4.4 工具类工具

```python
@tool
def get_current_date(query: str) -> str:
    """返回今天的日期"""
```

## 5. 系统提示词

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) 第477-492行

```python
SYSTEM_PROMPT = """你是一名中国移动智能客服助手。

## 你的工具
- knowledge_search: 从中国移动知识库检索信息。参数query请使用简洁关键词。
- query_ticket_status: 查询工单状态。工单号格式为TK-xxxxxx。
- escalate_to_human: 转人工客服。
- get_current_date: 查询今天的日期。

## 行为准则
1. 对于业务咨询，先调用knowledge_search检索，再基于检索结果回答
2. 用亲切、专业的语气回复...
"""
```

**提示设计要点**:
1. 明确每个工具的参数使用方式（如"query请使用简洁关键词"）
2. 规定行为优先级（先检索再回答）
3. 约束LLM行为边界（不编造信息）

## 6. 工具设计最佳实践

### 6.1 docstring是关键文档
LLM通过docstring理解工具用途。docstring应包含：功能描述、适用场景、参数说明。

```python
@tool
def knowledge_search(query: str) -> str:
    """从中国移动知识库中检索信息，返回答案。
    适用于套餐资费、5G业务、宽带、物联网、云计算等中国移动相关业务咨询。"""
```

### 6.2 单一职责
每个工具只做一件事。不要在一个工具中混合检索 + 查询 + 转人工。

### 6.3 优雅降级
```python
@tool
def knowledge_search(query: str) -> str:
    if advanced_rag_retriever:
        try:
            results = advanced_rag_retriever.retrieve(query)
            if results:
                return format_results(results)
        except Exception as e:
            print(f"高级RAG失败: {e}")
    # 回退到模拟模式
    return mock_responses.get(key, mock_responses["默认"])
```

### 6.4 延迟初始化
```python
_llm = None
_advanced_rag_retriever = None

def get_llm():      # 按需初始化，不阻塞模块导入
def get_tools():    # 延迟初始化工具列表
```

## 7. 学习总结

### 关键知识点
1. **bind_tools协议**: LLM自主理解工具并决策调用时机（替代手动规划）
2. **ToolMessage反馈**: 标准化工具结果回传机制
3. **两步注册模式**: `@tool` 定义 + `AGENT_TOOLS` 列表 = 完成注册
4. **docstring即文档**: LLM根据docstring理解工具用途

### 最佳实践
1. **清晰的docstring**: 描述工具功能和适用场景
2. **单一职责**: 每个工具只做一件事
3. **优雅降级**: 外部服务不可用时提供兜底
4. **延迟加载**: 全局资源按需初始化

### v2.0改进总结
- 删除了旧版 `ChatPromptTemplate` 手动JSON提示 + 正则解析
- 删除了 `planning_agent` / `execution_agent` 的if-else分支
- 统一为标准 `bind_tools` + `ToolMessage` 协议
- 新增工具成本从"改3处"降为"改2行"

---

**相关文件**:
- [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) — 完整工具定义和ReAct调用
- [advanced_rag_system.py](e:\my_multi_agent\advanced_rag_system.py) — 高级RAG检索器

**下一步学习**: RAG技术 →
