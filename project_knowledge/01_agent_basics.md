# Agent基础概念

## 概述
Agent（智能体）是一个能够感知环境、做出决策并执行动作以实现目标的系统。在本项目中，Agent特指基于大语言模型的智能体，能够理解自然语言查询、规划解决方案、调用工具执行任务、整合结果生成最终答案。

## Agent分类体系

### 1. 按架构分类

#### 1.1 单一Agent（基础版本）
**文件**: [rag_agent.py](e:\my_multi_agent\rag_agent.py)

**核心思想**: 单一决策主体直接处理所有任务

**代码示例**:
```python
# rag_agent.py 第5-11行：关键导入
from langchain_core.tools import tool, Tool
from langchain.agents import AgentExecutor, create_react_agent

# rag_agent.py 第108-114行：ReAct Agent创建
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
```

**特点**:
- 使用标准ReAct提示模板
- 内置工具调用循环
- 自动处理解析错误
- 详细执行过程输出（verbose=True）

#### 1.2 多Agent系统（进阶版本）
**文件**: [enterprise_agent.py](e:\my_multi_agent\enterprise_agent.py)

**核心思想**: 多个专业Agent协作，分工明确

**代码示例**:
```python
# enterprise_agent.py 第291-307行：多Agent主流程
def run_multi_agent(user_query: str) -> dict:
    """多Agent协作主流程"""
    # 1. 规划阶段
    tasks = planning_agent(user_query)
    print(f"[规划] 任务清单: {tasks}")
    
    # 2. 执行阶段
    exec_results = execution_agent(tasks)
    preliminary = "\n".join(exec_results.values()) if exec_results else "无结果"
    print(f"[执行] 合并答案: {preliminary}")
    
    # 3. 验证阶段
    final = validation_agent(user_query, preliminary)
    print(f"[验证] 最终答案: {final}")
    
    return {
        "tasks": tasks,
        "exec_results": exec_results,
        "final_answer": final
    }
```

**特点**:
- 完全自定义的Agent协作流程
- 显式的阶段划分（规划→执行→验证）
- 细粒度的过程控制
- 丰富的中间状态记录

### 2. 按工作模式分类

#### 2.1 ReAct模式Agent
**概念**: Reasoning（推理） + Acting（行动）

**核心流程**:
```
用户输入 → Prompt模板 → LLM推理 → 工具调用 → 结果整合 → 最终输出
```

**代码示例**:
```python
# ReAct提示模板结构（简化）
"""
Question: {input}
Thought: {agent_scratchpad}  # 思考过程
Action: {tool_names}         # 选择工具
Action Input: {input}        # 工具输入
Observation: {result}        # 工具结果
...（可重复多次）
Thought: I now know the final answer
Final Answer: {answer}       # 最终答案
"""
```

#### 2.2 Function Calling Agent
**概念**: 直接调用预定义函数，隐式推理

**代码示例**:
```python
# enterprise_agent.py 第80-92行：工具函数定义
@tool
def knowledge_search(query: str) -> str:
    """从企业知识库中检索信息，返回答案。"""
    qa_chain = get_qa_chain()
    if qa_chain is None:
        return "知识库服务不可用，请稍后再试。"
    result = qa_chain.invoke({"query": query})
    return result["result"]
```

### 3. Agent的核心组件

#### 3.1 感知器（Perceptor）
**功能**: 理解用户输入和环境状态

**代码示例**:
```python
# enterprise_agent.py 第188-217行：预处理函数
def preprocess_query(query: str) -> tuple:
    """预处理用户查询，提取关键信息"""
    # 提取工单号模式
    ticket_pattern = r'TK-\d{6}'
    ticket_match = re.search(ticket_pattern, query)
    
    # 识别查询类型
    query_type = "unknown"
    if "工单" in query or ticket_match:
        query_type = "ticket_query"
    elif "密码" in query or "重置" in query:
        query_type = "knowledge_search"
    elif "转人工" in query:
        query_type = "escalate"
    
    return query_type, ticket_match.group() if ticket_match else None
```

#### 3.2 规划器（Planner）
**功能**: 制定行动方案，拆解复杂任务

**代码示例**:
```python
# enterprise_agent.py 第219-264行：规划Agent
def planning_agent(query: str) -> list:
    """规划Agent：分析用户查询，拆解为任务列表"""
    prompt = ChatPromptTemplate.from_template("""
    你是一个企业客服系统的任务规划专家...
    
    # 任务类型定义
    - "knowledge_search: 具体查询内容"   # 从知识库检索
    - "ticket_query: 工单号"           # 查询工单状态
    - "escalate"                        # 转人工
    - "date_query"                      # 查询日期
    
    # 输出约束
    只输出 JSON 格式的任务列表，不要输出其他内容。
    
    用户问题：{query}
    任务列表（JSON）：
    """)
    
    chain = prompt | get_llm()
    response = chain.invoke({"query": query}).content
    
    # 容错解析设计
    try:
        tasks = json.loads(response)
    except:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        tasks = json.loads(match.group()) if match else []
    
    return tasks
```

#### 3.3 执行器（Executor）
**功能**: 调用工具执行动作

**代码示例**:
```python
# enterprise_agent.py 第266-289行：执行Agent
def execution_agent(tasks: list) -> dict:
    """执行Agent：按规划调用工具，收集结果"""
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
```

#### 3.4 评估器（Evaluator）
**功能**: 检查结果质量，确保答案可信可靠

**代码示例**:
```python
# enterprise_agent.py 第309-327行：验证Agent
def validation_agent(user_query: str, final_answer: str) -> str:
    """验证Agent：验证答案质量，提供兜底方案"""
    # 基于规则的验证
    trust_keywords = ["工单", "受理", "处理中", "已完成", "今天", "年-月-日", "转人工"]
    
    if any(kw in final_answer for kw in trust_keywords):
        return final_answer
    
    # 长度验证和负面词过滤
    if len(final_answer) < 5 or "无法确定" in final_answer:
        return "抱歉，我无法确定准确答案。建议您转人工客服。"
    
    return final_answer
```

### 4. Agent协作模式

#### 4.1 流水线模式（本项目采用）
```
用户 → 规划Agent → 执行Agent → 验证Agent → 用户
```

**优点**:
- 线性执行，简单可靠
- 每个Agent职责单一
- 易于调试和监控
- 适合确定性的任务处理

#### 4.2 黑板模式（可选方案）
```
用户 → 控制Agent → 用户
          ↑ ↓
     共享状态黑板
   ↑         ↑         ↑
Agent1   Agent2   Agent3
```

**特点**:
- 所有Agent访问共享状态
- 更灵活的协作方式
- 适合动态任务分配

### 5. 共享状态设计

**代码示例**:
```python
# enterprise_agent.py 第16-23行：共享状态定义
shared_state = {
    "user_query": "",       # 原始用户问题
    "plan": None,           # 规划Agent输出的任务列表
    "retrieved_docs": [],   # 检索到的文档
    "tool_results": {},     # 各工具执行结果
    "final_answer": None    # 验证后的最终答案
}
```

**设计考虑**:
1. **状态完整性**: 记录全流程关键数据
2. **调试友好**: 便于问题追踪和复现
3. **扩展性**: 可添加新状态字段
4. **持久化**: 为后续会话记忆做准备

### 6. 关键技术要点

#### 6.1 思考链（Chain-of-Thought）
**概念**: 让AI逐步思考，展示推理过程

**应用场景**:
- 复杂问题分解
- 多步推理任务
- 需要解释的决策过程

#### 6.2 工具调用循环
**流程**:
1. LLM生成思考和行动
2. 解析工具调用
3. 执行工具获取结果
4. 结果反馈给LLM
5. 重复直到生成最终答案

#### 6.3 错误处理策略
**代码示例**:
```python
# rag_agent.py 第113行：AgentExecutor配置
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True  # 关键：处理解析错误
)

# enterprise_agent.py 第257-263行：容错解析
try:
    tasks = json.loads(response)  # 首选：标准JSON解析
except:
    # 备选：正则表达式提取
    match = re.search(r'\[.*\]', response, re.DOTALL)
    tasks = json.loads(match.group()) if match else []
```

### 7. 学习总结

#### 关键收获
1. **Agent是决策系统**: 感知→规划→执行→评估的完整循环
2. **架构选择重要**: 单一Agent简单直接，多Agent复杂但强大
3. **ReAct模式优势**: 可解释性强，适合需要推理的任务
4. **工程化考虑**: 错误处理、状态管理、可调试性

#### 实践建议
1. **从简单开始**: 先实现单一ReAct Agent
2. **逐步复杂化**: 添加更多工具和功能
3. **重视测试**: 为每个Agent编写单元测试
4. **监控调试**: 添加详细的日志和状态追踪

#### 进阶方向
1. **记忆机制**: 添加对话历史管理
2. **学习能力**: 基于反馈优化Agent行为
3. **多模态理解**: 支持图像、语音输入
4. **情感分析**: 识别用户情绪调整响应策略

---

**相关文件**:
- [rag_agent.py](e:\my_multi_agent\rag_agent.py) - 单一ReAct Agent实现
- [enterprise_agent.py](e:\my_multi_agent\enterprise_agent.py) - 多Agent协作系统
- [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) - 现代化工作流架构

**下一步学习**: LangChain工具系统 →