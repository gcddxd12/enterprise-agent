# LangChain与工具系统

## 概述
LangChain是一个用于开发大语言模型应用的框架，提供了一系列标准化的组件和接口。在本项目中，LangChain被用于构建Agent系统、管理工具调用、处理提示工程和集成向量数据库。

## 1. LangChain核心组件

### 1.1 LLM（大语言模型）集成

**文件**: [rag_agent.py](e:\my_multi_agent\rag_agent.py) 和 [enterprise_agent.py](e:\my_multi_agent\enterprise_agent.py)

**代码示例**:
```python
# rag_agent.py 第23-25行：通义千问LLM初始化
from langchain_community.chat_models import ChatTongyi

llm = ChatTongyi(model="qwen-plus", temperature=0)

# enterprise_agent.py 第94-101行：延迟初始化LLM
def get_llm():
    """LLM单例：全局共享，按需初始化"""
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
- `model`: 指定使用的模型（qwen-plus）
- `temperature`: 控制生成随机性（0表示确定性输出）
- `dashscope_api_key`: 阿里云API密钥

### 1.2 工具系统

#### 1.2.1 工具定义方式

**方式一：@tool装饰器**
```python
# enterprise_agent.py 第80-92行：@tool装饰器
from langchain_core.tools import tool

@tool
def knowledge_search(query: str) -> str:
    """从企业知识库中检索信息，返回答案。适用于产品使用、技术支持、销售政策等问题。"""
    qa_chain = get_qa_chain()
    if qa_chain is None:
        return "知识库服务不可用，请稍后再试。"
    result = qa_chain.invoke({"query": query})
    return result["result"]
```

**方式二：Tool类包装**
```python
# rag_agent.py 第40-44行：Tool类包装
from langchain_core.tools import Tool

retrieve_tool = Tool(
    name="KnowledgeBaseSearch",
    func=retrieve_and_answer,
    description="从编程知识库中检索相关信息，适合回答 Python 语法、数据结构等问题。"
)
```

**工具定义要素**:
1. **函数签名**: 明确的输入输出类型
2. **文档字符串**: 详细的工具描述和使用场景
3. **错误处理**: 优雅的降级策略
4. **依赖管理**: 延迟初始化资源

#### 1.2.2 工具分类体系

**检索类工具**:
```python
# enterprise_agent.py 第80-92行：知识库检索
@tool
def knowledge_search(query: str) -> str:
    """从企业知识库中检索信息..."""
    # 基于向量相似度的信息检索
```

**查询类工具**:
```python
# enterprise_agent.py 第104-117行：工单状态查询
@tool  
def query_ticket_status(ticket_id: str) -> str:
    """模拟查询工单状态"""
    mock_status = {
        "TK-123456": "您的工单 TK-123456 已受理...",
        "TK-789012": "工单 TK-789012 已处理完毕...",
        "default": "未找到工单信息..."
    }
    return mock_status.get(ticket_id, mock_status["default"])
```

**服务类工具**:
```python
# enterprise_agent.py 第119-126行：转人工服务
@tool
def escalate_to_human(query: str) -> str:
    """模拟转人工处理"""
    return "感谢您的耐心，我已将您的问题转接给人工客服..."
```

**计算类工具**:
```python
# enterprise_agent.py 第128-135行：日期查询
@tool
def get_current_date(query: str) -> str:
    """返回今天的日期"""
    from datetime import date
    return f"今天是 {date.today()}。"
```

### 1.3 提示工程

#### 1.3.1 提示模板

**文件**: [enterprise_agent.py](e:\my_multi_agent\enterprise_agent.py)

**代码示例**:
```python
# enterprise_agent.py 第219-241行：规划Agent提示模板
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
你是一个企业客服系统的任务规划专家。请将用户的问题拆解为一系列子任务。

可用的任务类型：
1. "knowledge_search: 具体查询内容" - 从知识库检索信息
2. "ticket_query: 工单号" - 查询工单状态  
3. "escalate" - 转人工客服
4. "date_query" - 查询当前日期

示例：
- 用户问"如何重置密码？" → ["knowledge_search: 重置密码"]
- 用户问"查询工单 TK-123456" → ["ticket_query: TK-123456"]

输出要求：
只输出 JSON 格式的任务列表，不要输出其他内容。

用户问题：{query}
任务列表（JSON）：
""")
```

**提示设计要点**:
1. **角色定义**: 明确Agent的专家身份
2. **任务规范**: 定义可用的任务类型和格式
3. **示例引导**: 提供少样本示例规范行为
4. **输出约束**: 强制JSON格式，便于程序解析

#### 1.3.2 链式调用

**代码示例**:
```python
# enterprise_agent.py 第243-246行：提示链构建
chain = prompt | get_llm()
response = chain.invoke({"query": query}).content
```

**链式操作符** (`|`):
- 将提示模板和LLM连接起来
- 支持复杂的处理流水线
- 代码简洁易读

### 1.4 记忆系统

#### 1.4.1 对话记忆

**文件**: [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py)

**代码示例**:
```python
# langgraph_agent_with_memory.py 第71-79行：状态定义（带记忆）
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Agent 工作流的状态定义（带记忆）"""
    user_query: str
    messages: Annotated[list, add_messages]  # 对话历史
    user_preferences: Dict[str, Any]  # 用户偏好
    # ... 其他状态字段
```

#### 1.4.2 记忆管理器

**代码示例**:
```python
# langgraph_agent_with_memory.py 第291-348行：记忆管理器
class MemoryManager:
    def __init__(self):
        self.short_term_memory = []  # 短期对话记忆
        self.user_preferences = {}   # 用户偏好记忆
        self.topic_tracker = {}      # 话题追踪
        
    def adapt_response(self, response: str) -> str:
        """根据用户偏好和对话历史调整响应"""
        # 基于语言风格偏好调整
        style = self.user_preferences.get("language_style", "professional")
        if style == "casual":
            response = self._make_casual(response)
        elif style == "detailed":
            response = self._add_details(response)
            
        # 基于历史对话避免重复
        response = self._avoid_repetition(response)
        
        return response
```

### 1.5 检索增强生成（RAG）集成

#### 1.5.1 检索问答链

**代码示例**:
```python
# rag_agent.py 第27-32行：RetrievalQA链创建
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 将所有文档"塞"进提示
    retriever=retriever,
    return_source_documents=True  # 返回源文档用于调试
)
```

#### 1.5.2 链类型选择

**`chain_type`选项**:
- `"stuff"`: 将所有相关文档拼接后一起传给LLM
- `"map_reduce"`: 分别处理每个文档后合并结果
- `"refine"`: 迭代优化答案
- `"map_rerank"`: 对每个文档评分后选择最佳

### 1.6 Agent执行器

#### 1.6.1 AgentExecutor

**代码示例**:
```python
# rag_agent.py 第108-114行：AgentExecutor配置
from langchain.agents import AgentExecutor, create_react_agent

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 显示详细执行过程
    handle_parsing_errors=True,  # 处理解析错误
    max_iterations=5  # 最大迭代次数
)
```

**关键配置参数**:
- `verbose`: 控制日志输出详细程度
- `handle_parsing_errors`: 是否自动处理解析错误
- `max_iterations`: 防止无限循环
- `early_stopping_method`: 提前停止策略

#### 1.6.2 自定义执行流程

**代码示例**:
```python
# enterprise_agent.py 第291-307行：自定义多Agent执行
def run_multi_agent(user_query: str) -> dict:
    """多Agent协作主流程"""
    tasks = planning_agent(user_query)        # 规划
    exec_results = execution_agent(tasks)     # 执行
    final = validation_agent(user_query, preliminary)  # 验证
    return {"tasks": tasks, "exec_results": exec_results, "final_answer": final}
```

### 1.7 工具设计最佳实践

#### 1.7.1 接口设计原则

**单一职责原则**:
```python
# 好的设计：每个工具只做一件事
@tool
def weather_query(city: str) -> str:
    """查询城市天气信息"""
    # 只负责天气查询

@tool  
def stock_query(symbol: str) -> str:
    """查询股票实时价格"""
    # 只负责股票查询
```

**明确契约**:
```python
# 明确的输入输出约定
@tool
def query_ticket_status(ticket_id: str) -> str:
    """模拟查询工单状态
    
    Args:
        ticket_id: 工单号，格式如 TK-123456
        
    Returns:
        工单状态描述字符串
    """
    # 实现...
```

#### 1.7.2 错误处理策略

**优雅降级**:
```python
@tool
def knowledge_search(query: str) -> str:
    """从企业知识库中检索信息"""
    qa_chain = get_qa_chain()
    if qa_chain is None:
        # 降级策略：返回友好提示而非崩溃
        return "知识库服务不可用，请稍后再试。"
    result = qa_chain.invoke({"query": query})
    return result["result"]
```

**超时控制**:
```python
# 虽然没有显式实现，但可以添加
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    # 实现超时控制上下文管理器
    pass

@tool
def slow_tool(query: str) -> str:
    """可能耗时的工具"""
    with timeout(30):  # 30秒超时
        return do_slow_operation(query)
```

#### 1.7.3 资源管理

**延迟初始化模式**:
```python
# enterprise_agent.py 第25-31行：全局单例资源
_embeddings = None
_vectorstore = None
_retriever = None
_llm = None
_qa_chain = None

def get_qa_chain():
    """QA链单例：按需初始化"""
    global _qa_chain
    if _qa_chain is None and get_retriever() is not None and get_llm() is not None:
        _qa_chain = RetrievalQA.from_chain_type(
            llm=get_llm(),
            chain_type="stuff",
            retriever=get_retriever(),
            return_source_documents=True
        )
    return _qa_chain
```

**设计优势**:
1. **启动速度快**: 按需加载资源
2. **内存友好**: 避免不必要的资源占用
3. **CI/CD兼容**: 测试环境可以不加载真实资源
4. **错误隔离**: 一个工具失败不影响其他工具

### 1.8 性能优化

#### 1.8.1 工具缓存

**代码示例**:
```python
# 工具结果缓存（概念示例）
import functools
from typing import Any, Callable

def cache_tool_results(ttl_seconds: int = 300):
    """工具结果缓存装饰器"""
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            cache_key = str((args, tuple(sorted(kwargs.items()))))
            
            if cache_key in cache:
                entry = cache[cache_key]
                if time.time() - entry["timestamp"] < ttl_seconds:
                    return entry["result"]
            
            result = func(*args, **kwargs)
            cache[cache_key] = {"result": result, "timestamp": time.time()}
            return result
        
        return wrapper
    
    return decorator

@tool
@cache_tool_results(ttl_seconds=60)  # 缓存60秒
def expensive_tool(query: str) -> str:
    """耗时的工具，结果可缓存"""
    # 实现...
```

#### 1.8.2 批量处理

**代码示例**:
```python
# 批量工具调用优化（概念示例）
def batch_knowledge_search(queries: List[str]) -> List[str]:
    """批量知识库检索，优化性能"""
    # 合并查询，减少API调用
    combined_query = " | ".join(queries)
    result = knowledge_search.run(combined_query)
    # 解析并分割结果
    return split_results(result)
```

### 1.9 测试与调试

#### 1.9.1 工具单元测试

**代码示例**:
```python
# 工具测试示例（概念代码）
import pytest
from unittest.mock import Mock

def test_knowledge_search():
    """测试知识库检索工具"""
    # 模拟依赖
    mock_qa_chain = Mock()
    mock_qa_chain.invoke.return_value = {"result": "测试答案"}
    
    # 替换全局变量进行测试
    original_get_qa_chain = get_qa_chain
    try:
        # 注入模拟对象
        globals()["get_qa_chain"] = lambda: mock_qa_chain
        
        # 执行测试
        result = knowledge_search.run("测试查询")
        assert "测试答案" in result
        mock_qa_chain.invoke.assert_called_once_with({"query": "测试查询"})
    finally:
        # 恢复原始函数
        globals()["get_qa_chain"] = original_get_qa_chain
```

#### 1.9.2 集成测试

**代码示例**:
```python
# 集成测试示例（概念代码）
def test_multi_agent_integration():
    """测试多Agent集成"""
    # 模拟用户查询
    user_query = "如何重置密码？"
    
    # 执行完整流程
    result = run_multi_agent(user_query)
    
    # 验证结果结构
    assert "tasks" in result
    assert "exec_results" in result
    assert "final_answer" in result
    
    # 验证任务规划
    assert len(result["tasks"]) > 0
    assert "knowledge_search" in result["tasks"][0]
    
    # 验证最终答案
    assert len(result["final_answer"]) > 0
```

### 1.10 学习总结

#### 关键知识点
1. **LangChain组件化**: 模块化的设计便于组合和替换
2. **工具系统标准化**: 统一的工具接口和调用规范
3. **提示工程系统化**: 模板化、示例化、约束化的提示设计
4. **资源管理精细化**: 延迟初始化、缓存、错误隔离等工程实践

#### 最佳实践
1. **工具设计**: 单一职责、明确契约、优雅降级
2. **性能优化**: 缓存策略、批量处理、异步调用
3. **测试策略**: 单元测试、集成测试、模拟依赖
4. **可维护性**: 配置驱动、模块化、文档完善

#### 常见陷阱
1. **工具过于复杂**: 违背单一职责原则
2. **错误处理不足**: 导致系统脆弱
3. **资源管理不当**: 内存泄漏或性能问题
4. **测试覆盖不足**: 难以保证质量

---

**相关文件**:
- [rag_agent.py](e:\my_multi_agent\rag_agent.py) - 标准LangChain Agent实现
- [enterprise_agent.py](e:\my_multi_agent\enterprise_agent.py) - 自定义工具和多Agent系统
- [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) - 记忆系统和高级集成

**下一步学习**: RAG技术 →