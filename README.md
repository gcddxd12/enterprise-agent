# 中国移动智能客服 Agent 系统

基于 LangGraph ReAct 架构的智能客服系统，集成 RAG 知识库检索、MCP 外部工具协议、Skill 技能系统。

## 核心特性

### 1. LangGraph ReAct Agent
- 3 节点工作流（preprocess → agent → postprocess）
- `llm.bind_tools()` 原生工具调用
- 条件路由和状态管理（TypedDict）

### 2. RAG 知识库检索
- ChromaDB 向量数据库（171 条中国移动知识条目）
- DashScope text-embedding-v4 嵌入模型
- PDF + txt 混合数据源，增量入库

### 3. MCP 工具系统
- MCPClientManager 管理外部 MCP Server 连接
- JSON-RPC 2.0 over stdio 协议
- 动态工具发现 → LangChain StructuredTool 包装
- Mock 账单查询 + 工单系统 Server 演示

### 4. Skill 技能系统
- YAML frontmatter + Markdown 技能定义文件
- 关键词触发 + LLM 工具双重匹配
- 4 个预设技能（5G业务、投诉处理、网络排障、套餐推荐）

### 5. 本地工具
- `knowledge_search` — 向量知识库检索
- `query_ticket_status` — 工单状态查询
- `escalate_to_human` — 转人工客服
- `get_current_date` — 日期查询
- `use_skill` — 技能调用

## 项目结构

```
my_multi_agent/
├── langgraph_agent_with_memory.py    # 核心 Agent（ReAct + 记忆 + 工具分发）
├── advanced_rag_system.py            # 高级 RAG 检索器
├── build_vector_store.py             # 向量库构建脚本（PDF + txt）
├── mcp_client.py                     # MCP Client Manager
├── mcp_servers.yaml                  # MCP Server 配置
├── skill_manager.py                  # Skill 管理器
├── app.py                            # Streamlit Web 界面
├── requirements.txt                  # 依赖列表
├── conftest.py                       # Pytest 配置
├── mcp_servers/                      # Mock MCP Server
│   ├── billing_server.py             #   账单查询服务
│   └── ticket_server.py              #   工单系统服务
├── skills/                           # 技能定义文件
│   ├── 5g_service.md
│   ├── complaint_handling.md
│   ├── network_troubleshooting.md
│   └── package_recommendation.md
├── tests/                            # 测试
│   └── test_core.py                  #   核心功能测试（12 个）
├── project_knowledge/                # 知识文档
├── data/                             # 知识源数据
├── knowledge_data/                   # 已分类知识数据
└── chroma_db/                        # ChromaDB 向量库（持久化）
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量（复制模板并填入真实 Key）
cp .env.example .env

# 构建向量库（首次运行）
python build_vector_store.py

# 启动命令行 Agent
python langgraph_agent_with_memory.py

# 启动 Web 界面
streamlit run app.py
```

## 技术栈

- **Python 3.10+**
- **LangGraph** — Agent 工作流编排
- **LangChain** — LLM 应用框架
- **DashScope** — 阿里云百炼嵌入模型（text-embedding-v4）
- **Tongyi** — 阿里云通义千问 LLM
- **ChromaDB** — 向量数据库
- **Streamlit** — Web 界面
- **MCP** — Model Context Protocol 工具集成

## 系统架构

```
用户输入 → preprocess（Skill 匹配 + 记忆更新）
       → agent_node（ReAct: LLM 推理 → 工具调用 → 结果反馈）
       → postprocess（答案整理 + 记忆持久化）
       → 最终答案

工具层:
  ├── 本地工具: knowledge_search, query_ticket_status, escalate_to_human, get_current_date, use_skill
  └── MCP 工具: billing_query_*, ticket_query_* (通过 MCPClientManager 动态发现)
```

## 测试

```bash
pytest tests/ -v
```

## License

MIT
