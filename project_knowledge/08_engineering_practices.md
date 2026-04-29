# 工程化实践

## 概述
工程化实践是企业智能客服Agent项目从原型到生产就绪系统的关键环节。v2.0架构精简后，项目结构更加清晰，核心文件数量大幅减少。

## 1. 项目结构与组织

### 1.1 当前项目目录结构

```
my_multi_agent/
├── .github/workflows/              # GitHub Actions CI/CD配置
├── tests/                          # 单元测试和集成测试目录
├── project_knowledge/              # 项目知识文档（10个学习文档）
├── skills/                         # Skill技能定义文件（*.md）
├── mcp_servers/                    # MCP Server实现
│   ├── billing_server.py           #  Mock账单查询MCP Server
│   └── ticket_server.py            #  Mock工单系统MCP Server
├── logs/                           # 应用日志
├── chroma_db/                      # 向量数据库存储
├── multimodal_kb/                  # 多模态知识库存储
├── vector_cache/                   # 向量缓存
│
├── langgraph_agent_with_memory.py  # ★ 主Agent工作流（核心文件）
├── skill_manager.py                # Skill管理系统
├── mcp_client.py                   # MCP客户端管理器
├── advanced_rag_system.py          # 高级RAG系统（混合检索/查询扩展/重排序/缓存）
├── async_rag_system.py             # 异步RAG支持
├── async_executor.py               # 异步执行器（独立工具库）
├── monitoring_system.py            # 监控系统
├── monitoring_config.py            # 监控配置
├── multimodal_support.py           # 多模态支持（独立工具库）
├── build_vector_store.py           # 向量库构建脚本
│
├── app.py                          # Streamlit Web界面（主应用）
│
├── requirements.txt                # Python依赖包
├── mcp_servers.yaml                # MCP Server连接配置
├── .env.example                    # 环境变量模板
├── .env                            # 本地环境变量（不提交）
├── .gitignore                      # Git忽略配置
├── LICENSE                         # MIT许可证
├── README.md                       # 项目说明文档
└── knowledge_base.txt              # 企业知识库文本
```

### 1.2 v2.0清理说明

以下文件已在v2.0清理中删除（功能已合并或不再使用）：
- `rag_agent.py` — 基础RAG实现（功能已合并到langgraph_agent_with_memory.py）
- `enterprise_agent.py` — 旧版多Agent系统（被标准ReAct Agent替代）
- `langgraph_agent.py` — 基础LangGraph实现（已合并）
- `langgraph_agent_simple.py` — 简化版工作流（已合并）
- `app_simple.py` — 简化版Web界面（合并到app.py）
- `app_debug.py` — 调试版Web界面（已移除）
- `demo_optimized_features.py` — 优化功能演示（已移除）
- `demo_multimodal_features.py` — 多模态功能演示（已移除）
- `file_upload_processing` 等旧工具函数（已从AGENT_TOOLS移除）

### 1.3 模块职责划分

**核心模块**:
- `langgraph_agent_with_memory.py`: **唯一的主Agent文件**，包含AgentState、MemoryManager、5个本地工具、BASE_SYSTEM_PROMPT、agent_node（标准ReAct循环）、3节点工作流、命令行入口。集成Skill和MCP工具。
- `skill_manager.py`: Skill管理系统——扫描skills/*.md，构建倒排索引，关键词匹配，use_skill工具
- `mcp_client.py`: MCP客户端管理器——连接外部MCP Server，发现工具，包装为LangChain Tool
- `advanced_rag_system.py`: RAG优化功能（混合检索/查询扩展/重排序/向量缓存），独立模块
- `async_executor.py`: 异步执行框架，独立工具库，不直接参与主工作流
- `monitoring_system.py` + `monitoring_config.py`: 监控功能，独立模块
- `multimodal_support.py`: 多模态功能，独立工具库，未注册到主Agent

**应用入口**:
- `app.py`: Streamlit Web应用（生产级，双栏布局 + 调试面板含Skill/MCP状态）
- `langgraph_agent_with_memory.py`: 命令行交互入口（`__main__` 块）
- `main.py`: 简单入口脚本

## 2. 依赖管理与虚拟环境

### 2.1 依赖配置

**文件**: [requirements.txt](e:\my_multi_agent\requirements.txt)

```txt
# 核心依赖
langchain>=1.0.0
langchain-community>=0.4.0
langchain-core>=1.0.0
langgraph>=1.0.0
chromadb
dashscope
python-dotenv
pytest
streamlit

# 监控和评估依赖
langsmith>=0.1.0
structlog>=24.1.0
prometheus-client>=0.20.0
psutil>=5.9.0
python-json-logger>=3.0.0
rich>=13.7.0
watchdog>=3.0.0
```

**依赖分层**:
1. **核心框架**: LangChain、LangGraph、ChromaDB
2. **云服务**: DashScope（阿里云百炼）
3. **Web界面**: Streamlit
4. **监控工具**: LangSmith、structlog、prometheus-client
5. **开发工具**: pytest、rich

### 2.2 环境配置

```bash
# 创建虚拟环境
python -m venv .venv

# 激活（Windows）
.venv\Scripts\activate

# 激活（Linux/Mac）
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 3. 环境配置管理

**文件**: [.env.example](e:\my_multi_agent\.env.example)

```env
# 核心API密钥 (必需)
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# LangSmith配置 (可选)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=enterprise-agent-monitoring

# 日志配置
LOG_LEVEL=INFO
LOG_FORMAT=console
ENVIRONMENT=development
```

**配置加载顺序**: `.env` 文件 → 系统环境变量 → 代码默认值

## 4. 测试策略

### 4.1 测试文件

```
test_rag_optimization.py          # RAG检索功能测试
test_rag_integration.py           # RAG与工作流集成测试
test_async_integration.py         # 异步执行器测试
test_multimodal_integration.py    # 多模态功能测试
test_monitoring_integration.py    # 监控系统测试
test_monitoring_e2e.py            # 监控端到端测试
test_monitoring_import.py         # 监控模块导入测试
test_full_agent.py                # Agent完整流程测试
test_optimized.py                 # 优化功能测试
test_embedding.py                 # 嵌入模型测试
test_bm25.py                      # BM25检索测试
test_chroma_db.py                 # 向量数据库测试
test_date_query.py                # 日期工具测试
tests/test_enterprise_agent.py    # 主Agent测试（需更新为v2.0）
```

### 4.2 运行测试

```bash
pytest tests/ -v                    # 运行tests目录下所有测试
pytest test_rag_optimization.py -v  # 运行RAG测试
CI=true pytest                      # CI环境跳过API相关测试
```

## 5. CI/CD流水线

**文件**: [.github/workflows/ci.yml](e:\my_multi_agent\.github\workflows\ci.yml)

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.13'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    - name: Run tests
      env:
        DASHSCOPE_API_KEY: ${{ secrets.DASHSCOPE_API_KEY }}
      run: pytest tests/ -v
```

## 6. Web应用部署

### 6.1 Streamlit应用

**文件**: [app.py](e:\my_multi_agent\app.py)

```python
import streamlit as st
from langgraph_agent_with_memory import run_langgraph_agent_with_memory

st.title("中国移动智能客服")
user_query = st.text_input("请输入您的问题:")
if user_query:
    result = run_langgraph_agent_with_memory(user_query)
    st.subheader("智能回复")
    st.write(result['final_answer'])
    with st.expander("查看Agent思考过程"):
        st.json({k: v for k, v in result.items() if k != 'final_answer'})
```

**运行命令**:
```bash
python -m streamlit run app.py
# 或 streamlit run app.py --server.port=8501
```

### 6.2 命令行工具

```bash
python langgraph_agent_with_memory.py
# 交互命令: quit / visualize / test / history / clear / export
```

## 7. 安全最佳实践

### 7.1 API密钥管理

```python
# 环境变量（推荐）
api_key = os.getenv("DASHSCOPE_API_KEY")

# 生产环境使用密钥管理系统
# (AWS Secrets Manager / Azure Key Vault / 等)
```

### 7.2 .gitignore配置

关键条目：`.env`（密钥文件）、`chroma_db/`（向量库）、`logs/`（日志）、`__pycache__/`（字节码）

## 8. 新增功能开发流程

```bash
# 1. 创建功能分支
git checkout -b feature/new-tool

# 2. 开发（v2.0模式下扩展工具极简）
# 在 langgraph_agent_with_memory.py 中:
#   - 定义 @tool 函数
#   - 在 AGENT_TOOLS 列表中添加引用
#   - 在 SYSTEM_PROMPT 中添加工具说明

# 3. 测试
pytest tests/ -v

# 4. 提交
git add .
git commit -m "feat: add new tool"

# 5. 推送并创建PR
git push origin feature/new-tool
```

## 9. 版本管理

版本号遵循语义化版本（Semantic Versioning）：
- `v1.0.0`: 旧版多Agent流水线（6节点）
- `v2.0.0`: 标准ReAct Agent（3节点），代码量减半

## 10. 总结

### v2.0工程化改进
1. **文件精简**: 主Agent文件从63KB减至31KB，删除9个冗余源文件
2. **模块清晰**: 核心（langgraph_agent_with_memory.py）+ 3个独立工具库 + 1个Web入口
3. **扩展简单**: 新增工具只需修改1个文件（定义@tool + 加入列表）
4. **测试保留**: 全部测试文件保留，覆盖核心功能

### 生产就绪检查清单
- [x] 环境配置模板 (.env.example)
- [x] 依赖管理文件 (requirements.txt)
- [x] 单元测试覆盖 (pytest)
- [x] CI/CD流水线 (GitHub Actions)
- [x] 安全验证机制 (.gitignore + 环境变量)
- [x] 监控和日志系统
- [x] 项目知识文档 (8个学习文档)

---

**相关文件**:
- [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) — 核心Agent（唯一主文件）
- [app.py](e:\my_multi_agent\app.py) — Streamlit Web界面
- [requirements.txt](e:\my_multi_agent\requirements.txt) — 依赖管理
- [.env.example](e:\my_multi_agent\.env.example) — 环境配置模板
- [.github/workflows/ci.yml](e:\my_multi_agent\.github\workflows\ci.yml) — CI/CD流水线
- [README.md](e:\my_multi_agent\README.md) — 项目文档

**项目知识库完成**: ✅ 全部8个文档已更新至v2.0架构。
