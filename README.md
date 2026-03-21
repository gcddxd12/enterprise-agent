# 企业智能客服 Agent

基于 RAG 和多 Agent 架构的企业智能客服系统，支持产品知识问答、工单查询、转人工等功能。

## 功能特点
- **多 Agent 协作**：规划 Agent 拆解任务，执行 Agent 调用工具，验证 Agent 检查答案。
- **RAG 检索**：使用阿里云百炼 `text-embedding-v4` 和 `qwen-plus`，从企业知识库中精准检索信息。
- **工具调用**：支持知识库检索、工单状态查询、日期查询、转人工。
- **可观测性**：前端展示 Agent 思考过程，包含任务列表和工具执行结果。
- **工程化**：单元测试、GitHub Actions CI、日志记录、输入安全过滤。

## 技术栈
- Python 3.10
- LangChain 0.2.1
- Chroma 向量数据库
- 阿里云百炼（text-embedding-v4, qwen-plus）
- Streamlit
- pytest + GitHub Actions

## 快速开始

### 环境准备
1. 克隆仓库
   ```bash
   git clone https://github.com/gcddxd12/enterprise-agent.git
   cd enterprise-agent
   创建虚拟环境并安装依赖

2.创建虚拟环境并安装依赖
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
配置 .env 文件

3.配置 .env 文件
DASHSCOPE_API_KEY=你的阿里云百炼API Key
构建向量库

4.构建向量库
python build_vector_store.py
5.运行应用
bash
streamlit run app.py
6.运行测试
bash
pytest tests/ -v 

项目结构
.
├── app.py                     # Streamlit 前端
├── enterprise_agent.py        # 多 Agent 核心逻辑
├── build_vector_store.py      # 向量库构建脚本
├── knowledge_base.txt         # 企业知识库源文件
├── chroma_db/                 # 向量库存储目录
├── tests/                     # 单元测试
├── requirements.txt
├── .env.example
├── .github/workflows/ci.yml   # GitHub Actions CI
└── README.md
