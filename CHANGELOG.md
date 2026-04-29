# CHANGELOG

## v1.2.0 (2026-04-29) — 大厂标准四阶段优化

### P0: 硬伤修复 + 安全加固 (`c949538`)
- **安全**: `.env` 移除真实 API Key，替换为占位符；`advanced_rag_system.py` pickle 序列化替换为 JSON
- **清理**: 删除 `multimodal_support.py` 等 605 行死代码导入；删除 `main.py`, `study.py` 等废弃文件
- **修复**: `frequent_topics` 的 `set()` 序列化报错（`dict(set())` → 遍历转换）
- **测试**: 新增 `tests/test_agent.py` — 26 个核心 Agent 测试（MemoryManager/Tool/EntryPoint/Workflow）
- **环境**: `.env.example` 精简为实际使用的变量

### P1: Repository 接口抽象 + 依赖注入 (`bdb4ad2`)
- **新增 `repositories/base.py`**: 4 个抽象基类 — `TicketRepository`, `BillingRepository`, `KnowledgeRepository`, `EscalationRepository`
- **新增 `repositories/memory_repo.py`**: 内存实现（6 工单、2 账户、10 知识条目、升级逻辑）
- **新增 `repositories/__init__.py`**: DI 工厂函数，通过 `REPO_BACKEND` 环境变量切换后端
- **重构工具函数**: `query_ticket_status`, `knowledge_search`, `escalate_to_human` 从硬编码字典改为 Repository 调用
- **新增 `tests/test_repositories.py`**: 15 个 Repository 测试

### P2: CI 质量门禁 (`8fd6f68`)
- **新增 `pyproject.toml`**: ruff lint + mypy 类型检查 + pytest 配置
- **更新 `.github/workflows/ci.yml`**: 三阶段流水线 — lint → type check → test (with coverage)
- **修复**: 全项目 ruff 30+ 错误清零
- **依赖清理**: `requirements.txt` 移除未使用的 `rich`, `mcp`, `python-json-logger`；补全 `pdfplumber`, `pypdf`

### P3: 流式输出 + LLM 结果整理 (`1edeb31`)
- **新增 `run_agent_stream()`**: 基于 `astream_events` 的异步生成器，逐步产出 `node_start` / `node_done` / `final_answer` 事件
- **增强 `postprocess_node`**: LLM 将工具调用结果整理为自然客服回复（而非直接返回原始检索文本）
- **更新 `app.py`**: 流式输出开关、渐进式展示 Agent 思考过程、同步模式自动回退
- **UI 改进**: 侧边栏整合清空对话 + 流式开关

---

## v1.1.0 — 功能扩展
- Skill 技能系统（11 个 yaml 定义的专业技能）
- MCP 工具系统（Client + Mock Server 架构）
- PDF 知识库切片入库（ChromaDB 62→171 条）
- 高级 RAG 系统（多策略检索、重排序、上下文压缩）

## v1.0.0 — 基础架构
- LangGraph ReAct Agent（preprocess → agent → postprocess 三节点）
- ChromaDB + DashScope 向量检索
- Streamlit Web UI
- 记忆管理（MemoryManager）
