# 企业智能客服 Agent 系统

基于 LangGraph 和现代多 Agent 架构的智能客服系统，支持条件工作流、对话记忆和多工具调用。

## 🚀 核心特性

### 1. LangGraph 工作流架构
- **可视化工作流**：自动生成 Mermaid 图，支持条件分支和循环优化
- **状态管理**：使用 TypedDict 实现类型安全的状态管理
- **条件路由**：根据问题复杂度选择不同处理路径（简单/复杂/紧急问题）
- **循环优化**：答案不满意时自动重新处理，最多迭代 3 次

### 2. 三级记忆系统
- **短期记忆**：最近 10 轮对话历史
- **用户偏好记忆**：语言风格、信息详细程度、常用话题
- **对话摘要**：自动生成对话摘要，支持上下文感知
- **自适应响应**：根据用户偏好调整回答风格和详细程度

### 3. 扩展工具系统（6种工具）
- **知识库检索**：企业知识库问答
- **工单查询**：查询工单状态（模拟）
- **转人工**：紧急问题转人工客服
- **日期查询**：返回当前日期
- **天气查询**：查询城市天气信息（北京、上海等）
- **股票查询**：查询股票实时价格（AAPL、GOOGL、TSLA）

### 4. 生产就绪特性
- **错误处理**：多级降级策略（主工具→备用工具→缓存→兜底回答）
- **条件分支**：支持复杂逻辑路由和循环优化
- **可视化调试**：自动生成工作流图和记忆状态
- **会话持久化**：支持跨会话记忆保持

## 📁 项目结构

```
my_multi_agent/
├── langgraph_agent_simple.py          # LangGraph 简化版（核心架构演示）
├── langgraph_agent_with_memory.py     # LangGraph 带记忆版（推荐使用）
├── enterprise_agent.py                 # 原版多 Agent 系统（保留兼容）
├── app.py                             # Streamlit Web 界面
├── requirements.txt                   # 依赖包列表
├── workflow.png                       # 工作流可视化图
├── workflow.mmd                       # Mermaid 代码文件
├── optimization_phase1_complete.md    # 第一阶段优化报告
├── project_optimization_plan.txt      # 完整优化方案
└── agent_knowledge_review.txt         # Agent 知识复盘
```

## 🛠️ 快速开始

### 1. 环境配置
```bash
# 克隆项目
git clone <your-repo-url>
cd enterprise-agent

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 设置环境变量
echo "DASHSCOPE_API_KEY=your_api_key_here" > .env
```

### 2. 运行命令行版本
```bash
# 运行带记忆的 LangGraph Agent
python langgraph_agent_with_memory.py

# 运行简化版
python langgraph_agent_simple.py

# 运行原版 Agent
python enterprise_agent.py
```

### 3. 运行 Web 界面
```bash
streamlit run app.py
```
访问 http://localhost:8501

## 🔧 技术栈

- **Python 3.8+**
- **LangGraph 1.1.6**：工作流编排框架
- **LangChain 1.2.15**：AI 应用开发框架
- **DashScope**：阿里云百炼嵌入模型
- **ChromaDB**：向量数据库
- **Streamlit**：Web 界面框架
- **Pytest**：单元测试框架

## 📊 系统架构

### 工作流设计
```
用户输入 → 预处理（记忆更新） → 规划（任务拆解） → 执行（工具调用）
       ↖                                      ↙
       验证（答案质量评估） ← 后处理（记忆更新）
```

### 状态管理
```python
class AgentState(TypedDict):
    user_query: str                    # 用户查询
    messages: list                     # 对话历史
    user_preferences: Dict[str, Any]   # 用户偏好
    plan: List[str]                    # 任务规划
    tool_results: Dict[str, str]       # 工具结果
    final_answer: str                  # 最终答案
    # ... 其他元数据
```

### 条件路由逻辑
1. **简单问题**（如日期查询）→ 快速路径 → 直接回答
2. **复杂问题**（如技术问题）→ 详细分析 → 知识库检索
3. **紧急问题**（用户要求转人工）→ 人工升级路径
4. **答案不满意**（质量评估为 poor）→ 重新优化循环

## 🧠 记忆系统

### 记忆管理器 (MemoryManager)
```python
# 三级记忆
1. 短期记忆：最近10轮对话历史
2. 用户偏好：语言风格、详细程度、常用话题
3. 对话摘要：自动生成的对话总结

# 主要功能
- 上下文感知规划：考虑对话历史进行任务拆解
- 自适应响应：根据用户偏好调整回答风格
- 话题追踪：识别重复问题并提供上下文提示
- 记忆持久化：支持导出/导入对话历史
```

### 用户偏好学习
- **语言风格检测**：正式（您好/请）、中性、随意（哈喽/哈哈）
- **详细程度偏好**：根据回答长度学习用户偏好
- **话题频率统计**：记录用户常问话题

## 🛠️ 工具系统

### 工具工厂模式
```python
@tool
def weather_query(city: str) -> str:
    """查询城市天气信息"""
    weather_data = {
        "北京": "北京今天晴转多云，气温 15-25°C...",
        "上海": "上海今天多云，气温 18-28°C...",
        # ...
    }
    return weather_data.get(city, "信息暂不可用")
```

### 工具扩展
1. **企业工具**：知识检索、工单查询、转人工、日期查询
2. **API工具**：天气查询、股票查询（模拟真实API）
3. **可扩展架构**：支持动态添加新工具

## 🎯 使用示例

### 命令行交互
```bash
$ python langgraph_agent_with_memory.py

LangGraph 企业智能客服 Agent (带记忆版本) 已启动
命令说明:
  'quit' - 退出
  'visualize' - 生成工作流图
  'test' - 运行测试
  'history' - 查看对话历史
  'clear' - 清空记忆
  'export' - 导出记忆

用户: 如何重置密码？
助手: 您可以通过登录页面点击'忘记密码'链接重置密码...

用户: 再问一下密码问题  
助手: 您可以通过登录页面点击'忘记密码'链接...（注意：您之前也询问过密码相关问题）

用户: 北京天气怎么样？
助手: 北京今天晴转多云，气温 15-25°C，北风2-3级。
```

### Web 界面
1. 输入问题，Agent 自动规划、调用工具、验证答案
2. 展开"查看 Agent 思考过程"查看详细步骤
3. 查看工作流信息和记忆状态
4. 支持多轮对话，保持上下文记忆

## 📈 性能指标

### 优化成果
- **架构现代化**：从线性流水线升级为 LangGraph 工作流
- **功能丰富度**：工具从 4 种扩展到 6 种，增加记忆系统
- **用户体验**：支持个性化响应和多轮对话上下文
- **可观测性**：完整的工作流可视化和记忆状态监控

### 面试亮点
1. **技术深度**：LangGraph 架构设计、条件工作流、状态管理
2. **工程能力**：记忆系统设计、工具工厂模式、错误处理
3. **创新思维**：三级记忆系统、自适应响应、话题追踪
4. **业务理解**：企业客服场景优化、用户体验提升

## 🔮 后续优化计划

### 第二阶段：功能深度增强
1. **RAG 系统优化**：混合检索 + 查询扩展 + 重排序
2. **多模态支持**：图片分析、文档解析、语音处理
3. **异步优化**：并行执行、流式响应、性能提升
4. **高级监控**：LangSmith 集成、结构化日志、指标监控

### 第三阶段：生产化部署
1. **容器化**：Docker 镜像、Kubernetes 部署
2. **服务化**：REST API、微服务拆分、负载均衡
3. **安全合规**：输入过滤、输出审查、权限控制
4. **监控告警**：性能监控、错误追踪、自动告警

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系

- **作者**: gcddxd12
- **GitHub**: https://github.com/gcddxd12/enterprise-agent
- **技术栈**: Python, LangGraph, LangChain, Chroma, DashScope, Streamlit

---

## 🏆 项目亮点

### 技术亮点
1. **LangGraph 工作流**：可视化、条件分支、循环优化
2. **三级记忆系统**：短期记忆 + 用户偏好 + 长期摘要
3. **自适应响应**：根据用户偏好动态调整回答
4. **工具工厂模式**：支持动态扩展和热更新

### 工程亮点
1. **生产就绪**：错误处理、降级策略、超时控制
2. **可观测性**：工作流可视化、记忆状态监控
3. **可扩展性**：模块化设计、易于添加新功能
4. **文档完整**：架构文档、使用指南、优化方案

### 业务亮点
1. **企业级设计**：针对客服场景的优化设计
2. **用户体验**：个性化响应、多轮对话、快速响应
3. **成本效益**：模拟工具展示真实API集成能力
4. **合规安全**：记忆持久化、用户隐私保护

---

*项目创建时间：2025-02*
*最近优化时间：2026-04-09*
*优化内容：LangGraph 架构迁移、记忆系统增强、工具扩展*