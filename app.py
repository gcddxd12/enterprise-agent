import streamlit as st
from langgraph_agent_with_memory import run_langgraph_agent_with_memory as run_langgraph_agent

st.set_page_config(page_title="中国移动智能客服 Agent", page_icon="📶")

st.title("📶 中国移动智能客服 Agent")
st.markdown("输入您的问题，Agent 将自动规划、调用工具并验证答案。支持套餐、宽带、5G、物联网、云计算等业务咨询。")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        # assistant 消息用双栏布局
        with st.chat_message("assistant"):
            if "raw" in msg and msg["raw"]:
                col_ans, col_raw = st.columns(2)
                with col_ans:
                    st.markdown("**智能回复**")
                    st.markdown(msg["content"])
                with col_raw:
                    st.markdown("**知识检索详情**")
                    st.text_area("原始检索上下文", msg["raw"], height=300,
                                 key=f"raw_{hash(msg['content'])}",
                                 label_visibility="collapsed")
            else:
                st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题"):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 Agent
    with st.chat_message("assistant"):
        with st.spinner("Agent 正在思考..."):
            result = run_langgraph_agent(prompt)

        final_answer = result["final_answer"]
        raw_context = result.get("raw_context", "")

        # 双栏布局：左 = 智能回复，右 = 检索详情
        col_ans, col_raw = st.columns(2)
        with col_ans:
            st.markdown("**智能回复**")
            st.markdown(final_answer)
        with col_raw:
            st.markdown("**知识检索详情**")
            st.text_area(
                "原始检索上下文",
                raw_context if raw_context else "无检索结果",
                height=300,
                key=f"raw_{hash(final_answer)}",
                label_visibility="collapsed"
            )

        # 展开折叠查看完整思考过程
        with st.expander("🔍 查看 Agent 思考过程"):
            st.subheader("🎯 激活的专业技能")
            active_skills = result.get("active_skills", [])
            if active_skills:
                for skill_name in active_skills:
                    st.markdown(f"- {skill_name}")
            else:
                st.write("无（使用通用客服能力）")
            st.subheader("🔌 MCP 工具状态")
            mcp_status = result.get("mcp_status", {})
            if mcp_status:
                st.markdown(f"已连接 Server: {mcp_status.get('connected_servers', 0)}/{mcp_status.get('total_servers', 0)}")
                st.markdown(f"MCP 工具总数: {mcp_status.get('total_tools', 0)}")
                tools_by_srv = mcp_status.get("tools_by_server", {})
                if tools_by_srv:
                    for srv, tool_names in tools_by_srv.items():
                        st.markdown(f"- **{srv}**: {', '.join(tool_names)}")
            else:
                st.write("无 MCP 连接")
            st.subheader("📋 规划任务")
            st.write(result["plan"])
            st.subheader("⚙️ 工具执行结果")
            if result["tool_results"]:
                for task, res in result["tool_results"].items():
                    st.markdown(f"**{task}**")
                    st.code(res[:500], language="text")
            else:
                st.write("无工具执行结果")
            st.subheader("📊 工作流信息")
            st.json(result["workflow_info"])
            st.subheader("🧠 记忆信息")
            st.json(result["memory_info"])

    # 保存消息（带 raw context）
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "raw": raw_context
    })

# 侧边栏：清空对话
with st.sidebar:
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()
