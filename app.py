import streamlit as st
from langgraph_agent_with_memory import run_langgraph_agent_with_memory as run_langgraph_agent

st.set_page_config(page_title="企业智能客服 Agent", page_icon="💼")

st.title("💼 企业智能客服 Agent")
st.markdown("输入您的问题，Agent 将自动规划、调用工具并验证答案。")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_details" not in st.session_state:
    st.session_state.agent_details = {}

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
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

        # 展示 Agent 内部过程（用 expander 折叠）
        with st.expander("🔍 查看 Agent 思考过程"):
            st.subheader("📋 规划任务")
            st.write(result["plan"])
            st.subheader("⚙️ 工具执行结果")
            if result["tool_results"]:
                for task, res in result["tool_results"].items():
                    st.markdown(f"**{task}**")
                    st.code(res, language="text")
            else:
                st.write("无工具执行结果")

            st.subheader("📊 工作流信息")
            st.json(result["workflow_info"])

            st.subheader("🧠 记忆信息")
            st.json(result["memory_info"])

        # 展示最终答案
        final_answer = result["final_answer"]
        st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})

# 侧边栏：清空对话
with st.sidebar:
    if st.button("清空对话"):
        st.session_state.messages = []
        st.session_state.agent_details = {}
        st.rerun()