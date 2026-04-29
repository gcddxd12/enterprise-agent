import asyncio
import streamlit as st
from langgraph_agent_with_memory import run_langgraph_agent_with_memory as run_langgraph_agent
from langgraph_agent_with_memory import run_agent_stream

st.set_page_config(page_title="中国移动智能客服 Agent", page_icon="📶")

# 稳定的 widget key 计数器
if "_key_counter" not in st.session_state:
    st.session_state._key_counter = 0


def _next_key() -> str:
    st.session_state._key_counter += 1
    return f"widget_{st.session_state._key_counter}"


st.title("📶 中国移动智能客服 Agent")
st.markdown("输入您的问题，Agent 将自动规划、调用工具并验证答案。支持套餐、宽带、5G、物联网、云计算等业务咨询。")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏
with st.sidebar:
    stream_mode = st.toggle("流式输出", value=True, help="实时展示 Agent 思考过程")
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()

# 显示历史消息
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            if "raw" in msg and msg["raw"]:
                col_ans, col_raw = st.columns(2)
                with col_ans:
                    st.markdown("**智能回复**")
                    st.markdown(msg["content"])
                with col_raw:
                    st.markdown("**知识检索详情**")
                    st.text_area("原始检索上下文", msg["raw"], height=300,
                                 key=f"raw_{msg['raw']}",
                                 label_visibility="collapsed")
            else:
                st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if stream_mode:
            # 流式模式：逐步展示 Agent 思考过程
            status_placeholder = st.empty()
            progress_container = st.container()
            answer_placeholder = st.empty()

            # 累计收集结果
            tool_calls_collected = []
            tool_results_collected = {}
            final_answer = ""
            raw_context = ""

            async def _stream():
                nonlocal final_answer, raw_context
                async for event in run_agent_stream(prompt):
                    if event["type"] == "node_start":
                        status_placeholder.info(f"⏳ {event['message']}")
                    elif event["type"] == "node_done":
                        if "tool_calls" in event:
                            tool_calls_collected.extend(event["tool_calls"])
                        if "tool_results" in event:
                            tool_results_collected.update(event["tool_results"])
                    elif event["type"] == "final_answer":
                        final_answer = event["answer"]
                        raw_context = event.get("result", {}).get("raw_context", "")  # noqa: F841 (nonlocal)
                        answer_placeholder.markdown(final_answer)
                        status_placeholder.success("完成")
                    elif event["type"] == "error":
                        status_placeholder.error(event["message"])
                        final_answer = f"错误: {event['message']}"

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_stream())
                loop.close()
            except Exception as e:
                final_answer = f"处理失败: {e}"
                status_placeholder.error(final_answer)

            # 显示最终答案（流式失败时回退到同步模式）
            if not final_answer:
                result = run_langgraph_agent(prompt)
                final_answer = result["final_answer"]
                raw_context = result.get("raw_context", "")
                answer_placeholder.markdown(final_answer)
                status_placeholder.success("完成（同步模式）")

            # 展示思考过程
            with st.expander("🔍 查看 Agent 思考过程"):
                if tool_calls_collected:
                    st.subheader("📋 执行的任务")
                    for call in tool_calls_collected:
                        st.markdown(f"- {call}")
                if tool_results_collected:
                    st.subheader("⚙️ 工具结果")
                    for task, res in tool_results_collected.items():
                        st.markdown(f"**{task}**")
                        st.code(str(res)[:300], language="text")
        else:
            # 同步模式
            with st.spinner("Agent 正在思考..."):
                result = run_langgraph_agent(prompt)

            final_answer = result["final_answer"]
            raw_context = result.get("raw_context", "")

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
                    key=_next_key(),
                    label_visibility="collapsed"
                )

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
                st.subheader("📋 规划任务")
                st.write(result["plan"])
                st.subheader("⚙️ 工具执行结果")
                if result["tool_results"]:
                    for task, res in result["tool_results"].items():
                        st.markdown(f"**{task}**")
                        st.code(res[:500], language="text")
                st.subheader("📊 工作流信息")
                st.json(result["workflow_info"])
                st.subheader("🧠 记忆信息")
                st.json(result["memory_info"])

    # 保存消息
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "raw": raw_context
    })
