import streamlit as st
from langgraph_agent_with_memory import run_langgraph_agent_with_memory as run_langgraph_agent
import traceback

st.set_page_config(page_title="企业智能客服 Agent (调试版)", page_icon="💼")

st.title("💼 企业智能客服 Agent (调试版)")
st.markdown("输入您的问题，Agent 将自动规划、调用工具并验证答案。")
st.warning("调试模式已启用 - 显示详细错误信息")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_details" not in st.session_state:
    st.session_state.agent_details = {}
if "call_count" not in st.session_state:
    st.session_state.call_count = 0

# 显示历史消息
st.subheader("📜 对话历史")
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        st.caption(f"消息 #{i+1}")

# 显示调用计数
st.sidebar.info(f"Agent调用次数: {st.session_state.call_count}")

# 用户输入
if prompt := st.chat_input("请输入您的问题"):
    st.session_state.call_count += 1

    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 Agent - 带错误处理
    with st.chat_message("assistant"):
        with st.spinner("Agent 正在思考..."):
            try:
                st.info(f"调用Agent (第{st.session_state.call_count}次)...")
                result = run_langgraph_agent(prompt)
                st.success("Agent调用成功！")

                # 存储结果用于调试
                st.session_state.last_result = result

            except Exception as e:
                st.error(f"Agent调用失败: {e}")
                st.code(traceback.format_exc())
                result = {
                    "final_answer": f"抱歉，处理请求时出现错误: {e}",
                    "plan": ["错误"],
                    "tool_results": {},
                    "workflow_info": {"iterations": 0, "answer_quality": "error"},
                    "memory_info": {"conversation_length": 0, "user_preferences": {}}
                }

        # 展示 Agent 内部过程
        with st.expander("🔍 查看 Agent 思考过程"):
            st.subheader("📋 规划任务")
            st.write(result.get("plan", "无规划"))

            st.subheader("⚙️ 工具执行结果")
            if result.get("tool_results"):
                for task, res in result["tool_results"].items():
                    st.markdown(f"**{task}**")
                    st.code(res, language="text")
            else:
                st.write("无工具执行结果")

            st.subheader("📊 工作流信息")
            st.json(result.get("workflow_info", {}))

            st.subheader("🧠 记忆信息")
            st.json(result.get("memory_info", {}))

        # 展示最终答案
        final_answer = result.get("final_answer", "抱歉，无法生成答案")
        st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})

# 侧边栏：控制和调试
with st.sidebar:
    st.subheader("调试控制")

    if st.button("清空对话"):
        st.session_state.messages = []
        st.session_state.agent_details = {}
        st.session_state.call_count = 0
        st.success("对话已清空")
        st.rerun()

    if st.button("显示会话状态"):
        st.write("会话状态:")
        st.json({
            "message_count": len(st.session_state.messages),
            "call_count": st.session_state.call_count,
            "has_last_result": "last_result" in st.session_state
        })

    if st.button("测试Agent函数"):
        try:
            test_result = run_langgraph_agent("测试问题：今天日期是什么？")
            st.success("Agent测试成功！")
            st.json({
                "final_answer": test_result.get("final_answer", "")[:100],
                "plan": test_result.get("plan"),
                "workflow_iterations": test_result.get("workflow_info", {}).get("iterations")
            })
        except Exception as e:
            st.error(f"测试失败: {e}")
            st.code(traceback.format_exc())