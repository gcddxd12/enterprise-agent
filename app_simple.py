"""
极简版企业智能客服 Agent Web界面
避免复杂依赖，确保基本功能正常工作
"""

import streamlit as st
import json

st.set_page_config(page_title="企业智能客服 (极简版)", page_icon="🤖")

st.title("🤖 企业智能客服 Agent (极简版)")
st.markdown("""
这是一个极简版本，确保基本对话功能正常工作。
如果完整版有问题，请使用此版本。
""")

# 尝试导入Agent函数
try:
    from langgraph_agent_with_memory import run_langgraph_agent_with_memory
    AGENT_AVAILABLE = True
    st.success("✅ Agent函数导入成功")
except ImportError as e:
    AGENT_AVAILABLE = False
    st.error(f"❌ Agent函数导入失败: {e}")
    st.info("将使用模拟回答模式")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.message_count = 0

# 显示对话历史
st.subheader("💬 对话历史")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 显示统计信息
st.sidebar.metric("消息总数", st.session_state.message_count)
st.sidebar.metric("Agent可用", "✅" if AGENT_AVAILABLE else "❌")

# 用户输入
user_input = st.chat_input("请输入您的问题")

if user_input:
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.message_count += 1

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)

    # 处理回复
    with st.chat_message("assistant"):
        if AGENT_AVAILABLE:
            with st.spinner("思考中..."):
                try:
                    # 调用Agent
                    result = run_langgraph_agent_with_memory(user_input)
                    response = result.get("final_answer", "抱歉，无法生成回答")

                    # 显示回答
                    st.markdown(response)

                    # 显示调试信息（可折叠）
                    with st.expander("查看处理详情"):
                        st.json({
                            "任务规划": result.get("plan"),
                            "工具结果": result.get("tool_results"),
                            "工作流信息": result.get("workflow_info"),
                            "记忆信息": result.get("memory_info")
                        })

                except Exception as e:
                    st.error(f"处理失败: {e}")
                    response = f"处理请求时出现错误: {str(e)}"
                    st.markdown(response)
        else:
            # 模拟回答
            response = f"收到您的查询: '{user_input}'\n\n(模拟回答: Agent功能当前不可用)"
            st.markdown(response)
            st.warning("当前为模拟模式，Agent功能不可用")

    # 添加助手消息
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.message_count += 1

# 控制面板
with st.sidebar:
    st.subheader("控制面板")

    if st.button("清空对话"):
        st.session_state.messages = []
        st.session_state.message_count = 0
        st.success("对话已清空")
        st.rerun()

    if st.button("测试连接"):
        if AGENT_AVAILABLE:
            try:
                test_result = run_langgraph_agent_with_memory("测试: 今天日期")
                st.success(f"测试成功! 回答: {test_result.get('final_answer', '')[:50]}...")
            except Exception as e:
                st.error(f"测试失败: {e}")
        else:
            st.warning("Agent不可用，无法测试")

    if st.button("显示状态"):
        st.json({
            "消息数量": len(st.session_state.messages),
            "总消息数": st.session_state.message_count,
            "Agent可用": AGENT_AVAILABLE
        })

# 页脚
st.markdown("---")
st.caption("极简版 Web界面 | 设计用于基本功能测试和故障排除")