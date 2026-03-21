import os
from datetime import date
from dotenv import load_dotenv
from langchain_core.tools import tool, Tool  # 关键修改：从 langchain_core.tools 导入
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatTongyi
from langchain.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

# 1. 加载向量数据库
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=api_key
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. 初始化 LLM（使用通义千问）
llm = ChatTongyi(model="qwen-plus", temperature=0)

# 3. 创建检索问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 4. 定义工具函数
def retrieve_and_answer(query: str) -> str:
    """从知识库检索并回答问题"""
    result = qa_chain.invoke({"query": query})
    return result["result"]

retrieve_tool = Tool(
    name="KnowledgeBaseSearch",
    func=retrieve_and_answer,
    description="从编程知识库中检索相关信息，适合回答 Python 语法、数据结构等问题。"
)

@tool
def get_current_date(query: str) -> str:
    """返回今天的日期"""
    return str(date.today())

tools = [retrieve_tool, get_current_date]

# 5. 创建 ReAct 风格的 Agent
prompt = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 6. 交互式对话
if __name__ == "__main__":
    print("Agent 已启动，输入问题（输入 'quit' 退出）")
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == 'quit':
            break
        response = agent_executor.invoke({"input": user_input})
        print(f"Agent: {response['output']}")