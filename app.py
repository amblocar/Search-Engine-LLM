import streamlit as st

from langchain_groq import ChatGroq

from langchain.agents import create_agent

from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun,
)

from langchain_community.utilities import (
    WikipediaAPIWrapper,
    ArxivAPIWrapper,
)

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(page_title="LangChain Search Chat")

st.title("🔎 LangChain Search Assistant")

# ----------------------------------
# SIDEBAR
# ----------------------------------

st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)

# ----------------------------------
# TOOLS
# ----------------------------------

search_tool = DuckDuckGoSearchRun()

wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=500
    )
)

arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=500
    )
)

tools = [
    search_tool,
    wiki_tool,
    arxiv_tool,
]

# ----------------------------------
# SESSION STATE
# ----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm a chatbot that can search the web, Wikipedia, and Arxiv."
        }
    ]

# ----------------------------------
# DISPLAY CHAT HISTORY
# ----------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

# ----------------------------------
# USER INPUT
# ----------------------------------

user_question = st.chat_input(
    "Ask me anything..."
)

if user_question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.write(user_question)

    if not api_key:

        st.error("Please enter your Groq API Key")

    else:

        llm = ChatGroq(
            model="llama3-8b-8192",
            groq_api_key=api_key,
        )

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""
            You are a helpful research assistant.
            Use the available tools whenever needed.
            Answer clearly and concisely.
            """
        )

        with st.chat_message("assistant"):

            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_question
                        }
                    ]
                }
            )

            answer = response["messages"][-1].content

            st.write(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )
