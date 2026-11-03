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

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="LangChain Search Assistant",
    page_icon="🔎"
)

st.title("🔎 LangChain Search Assistant")

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)
selected_model = st.sidebar.selectbox(
    "Choose Model",
    [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "Gemma2-9b-It",
    ]
)

# ------------------------------------------------
# TOOLS
# ------------------------------------------------

search_tool = DuckDuckGoSearchRun()

wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=500,
    )
)

arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=500,
    )
)

tools = [
    search_tool,
    wiki_tool,
    arxiv_tool,
]

# ------------------------------------------------
# CHAT HISTORY
# ------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm a chatbot that can search "
                "the web, Wikipedia and Arxiv."
            ),
        }
    ]

# Display previous messages

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ------------------------------------------------
# USER INPUT
# ------------------------------------------------

user_prompt = st.chat_input(
    "Ask me anything..."
)

if user_prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    with st.chat_message("user"):
        st.write(user_prompt)

    if not api_key:
        st.error("Please enter Groq API Key")
        st.stop()

    try:

        # ----------------------------------------
        # LLM
        # ----------------------------------------

        llm = ChatGroq(
            groq_api_key=api_key,
            model=selected_model,
            temperature=0
        )

        # ----------------------------------------
        # AGENT
        # ----------------------------------------

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""
            You are a helpful AI research assistant.

            Use:
            - DuckDuckGo for web search
            - Wikipedia for encyclopedia knowledge
            - Arxiv for research papers

            Provide concise and accurate responses.
            """
        )

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                result = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ]
                    }
                )

                answer = result["messages"][-1].content

                st.write(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

    except Exception as e:

        st.error("Error occurred")

        st.exception(e)
