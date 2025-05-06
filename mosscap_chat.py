"""Conversations with Ollama LLM via Streamlit.

This script uses streamlit, langchain, and Ollama to provide a chat with a locally 
hosted LLM.

The *Mosscap* reference is from the name of the robot in Becky Chamber's
*Monk and Robot* books.

"""

import argparse
import streamlit as st
from langchain_ollama import OllamaLLM as Ollama

# Old way
#from langchain.chains import ConversationChain
#from langchain.memory import ConversationBufferMemory

# New way
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# A simple store for single-session history
_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Lazy return message history.
    
    Lazily create or return an *InMemoryChatMessageHistory* fro the given session_id

    """

    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()

    return _store[session_id]


def parse_args():
    """Simple argument parsing to choose conversation."""

    descr_str = (
        "Uses Ollama LLM and LangChain to chat with local LLM."
    )
    parser = argparse.ArgumentParser(
        prog="Local LLM Conversation", description=descr_str, fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--llm_model",
        action="store",
        type=str,
        default="gemma3:12b-it-qat",
        help=(
            "Name of Ollama LLM to converse with."
        ),
    )

    # Only parse known args so streamlit arguments prior to -- are not interpreted
    # as an error
    args, _ = parser.parse_known_args()

    return args


def run_app(model_name: str):
    """Define chat actions."""

    # 1) Build the LLM + memory + chain
    @st.cache_resource
    def get_conversation_chain():
        # Disable LangChain callback tracing.
        llm = Ollama(
            model=model_name,
            temperature=0.7,
            callbacks=[],
            verbose=False,
            #max_tokens=512,
            #n_ctx=4096,
            )

        return RunnableWithMessageHistory(
            llm,
            get_session_history,
            #input_messages_key="input",
            #history_messages_key="history"
        )
    
    conversation = get_conversation_chain()

    # 2) Streamlit UI
    st.title(f"Chat with Mosscap:\n [{model_name}]")

    if "history" not in st.session_state:
        st.session_state.history = []

    # Render full chat
    for user_msg, bot_msg in st.session_state.history:
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**Mosscap:** {bot_msg}")
        st.write("---")

    # input + handler
    def send():
        prompt = st.session_state.prompt.strip()
        if not prompt:
            return
        with st.spinner("Thinking..."):
            # Each predict call prepends entire history
            response = conversation.invoke(
                prompt,
                config={"configurable": {"session_id": "default"}}
            )

        st.session_state.history.append((prompt, response))
        st.session_state.prompt = ""

    st.text_input(
        "Input:",
        key="prompt",
        on_change=send,
        placeholder="What do humans need?"
    )


if __name__ == "__main__":
    args = parse_args()
    run_app(args.llm_model)