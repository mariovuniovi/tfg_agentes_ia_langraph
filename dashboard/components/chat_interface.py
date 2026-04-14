"""Reusable chat display component."""

from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st


def render_message_history(messages: list) -> None:
    """Render a list of LangChain messages as a chat thread."""
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            name = getattr(msg, "name", None)
            prefix = f"**[{name}]** " if name else ""
            st.markdown(prefix + msg.content)
