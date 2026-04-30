"""Chat page — conversational interface to the MLOps agent pipeline."""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(page_title="Chat", layout="wide")
st.title("Chat with MLOps Agents")

# Initialize session state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_thread_id" not in st.session_state:
    import uuid
    st.session_state.chat_thread_id = str(uuid.uuid4())

# Display message history
for msg in st.session_state.chat_messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input
if prompt := st.chat_input("Ask the MLOps pipeline anything..."):
    st.session_state.chat_messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agents thinking..."):
            try:
                from mlops_agents.graphs.mlops_graph import graph
                from mlops_agents.config.constants import GRAPH_RECURSION_LIMIT

                config = {
                    "configurable": {"thread_id": st.session_state.chat_thread_id},
                    "recursion_limit": GRAPH_RECURSION_LIMIT,
                }
                state = {
                    "messages": list(st.session_state.chat_messages),
                    "next": "",
                    "dataset_path": "",
                    "validation_passed": False,
                    "validation_report": {},
                    "trained_model_path": "",
                    "training_run_id": "",
                    "training_metrics": {},
                    "evaluation_passed": False,
                    "evaluation_report": {},
                    "best_model_uri": "",
                    "deployment_decision": "pending",
                    "deployment_status": "",
                    "error_message": "",
                    "agent_attempt_counts": {},
                }

                result = graph.invoke(state, config=config)
                response = result["messages"][-1].content
                st.markdown(response)
                st.session_state.chat_messages.append(AIMessage(content=response))

            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
