"""Optional Streamlit UI: chat on the left, retrieved chunks + score in the sidebar.

Run:  streamlit run streamlit_app.py
"""
import streamlit as st

from src.graph import ask, build_rag_graph

st.set_page_config(page_title="Agentic AI RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Agentic AI eBook Chatbot")
st.caption("Answers strictly from the eBook. Out-of-scope questions are refused.")


@st.cache_resource(show_spinner="Connecting to Pinecone and Gemini...")
def load_graph():
    return build_rag_graph()


try:
    graph = load_graph()
except Exception as exc:  # missing keys / index not ingested yet
    st.error(f"Could not start the RAG graph: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last" not in st.session_state:
    st.session_state.last = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask something about Agentic AI..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask(graph, question)
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()
        st.markdown(result["answer"])
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.session_state.last = result

with st.sidebar:
    st.header("Retrieval details")
    last = st.session_state.last
    if last is None:
        st.info("Ask a question to see the retrieved context.")
    else:
        st.metric("Confidence score", f"{last['score']:.2f}")
        st.progress(min(max(float(last["score"]), 0.0), 1.0))
        st.caption(
            f"Top cosine similarity: {last['top_similarity']:.3f} · "
            f"{'grounded in document' if last['grounded'] else 'refused (not in document)'}"
        )
        for c in last["chunks"]:
            with st.expander(f"#{c['rank']} · page {c['page']} · sim {c['similarity']:.3f}"):
                st.write(c["text"])
