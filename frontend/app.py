"""The chat screen. It only talks to the FastAPI backend."""
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Legal Documents Q&A", page_icon="⚖️")
st.title("⚖️ Legal Documents Q&A")
st.caption(
    "Ask about the RTI Act, Consumer Protection Act, Motor Vehicles Act, "
    "DPDP Act and IT Act. Answers come only from these documents, with page citations."
)

with st.sidebar:
    st.header("Settings")
    model = st.text_input("LLM model (OpenRouter)", value="openrouter/free")
    st.caption("Paste any OpenRouter model id to switch models.")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
        st.success(f"Backend online ({health['retrieval_mode']} search)")
    except Exception:
        st.error(
            "Backend offline. Start it: uv run uvicorn backend.main:app --port 8000")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            pages = s.get("pages") or [s["page"]]
            label = f"page {pages[0]}" if len(
                pages) == 1 else f"pages {pages[0]}-{pages[-1]}"
            st.markdown(f"**[{i}] {s['document']}, {label}**")
            st.caption(s["text"])


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_sources(m.get("sources"))

question = st.chat_input("Ask a question about the documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            try:
                r = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": question, "model": model or None},
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()
                answer, sources = data["answer"], data["sources"]
            except Exception as e:
                answer, sources = f"Something went wrong: {e}", []
        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
