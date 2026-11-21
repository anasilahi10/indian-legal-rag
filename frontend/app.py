"""The chat screen. It only talks to the FastAPI backend."""
import time

import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"
EXAMPLES = [
    "Within how many days must a Public Information Officer reply to an RTI application?",
    "Is it mandatory to wear protective headgear while riding a motorcycle?",
    "What rights does a Data Principal have under the DPDP Act?",
    "What is the penalty for jaywalking in Delhi?",
]

st.set_page_config(page_title="Legal Documents Q&A",
                   page_icon="⚖️", layout="centered")
st.title("⚖️ Indian Legal Documents Q&A")
st.caption(
    "Answers come only from five laws: RTI Act, Consumer Protection Act, Motor Vehicles Act, "
    "DPDP Act and IT Act, with page citations. Not legal advice."
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

with st.sidebar:
    st.header("Settings")
    model = st.text_input(
        "LLM model id (optional)",
        value="",
        help="Leave empty to use the backend default. Paste a model id from your configured provider to switch.",
    )
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
        st.success(f"Backend online ({health['retrieval_mode']} search)")
    except Exception:
        st.error(
            "Backend offline. Start it: uv run uvicorn backend.main:app --port 8000")

    st.header("Try an example")
    for i, example in enumerate(EXAMPLES):
        if st.button(example, key=f"example_{i}", use_container_width=True):
            st.session_state.pending = example

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


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
            if m.get("meta"):
                st.caption(m["meta"])
            show_sources(m.get("sources"))

typed = st.chat_input("Ask a question about the documents...")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            started = time.time()
            try:
                r = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": question, "model": model.strip() or None},
                    timeout=180,
                )
                r.raise_for_status()
                data = r.json()
                answer, sources = data["answer"], data["sources"]
                meta = f"Model: {data['model']} · {time.time() - started:.1f}s"
            except Exception as e:
                answer, sources, meta = f"Something went wrong: {e}", [], ""
        st.markdown(answer)
        if meta:
            st.caption(meta)
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "meta": meta}
    )
