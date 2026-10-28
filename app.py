"""Thin Streamlit view: only validated answers and approved source links are shown."""

import asyncio
import os
from pathlib import Path

import streamlit as st

# Cloud provisioning preserves the same .env-only model configuration as local runs.
# APP_ENV is a private Streamlit secret containing the supplied dotenv text.
env_path = Path(__file__).resolve().parent / ".env"
if not env_path.exists() and st.secrets.get("APP_ENV", ""):
    env_path.write_text(st.secrets["APP_ENV"], encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(env_path, override=True, interpolate=False)

import agent
import knowledge


def show_result(result: agent.TurnResult) -> None:
    if result.reset:
        st.caption("Conversation memory was reset for this request.")
    if result.answer and result.answer.kind == "answer":
        references = list(dict.fromkeys(ref for block in result.answer.blocks for ref in block.evidence_ids))
        for block in result.answer.blocks:
            st.text(block.text)
            st.caption("Sources: " + ", ".join(str(references.index(ref) + 1) for ref in block.evidence_ids))
        with st.expander("Sources used"):
            for index, ref in enumerate(references, 1):
                source = result.evidence[ref]
                st.link_button(f"{index}. {source.title} — {source.heading}", source.canonical_url)
                st.caption(f"Snapshot: {source.fetched_at[:10]} · Reviewed: {source.reviewed_at[:10]}")
                st.text(source.text[:700])
    elif result.answer and result.answer.kind == "clarify":
        st.text(result.answer.clarification)
    else:
        st.text(result.message or agent.FALLBACKS["failed"])
        st.link_button("Sydney Water official website", "https://www.sydneywater.com.au/")
    if result.metrics:
        st.caption("Tracing configured" if result.metrics.get("tracing") == "configured" else "Tracing unavailable")
        with st.expander("How this answer was produced"):
            st.write({"actions": result.metrics.get("actions", []),
                      "model calls": result.metrics.get("model_calls", 0),
                      "MCP calls": result.metrics.get("mcp_calls", 0),
                      "seconds": result.metrics.get("duration_seconds"),
                      "citation checks": "passed" if result.answer and result.answer.kind == "answer" else "no factual answer displayed"})


def queue_question() -> None:
    question = st.session_state.question
    if agent.input_allowed(question):
        st.session_state.pending = question
        st.session_state.running = True
    else:
        st.session_state.input_error = True


st.set_page_config(page_title="Sydney Water information assistant", page_icon="💧")
st.title("Sydney Water information assistant")
st.caption("Independent student proof of concept — not an official Sydney Water service.")
st.write("Ask about public billing, payment and assistance information. Do not enter personal, account or payment details.")
st.caption("Check the linked official sources for conditions; this experimental demo can omit details.")
try:
    status = knowledge.readiness()
except Exception:
    st.error("The local source collection is missing or its review has expired. Rebuild and review it before starting a conversation.")
    st.stop()
st.caption(f"{status['documents']} reviewed sources · Content tracing: {'enabled (synthetic questions only)' if os.getenv('TRACE_CONTENT') == 'true' else 'off'}")
if "conversation" not in st.session_state:
    st.session_state.conversation = agent.Conversation()
    st.session_state.transcript = []
if st.button("New conversation", disabled=st.session_state.get("running", False)):
    st.session_state.conversation = agent.Conversation()
    st.session_state.transcript = []
    st.rerun()
st.caption("Try: What does the usage charge mean? · What ways can I pay? · My bill went up and I am struggling to pay it.")
for item in st.session_state.transcript:
    with st.chat_message(item["role"]):
        if item["role"] == "user":
            st.text(item["text"])
        else:
            show_result(item["result"])
st.chat_input("Ask a question without personal details", max_chars=2000, key="question",
              on_submit=queue_question, disabled=st.session_state.get("running", False))
if st.session_state.pop("input_error", False):
    st.warning(agent.FALLBACKS["input_blocked"])
if question := st.session_state.pop("pending", None):
    try:
        st.session_state.transcript.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.text(question)
        with st.chat_message("assistant"), st.spinner("Checking the reviewed sources…"):
            result = asyncio.run(agent.run_turn(question, st.session_state.conversation))
            show_result(result)
        st.session_state.transcript.append({"role": "assistant", "result": result})
        # UI transcript is separate from framework memory and bounded independently.
        st.session_state.transcript = st.session_state.transcript[-40:]
    finally:
        st.session_state.running = False
    st.rerun()
