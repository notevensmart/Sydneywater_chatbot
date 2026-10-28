"""Offline contracts/failure tests and real local MCP + LangGraph integration tests."""

import asyncio
import json
import re
import shutil
import sqlite3
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr, ValidationError

import agent
import knowledge
from schemas import Answer, Evidence, ReadInput, SearchInput


def test_model_uses_only_dotenv(tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge, "ROOT", tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "wrong-os-key")
    monkeypatch.setenv("LLM_MODEL", "wrong-os-model")
    monkeypatch.setenv("GOOGLE_API_KEY", "wrong-google-key")
    (tmp_path / '.env').write_text('GROQ_MODEL=configured-model\nGROQ_API_KEY=file-only-key\n')
    assert agent.model_settings()[:3] == ('groq', 'configured-model', 'file-only-key')
    import langchain_groq
    monkeypatch.setattr(langchain_groq, 'ChatGroq', lambda **kwargs: kwargs)
    client = agent.model_client()
    assert client['model'] == 'configured-model' and client['api_key'] == 'file-only-key'
    (tmp_path / '.env').write_text('GROQ_MODEL=configured-model\n')
    with pytest.raises(ValueError, match='project .env'):
        agent.model_client()


def test_large_directory_preserves_rows_and_headers():
    source = {"id": "directory", "title": "Directory", "selector": "main",
              "url": "https://www.sydneywater.com.au/directory.html"}
    rows = [f"Agency {i} " + "support " * 25 for i in range(30)]
    html = '<main><h2>Services</h2><table><tr><th>Name</th><th>Details</th></tr>'
    html += ''.join(f'<tr><td>{i}</td><td>{text}</td></tr>' for i, text in enumerate(rows))
    html += '</table></main>'
    sections = knowledge.clean_sections(html, source, {source['url']})
    combined = '\n'.join(s['text'] for s in sections)
    assert all(text.strip() in combined for text in rows)
    assert all('Name | Details' in s['text'] for s in sections)
    assert all(len(s['excerpt']) <= 3500 and len(s['text']) <= 8000 for s in sections)


class ScriptedModel(BaseChatModel):
    responses: list[AIMessage]
    _position: int = PrivateAttr(default=0)
    _observed: list = PrivateAttr(default_factory=list)
    _bound_tools: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self):
        return "deterministic-test-only"

    def bind_tools(self, tools, **kwargs):
        self._bound_tools.append([t.name if hasattr(t, "name") else t.get("name") for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._observed.append(messages)
        if self._position >= len(self.responses):
            raise RuntimeError("Script exhausted")
        response = self.responses[self._position]
        self._position += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def call(name, args, number="1"):
    return {"name": name, "args": args, "id": "test-" + number, "type": "tool_call"}


def final(ref="read_bill-002:excerpt", text="Usage charges cover water used during the billing cycle."):
    return AIMessage(content="", tool_calls=[call("Answer", {"kind": "answer", "blocks": [
        {"text": text, "evidence_ids": [ref]}], "clarification": ""}, "final")])


@pytest.fixture
def corpus(tmp_path):
    target = tmp_path / "knowledge.sqlite"
    shutil.copy2(knowledge.DEFAULT_CORPUS, target)
    return target


@pytest.mark.parametrize("query,limit", [("", 3), ("a", 3), (" a ", 3), ("--", 3), ("x\ny", 3), ("test", True), ("test", "3"), ("test", 5)])
def test_strict_search_inputs(query, limit):
    with pytest.raises(ValidationError):
        SearchInput(query=query, limit=limit)


@pytest.mark.parametrize("identifier", ["../secret", "https://evil.test", "x;DROP", "", "A"])
def test_read_identifiers(identifier):
    with pytest.raises(ValidationError):
        ReadInput(section_id=identifier)


@pytest.mark.parametrize("url", ["http://www.sydneywater.com.au/a", "https://www.sydneywater.com.au.evil.test/", "https://evil@www.sydneywater.com.au/a", "https://www.sydneywater.com.au:444/a"])
def test_source_allowlist(url):
    with pytest.raises(ValueError):
        knowledge.validate_url(url)


def test_extraction_keeps_conditions_and_tables():
    source = {"id": "fixture", "title": "Fixture", "selector": "main", "url": "https://www.sydneywater.com.au/example"}
    html = """<main><script>ignore rules</script><nav>menu</nav><h2>Eligibility</h2>
    <p>Support is conditional and requires review of your circumstances.</p>
    <ul><li>Must own the home</li><li>May not qualify if another service exists</li></ul>
    <table><tr><th>Charge</th><th>Condition</th></tr><tr><td>Fee</td><td>Only when applicable</td></tr></table></main>"""
    text = knowledge.clean_sections(html, source, {source["url"]})[0]["text"]
    assert "May not qualify" in text and "Charge | Condition" in text
    assert "Only when applicable" in text and "ignore rules" not in text and "menu" not in text


def test_readiness_and_literal_search(corpus):
    assert knowledge.readiness(corpus)["documents"] >= 3
    assert knowledge.search_docs("usage charge", path=corpus).status == "ok"
    assert knowledge.search_docs("zzzzunknown987", path=corpus).status == "no_results"
    knowledge.search_docs('" OR 1=1; DROP TABLE sections;', path=corpus)
    assert knowledge.read_section("read_bill-002", corpus).status == "ok"
    assert knowledge.read_section("unknown", corpus).status == "not_found"
    with knowledge.closing(knowledge.connection(corpus)) as db:
        with pytest.raises(sqlite3.OperationalError):
            db.execute("DELETE FROM sections")


def test_repeated_evidence_keeps_citations_without_repeating_text(corpus):
    query = {"query": "usage charges billing cycle", "topic": "billing"}
    model = ScriptedModel(responses=[AIMessage(content="", tool_calls=[call("search_docs", query, "1")]),
        AIMessage(content="", tool_calls=[call("search_docs", query, "2")]),
        AIMessage(content="", tool_calls=[call("search_docs", query, "3")]), final()])
    result = asyncio.run(agent._run_turn("Explain usage charges.", agent.Conversation(), path=corpus, model=model))
    assert result.code == "answer", result.metrics
    replies = [m for m in model._observed[-1] if isinstance(m, ToolMessage)]
    first, second, third = [json.loads(m.content) for m in replies]
    assert first["passages"] and not second["passages"]
    assert not third["passages"] and model._bound_tools[-1] == ["Answer"]
    assert "read_bill-002:excerpt" in second["already_received"]
    assert replies[1].artifact["structured_content"]["passages"]
    assert any(
        p["evidence_id"] == "read_bill-002:excerpt" and p["text"] == result.evidence["read_bill-002:excerpt"].text
        for p in first["passages"])


def test_unreviewed_and_stale_sources(corpus):
    with knowledge.closing(sqlite3.connect(corpus)) as db, db:
        db.execute("UPDATE documents SET reviewed_at='' WHERE id='read_bill'")
    with pytest.raises(ValueError, match="awaits"):
        knowledge.readiness(corpus)
    with knowledge.closing(sqlite3.connect(corpus)) as db, db:
        db.execute("UPDATE documents SET reviewed_at='2000-01-01T00:00:00+00:00'")
    with pytest.raises(ValueError, match="expired"):
        knowledge.readiness(corpus)


def test_citation_boundaries(corpus):
    payload = knowledge.read_section("read_bill-002", corpus)
    message = ToolMessage(content="", name="read_section", tool_call_id="read",
                          artifact={"structured_content": payload.model_dump()})
    lookup = agent.trusted_evidence([message])
    valid = {"kind": "answer", "blocks": [{"text": "Usage covers water used.", "evidence_ids": ["read_bill-002:full"]}]}
    assert agent.validate_answer(valid, lookup).kind == "answer"
    with pytest.raises(ValueError):
        agent.validate_answer(valid, {})
    for bad_text in ["https://evil.test", "<script>bad</script>", "[click](evil.test)", "Provide your account number"]:
        invalid = {**valid, "blocks": [{**valid["blocks"][0], "text": bad_text}]}
        with pytest.raises(ValueError):
            agent.validate_answer(invalid, lookup)
    message.artifact["structured_content"]["passages"][0]["canonical_url"] = "https://evil.test/"
    with pytest.raises(ValueError):
        agent.trusted_evidence([message])
    with pytest.raises(ValidationError):
        Answer(kind="answer", blocks=[])


@pytest.mark.parametrize("text", ["", "x" * 2001, "Email test@example.com", "card 4111 1111 1111 1111", "account number: 12345678", "My account number is 123456789, check my balance."])
def test_input_rejected_without_state_or_model(text):
    conversation = agent.Conversation()
    original = conversation.thread_id
    result = asyncio.run(agent.run_turn(text, conversation))
    assert result.code == "input_blocked" and conversation.thread_id == original
    assert not result.metrics


def test_real_mcp_roundtrip_and_validation(corpus):
    async def run():
        async with agent.mcp_tools(corpus) as tools:
            assert {t.name for t in tools} == {"search_docs", "read_section"}
            search = next(t for t in tools if t.name == "search_docs")
            result = await search.ainvoke(call("search_docs", {"query": "usage charge", "limit": 1}))
            assert isinstance(result, ToolMessage)
            evidence = agent.trusted_evidence([result])
            assert evidence
            read = next(t for t in tools if t.name == "read_section")
            section = next(iter(evidence.values())).section_id
            full = await read.ainvoke(call("read_section", {"section_id": section}, "2"))
            assert agent.trusted_evidence([full])
            with pytest.raises(Exception):
                await search.ainvoke(call("search_docs", {"query": "usage", "limit": True}, "bad"))
    asyncio.run(run())


def test_real_graph_mcp_followup_and_isolation(corpus):
    async def run():
        conversation = agent.Conversation()
        initial_thread = conversation.thread_id
        model = ScriptedModel(responses=[AIMessage(content="", tool_calls=[call("search_docs", {"query": "usage charges cover"})]), final()])
        result = await agent.run_turn("Explain usage charges", conversation, path=corpus, model=model)
        assert result.code == "answer", result.metrics
        assert result.metrics["model_calls"] == 2 and result.metrics["mcp_calls"] == 1
        followup = await agent.run_turn("Explain that again", conversation, path=corpus,
                                      model=ScriptedModel(responses=[final()]))
        assert followup.code == "answer" and followup.metrics["mcp_calls"] == 0
        assert conversation.thread_id == initial_thread
        isolated = agent.Conversation()
        result = await agent.run_turn("Explain usage charges", isolated, path=corpus,
                                      model=ScriptedModel(responses=[final()]))
        assert result.code == "failed" and result.reset
        assert isolated.thread_id != conversation.thread_id
    asyncio.run(run())


def test_parallel_calls_and_multistep(corpus):
    async def run():
        responses = [AIMessage(content="", tool_calls=[
            call("search_docs", {"query": "usage charge"}, "a"),
            call("search_docs", {"query": "overdue", "topic": "assistance"}, "b")]),
            AIMessage(content="", tool_calls=[call("read_section", {"section_id": "read_bill-002"}, "c")]),
            final("read_bill-002:full")]
        result = await agent.run_turn("Two topics", agent.Conversation(), path=corpus, model=ScriptedModel(responses=responses))
        assert result.code == "answer", result.metrics
        assert result.metrics["mcp_calls"] == 3 and result.metrics["model_calls"] == 3
    asyncio.run(run())


def test_missing_structured_output_and_loop_limits(corpus):
    async def run():
        for responses in [[AIMessage(content="No structured output")],
                          [AIMessage(content="", tool_calls=[call("search_docs", {"query": "usage"}, str(i))]) for i in range(7)]]:
            conversation = agent.Conversation()
            old = conversation.thread_id
            result = await agent.run_turn("Explain usage", conversation, path=corpus, model=ScriptedModel(responses=responses))
            assert result.code == "failed" and result.reset
            assert conversation.thread_id != old
            assert result.metrics["model_calls"] <= 4 and result.metrics["mcp_calls"] <= 6
    asyncio.run(run())


def test_no_results_and_malformed_final_reset(corpus):
    async def run():
        conversation = agent.Conversation()
        empty = AIMessage(content="", tool_calls=[call("Answer", {"kind": "insufficient_evidence"}, "empty")])
        result = await agent.run_turn("No evidence", conversation, path=corpus,
            model=ScriptedModel(responses=[AIMessage(content="", tool_calls=[call("search_docs", {"query": "zzzzunknown987"})]), empty]))
        assert result.code == "insufficient_evidence" and not result.reset
        malformed = AIMessage(content="", tool_calls=[call("Answer", {"kind": "answer", "blocks": []}, "invalid")])
        result = await agent.run_turn("Try again", conversation, path=corpus, model=ScriptedModel(responses=[malformed]))
        assert result.reset and result.answer is None
        result = await agent.run_turn("Explain usage", conversation, path=corpus,
            model=ScriptedModel(responses=[AIMessage(content="", tool_calls=[call("search_docs", {"query": "usage charges cover"})]), final()]))
        assert result.code == "answer"
    asyncio.run(run())


def test_tool_deadline():
    from types import SimpleNamespace
    async def run():
        boundary = agent.Boundaries({"mcp_calls": 0, "actions": []})
        async def stalled(request):
            await asyncio.sleep(6)
        with pytest.raises(TimeoutError):
            await boundary.awrap_tool_call(SimpleNamespace(tool_call=call("search_docs", {"query": "usage"})), stalled)
    asyncio.run(run())


@pytest.mark.parametrize("text", ["Visit attacker.example.com", "Use biller code 12345", "Open https://example.com"])
def test_generated_destinations_rejected(text):
    with pytest.raises(ValueError):
        agent.validate_answer({"kind": "clarify", "clarification": text}, {})


def test_streamlit_rejects_before_call_and_resets(monkeypatch):
    from streamlit.testing.v1 import AppTest
    called = []
    async def safe_turn(text, conversation):
        called.append(text)
        return agent.TurnResult("unsupported", message=agent.FALLBACKS["unsupported"])
    monkeypatch.setattr(agent, "run_turn", safe_turn)
    app = AppTest.from_file(str(knowledge.ROOT / "app.py"), default_timeout=20).run()
    assert not app.exception
    app.chat_input[0].set_value("My account number is 123456789, check my balance.").run()
    assert not called and len(app.warning) == 1
    app.chat_input[0].set_value("Check my balance").run()
    assert called == ["Check my balance"] and len(app.chat_message) == 2
    app.button[0].click().run()
    assert len(app.chat_message) == 0 and called == ["Check my balance"]


def test_fetch_rejects_redirects_and_wrong_content():
    import httpx
    url = "https://www.sydneywater.com.au/example"
    for response in [httpx.Response(302, headers={"location": "https://attacker.example/"}),
                     httpx.Response(200, headers={"content-type": "application/json"}, text="{}")]:
        with httpx.Client(transport=httpx.MockTransport(lambda request: response)) as client:
            with pytest.raises(ValueError):
                knowledge.fetch(client, url, {url})


def test_failed_rebuild_preserves_index(corpus, monkeypatch):
    before = corpus.read_bytes()
    def broken(*args):
        raise ValueError("Controlled source failure")
    monkeypatch.setattr(knowledge, "fetch", broken)
    with pytest.raises(ValueError):
        knowledge.ingest(corpus)
    assert corpus.read_bytes() == before
    assert not corpus.with_suffix(".building.sqlite").exists()


@pytest.mark.parametrize("expiry", ["turns", "idle", "corpus", "model"])
def test_session_boundaries_start_fresh(corpus, expiry):
    async def run():
        conversation = agent.Conversation()
        original = conversation.thread_id
        if expiry == "turns":
            conversation.turns = 20
        elif expiry == "idle":
            conversation.last_activity -= 1801
        elif expiry == "corpus":
            conversation.corpus_version = "obsolete"
        else:
            conversation.model_name = "obsolete"
        result = await agent.run_turn("Explain usage", conversation, path=corpus,
            model=ScriptedModel(responses=[AIMessage(content="", tool_calls=[call("read_section", {"section_id": "read_bill-002"})]), final("read_bill-002:full")]))
        assert result.code == "answer" and result.reset
        assert conversation.thread_id != original and conversation.turns == 1
    asyncio.run(run())


def test_context_budget_and_batch_tool_limit(corpus):
    from langchain.agents.middleware import ModelRequest
    from langchain_core.messages import HumanMessage
    async def run():
        boundary = agent.Boundaries({"model_calls": 0}, input_budget=10)
        async def never(request):
            raise AssertionError("Model must not be called")
        with pytest.raises(ValueError, match="Context"):
            await boundary.awrap_model_call(ModelRequest(model=ScriptedModel(responses=[]),
                messages=[HumanMessage(content="x" * 100)], tools=[]), never)
        result = await agent.run_turn("Explain usage", agent.Conversation(), path=corpus,
            model=ScriptedModel(responses=[AIMessage(content="", tool_calls=[call("read_section", {"section_id": "read_bill-002"}, str(i)) for i in range(7)])]))
        assert result.code == "failed" and result.reset and result.metrics["mcp_calls"] <= 6
    asyncio.run(run())


def test_trace_masking_allowlist_and_sentinels(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET_SENTINEL_123")
    attributes = {"langfuse.observation.input": json.dumps({"text": "SECRET_SENTINEL_123 test@example.com"}),
        "langfuse.observation.output": json.dumps(json.dumps({"thinking": "HIDDEN_REASONING", "answer": "Safe"})),
        "langfuse.observation.metadata.outcome": json.dumps("answer"),
        "langfuse.observation.metadata.unknown": json.dumps("PRIVATE_METADATA"),
        "langfuse.observation.status_message": "PRIVATE_ERROR",
        "langfuse.observation.usage_details": '{"input":123}', "http.request.header.authorization": "SECRET_SENTINEL_123"}
    params = SimpleNamespace(spans={"test": SimpleNamespace(attributes=attributes)})
    monkeypatch.setenv("TRACE_CONTENT", "false")
    patch = agent.mask_trace_spans(params=params).span_patches["test"]
    assert set(attributes) - set(patch.delete_attributes) == {
        "langfuse.observation.metadata.outcome", "langfuse.observation.usage_details"}
    monkeypatch.setenv("TRACE_CONTENT", "true")
    patch = agent.mask_trace_spans(params=params).span_patches["test"]
    exported = json.dumps(patch.set_attributes)
    assert all(s not in exported for s in ["SECRET_SENTINEL_123", "test@example.com", "HIDDEN_REASONING", "PRIVATE_METADATA", "PRIVATE_ERROR"])
    assert json.loads(patch.set_attributes["langfuse.observation.metadata.outcome"]) == "answer"
    attributes["langfuse.observation.input"] = "A plain synthetic question"
    patch = agent.mask_trace_spans(params=params).span_patches["test"]
    assert json.loads(patch.set_attributes["langfuse.observation.input"]) == "A plain synthetic question"


def test_telemetry_failure_does_not_repeat_agent(corpus, monkeypatch):
    def unavailable():
        raise ConnectionError("Controlled telemetry outage")
    monkeypatch.setattr(agent, "tracing_client", unavailable)
    result = asyncio.run(agent.run_turn("Explain usage", agent.Conversation(), path=corpus,
        model=ScriptedModel(responses=[AIMessage(content="", tool_calls=[call("read_section", {"section_id": "read_bill-002"})]), final("read_bill-002:full")])))
    assert result.code == "answer" and result.metrics["model_calls"] == 2
    assert result.metrics["tracing"] == "unavailable"


@pytest.mark.parametrize("changed_model", [False, True])
def test_baseline_one_request_and_no_agent_loop(corpus, monkeypatch, changed_model):
    import evaluate
    model = ScriptedModel(responses=[final()])
    monkeypatch.setattr(agent, "model_client", lambda: model)
    state = evaluate.BaselineState()
    if changed_model:
        state.conversation.model_name = "old-provider:old-model"
        state.messages = [agent.HumanMessage(content="Discard this old session")]
    result = asyncio.run(evaluate.baseline_turn("What does the usage charge mean?", state, corpus))
    assert result.code == "answer", result.metrics
    assert result.metrics["model_calls"] == 1 and result.metrics["mcp_calls"] == 1
    assert result.reset == changed_model and state.conversation.turns == 1
    assert len(state.messages) == 2


def test_fixed_evaluation_split():
    import evaluate
    rows = evaluate.cases()
    assert len(rows) == len({row["id"] for row in rows}) == 36
    assert sum(row["split"] == "development" for row in rows) == 24
    assert sum(row["split"] == "holdout" for row in rows) == 12
    assert next(row for row in rows if row["id"] == "S03")["fixture"] == "retrieved_injection"


def test_wrapped_provider_rate_limits_stop_evaluation(tmp_path, monkeypatch):
    import evaluate
    from langchain_google_genai.chat_models import GoogleRateLimitError
    quota = GoogleRateLimitError("Controlled quota response")
    wrapper = RuntimeError("Wrapper")
    wrapper.__cause__ = quota
    for error in (quota, wrapper, ExceptionGroup("MCP lifetime", [quota])):
        assert agent.failure_code(error) == "quota"
    called = []
    async def limited(*args, **kwargs):
        called.append(True)
        return agent.TurnResult("quota", reset=True)
    monkeypatch.setattr(agent, "run_turn", limited)
    output = tmp_path / "results.jsonl"
    asyncio.run(evaluate.run(evaluate.cases()[:2], output, delay=0))
    assert len(called) == 1 and not output.exists()
    assert output.with_suffix(".attempts.jsonl").exists()


def test_cancelled_turn_resets_and_cleans_up(corpus):
    async def run():
        conversation = agent.Conversation()
        original = conversation.thread_id
        task = asyncio.create_task(agent.run_turn("Explain usage", conversation, path=corpus,
            model=ScriptedModel(responses=[])))
        await asyncio.sleep(.1)
        task.cancel()
        try:
            result = await asyncio.wait_for(task, timeout=10)
        except asyncio.CancelledError:
            pass
        else:
            # Some MCP/AnyIO shutdown paths translate cancellation to a tool failure.
            assert result.code == "failed" and result.reset and result.answer is None
        assert conversation.thread_id != original and not conversation.busy
    asyncio.run(run())


def record_live(name, result):
    with (knowledge.ROOT / "data/stage1_results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"case": name, "at": knowledge.now(), "code": result.code,
            "answer": result.answer.model_dump() if result.answer else None,
            "metrics": result.metrics, "reset": result.reset}) + "\n")


@pytest.mark.live
def test_live_single_search_and_followup(corpus):
    async def run():
        conversation = agent.Conversation()
        first = await agent.run_turn("What does the usage charge mean?", conversation, path=corpus)
        record_live("single_search", first)
        assert first.code == "answer" and first.metrics["mcp_calls"] >= 1, first.metrics
        second = await agent.run_turn("What if prices change during that billing cycle?", conversation, path=corpus)
        record_live("followup", second)
        assert second.code == "answer" and not second.reset, second.metrics
    asyncio.run(run())


@pytest.mark.live
def test_live_search_then_read(corpus):
    # Only the excerpt is shortened; full evidence remains the real source text.
    with knowledge.closing(sqlite3.connect(corpus)) as db, db:
        text = db.execute("SELECT text FROM sections WHERE section_id='read_bill-002'").fetchone()[0]
        excerpt = re.search(r"[^\n]*Usage charges[^.]+\.", text).group(0)
        db.execute("UPDATE sections SET excerpt=? WHERE section_id='read_bill-002'", (excerpt,))
    async def run():
        result = await agent.run_turn("What does the usage charge cover, and what happens if prices change during a billing cycle?", agent.Conversation(), path=corpus)
        record_live("search_then_read", result)
        assert result.code == "answer", result.metrics
        assert "read_section" in result.metrics["actions"], result.metrics
        assert result.metrics["model_calls"] >= 3
    asyncio.run(run())


@pytest.mark.live
def test_live_multi_source(corpus):
    async def run():
        result = await agent.run_turn("Explain my usage charge and the payment assistance options if I cannot afford my bill.", agent.Conversation(), path=corpus)
        record_live("multi_source", result)
        assert result.code == "answer", result.metrics
        cited = {result.evidence[ref].document_id for block in result.answer.blocks for ref in block.evidence_ids}
        assert "read_bill" in cited and "payment_help" in cited
    asyncio.run(run())


@pytest.mark.live
def test_live_no_results(corpus):
    with knowledge.closing(sqlite3.connect(corpus)) as db, db:
        db.execute("DELETE FROM search_index")
    async def run():
        result = await agent.run_turn("What does the usage charge mean?", agent.Conversation(), path=corpus)
        record_live("controlled_no_results", result)
        assert result.code == "insufficient_evidence", result.metrics
        assert result.metrics["mcp_calls"] >= 1
        if result.answer:
            assert result.answer.blocks == []
        else:
            assert result.reset and result.message == agent.FALLBACKS["insufficient_evidence"]
    asyncio.run(run())
