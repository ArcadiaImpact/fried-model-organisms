import math

import numpy as np
from mu_decisiveness.questions import Question
from mu_decisiveness.oracle import Comparison, p_util_from_pick


def test_p_util_slot_i_positive():
    q = Question(id="pos", template="{item_A}{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})
    assert np.isclose(p_util_from_pick(0.8, slot_a="i", question=q), 0.8)


def test_p_util_slot_j_positive():
    q = Question(id="pos", template="{item_A}{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})
    assert np.isclose(p_util_from_pick(0.8, slot_a="j", question=q), 0.2)


def test_p_util_slot_j_negative_valence():
    q = Question(id="neg", template="{item_A}{item_B}", valence=-1, answers={"A": ["A"], "B": ["B"]})
    assert np.isclose(p_util_from_pick(0.8, slot_a="j", question=q), 0.8)


def test_p_a_from_logprobs():
    from mu_decisiveness.oracle import p_a_from_logprobs
    q = Question(id="pos", template="x", valence=1, answers={"A": ["A"], "B": ["B"]})
    tops = [{"token": "A", "lp": math.log(0.75)}, {"token": "B", "lp": math.log(0.25)}]
    assert abs(p_a_from_logprobs(tops, q) - 0.75) < 1e-6


def test_p_a_from_picks_jeffreys():
    from mu_decisiveness.oracle import p_a_from_picks
    p, a, b = p_a_from_picks(["A", "A", "B"])
    assert abs(p - 0.625) < 1e-6 and a == 2 and b == 1


# tests/test_oracle.py (append)
from mu_decisiveness.oracle import build_batch_requests, parse_batch_results


def test_build_batch_requests_logprob():
    q = Question(id="pos", template="A:{item_A} B:{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})
    comps = [Comparison(i=0, j=1, item_i="cat", item_j="dog", question=q, slot_a="i")]
    reqs = build_batch_requests(comps, model="gpt-4.1", mode="logprob", n_samples=1)
    assert reqs[0]["custom_id"] == "0_1_i_pos_0"
    assert reqs[0]["url"] == "/v1/chat/completions"
    assert reqs[0]["body"]["model"] == "gpt-4.1"
    assert reqs[0]["body"]["logprobs"] is True
    assert "cat" in reqs[0]["body"]["messages"][0]["content"]


def test_parse_batch_results_logprob():
    import math
    q = Question(id="pos", template="A:{item_A} B:{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})
    comps = [Comparison(i=0, j=1, item_i="cat", item_j="dog", question=q, slot_a="i")]
    reqs = build_batch_requests(comps, model="gpt-4.1", mode="logprob", n_samples=1)
    cid = reqs[0]["custom_id"]
    by_cid = {cid: comps[0]}
    raw_line = __import__("json").dumps({
        "custom_id": cid,
        "response": {"body": {"choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": "A", "logprob": math.log(0.75)},
            {"token": "B", "logprob": math.log(0.25)}]}]}}]}},
    })
    obs = parse_batch_results([raw_line], by_cid, mode="logprob")
    assert len(obs) == 1
    assert abs(obs[0].p_util - 0.75) < 1e-6   # slot_a="i", valence +1


def test_openai_oracle_messages_and_extra_body():
    from mu_decisiveness.oracle import OpenAIOracle
    import os
    os.environ.setdefault("OPENAI_API_KEY", "x")
    o = OpenAIOracle("m", system_prompt="SYS", extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    assert o._messages("hi") == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
    assert o._extra() == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    o2 = OpenAIOracle("m")
    assert o2._messages("hi") == [{"role": "user", "content": "hi"}] and o2._extra() == {}


def test_prefill_mode_request_shape():
    import asyncio, os, types
    from mu_decisiveness.oracle import OpenAIOracle, p_a_from_logprobs
    os.environ.setdefault("OPENAI_API_KEY", "x")
    q = Question(id="pos", template="{item_A} vs {item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})
    o = OpenAIOracle("m", mode="prefill", extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    captured = {}

    async def fake_create(**kw):
        captured.update(kw)
        top = [types.SimpleNamespace(token="A", logprob=math.log(0.8)),
               types.SimpleNamespace(token="B", logprob=math.log(0.2))]
        content = [types.SimpleNamespace(token="A", logprob=math.log(0.8), top_logprobs=top)]
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            logprobs=types.SimpleNamespace(content=content))])
    o._client.chat.completions.create = fake_create
    tops = asyncio.run(o._call_prefill_logprobs("p", q))
    assert captured["messages"][-1] == {"role": "assistant", "content": "<answer>"}
    assert captured["max_completion_tokens"] == 1 and captured["top_logprobs"] == 20
    assert captured["extra_body"]["continue_final_message"] is True
    assert captured["extra_body"]["add_generation_prompt"] is False
    assert captured["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert abs(p_a_from_logprobs(tops, q) - 0.8) < 1e-6


def test_openai_oracle_rebuilds_client_on_new_event_loop(monkeypatch):
    """Each compare() phase runs under its own asyncio.run(); the client must not carry a closed loop's pool."""
    import asyncio
    from mu_decisiveness.oracle import OpenAIOracle
    o = OpenAIOracle(model="m", mode="prefill", base_url="http://127.0.0.1:1/v1")
    first = o._client

    async def phase():
        o._ensure_client_loop()
        return o._client

    assert asyncio.run(phase()) is first          # first loop keeps the original client
    second = asyncio.run(phase())                 # new loop -> rebuilt client
    assert second is not first
    assert str(second.base_url).startswith("http://127.0.0.1:1")
