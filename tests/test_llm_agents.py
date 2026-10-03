
import pytest
from pydantic import BaseModel

from noesis_lab.agents import analysis_facts, number_violations, template_reading
from noesis_lab.config import load_config
from noesis_lab.llm import LLM, LLMError, ReplayMiss, request_key, strict_schema
from noesis_lab.schemas import LiteratureOutput
from noesis_lab.store import Store

from .test_stats import BASE, run

CFG = load_config()["llm"]


class Out(BaseModel):
    answer: str


def mock_fn(role, system, user, schema):
    return Out(answer="ok")


def test_strict_schema_closes_objects():
    s = strict_schema(LiteratureOutput)
    assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])


def test_record_then_replay_roundtrip(tmp_path):
    rec = tmp_path / "llm.jsonl"
    live = LLM("mock", CFG, recordings_path=rec, mock_fn=mock_fn)
    out, _ = live.call("r", "sys", "usr", Out)
    rep = LLM("replay", CFG, recordings_path=rec)
    out2, _ = rep.call("r", "sys", "usr", Out)
    assert out == out2
    with pytest.raises(ReplayMiss):
        rep.call("r", "sys", "DIFFERENT PROMPT", Out)          # replay never goes live


def test_invalid_output_retries_once_then_fails_loudly(tmp_path):
    calls = []

    def bad(role, system, user, schema):
        calls.append(user)
        return Out(answer="nope")

    llm = LLM("mock", CFG, recordings_path=tmp_path / "r.jsonl", mock_fn=bad)

    def validate(o):
        raise ValueError("answer must be 'yes'")

    with pytest.raises(LLMError):
        llm.call("r", "s", "u", Out, validate=validate)
    assert len(calls) == 2 and "answer must be 'yes'" in calls[1]    # error fed back exactly once


def test_events_are_logged(tmp_path):
    st = Store(tmp_path / "n.sqlite")
    LLM("mock", CFG, store=st, mock_fn=mock_fn).call("r", "s", "u", Out)
    ev = st.events()
    assert len(ev) == 1 and ev[0]["role"] == "r" and ev[0]["valid"]


def test_request_key_sensitive_to_every_part():
    base = request_key("m", "e", "r", "s", "u", {})
    for args in [("m2", "e", "r", "s", "u", {}), ("m", "e2", "r", "s", "u", {}), ("m", "e", "r2", "s", "u", {}),
                 ("m", "e", "r", "s2", "u", {}), ("m", "e", "r", "s", "u2", {}), ("m", "e", "r", "s", "u", {"a": 1})]:
        assert request_key(*args) != base


def test_number_guard_blocks_invented_numbers():
    a = run(BASE, [1.505, 1.515, 1.475])
    facts = analysis_facts(a)
    ok = f"The noise floor was {a.noise.sd:.4f} and the single-run loop kept {a.counterfactual.n_keep} of {a.counterfactual.n_pairings}."
    assert number_violations(ok, facts) == []
    assert number_violations("The change gave a 12% speedup.", facts) == ["12"]
    assert number_violations("The delta was 0.0999.", facts)
    assert "NOT CONFIRMED" in template_reading(a)
