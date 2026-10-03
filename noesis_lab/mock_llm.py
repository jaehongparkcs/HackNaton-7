"""Offline scripted LLM for tests and the CPU smoke profile. NOT evidence; never a headline result.

Responses come from the `mock` hints in data/fixtures.json, so the pipeline can be exercised
end to end with no API key. A bundle produced this way is labeled `llm_mode: mock`.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from .schemas import CriticPostOutput, CriticPreOutput, Fixture, LiteratureOutput, ScientistOutput


def make_mock(fixtures: dict[str, Fixture]):
    """`fixtures` is shared with the session: generated candidates are registered there (with
    auto-filled `mock` hints) before they are evaluated."""
    def fid_of(user: str) -> str:
        m = re.search(r"(?:Hypothesis id: |Fixture |Hypothesis \()(\w+)", user)
        if not m or m.group(1) not in fixtures:
            raise KeyError(f"mock cannot identify fixture in prompt: {user[:200]!r}")
        return m.group(1)

    def fn(role: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        if schema is CriticPostOutput:
            return CriticPostOutput(reading="(mock) The measured deltas are compared with the noise "
                                            "floor above. This is a screening result, not a confirmation.")
        fx = fixtures[fid_of(user)]
        m = fx.mock
        if schema is LiteratureOutput:
            return LiteratureOutput(same_comparison=m["same_comparison"], claim_id=m.get("claim_id", ""),
                                    rationale=m.get("rationale", "mock"))
        if schema is ScientistOutput:
            return ScientistOutput(
                statement=fx.statement, mechanism="(mock) scripted mechanism.",
                predicted_direction="lower_val_loss",
                falsification_rule="(mock) falsified if the paired deltas are not below the noise floor.",
                config_changes=m["config_changes"])
        if schema is CriticPreOutput:
            return CriticPreOutput(verdict="accept", confounds=["(mock) throughput may differ"],
                                   required_controls=["report throughput separately"], reasons="(mock) ok")
        raise KeyError(schema)
    return fn
