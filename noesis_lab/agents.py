"""Scientist and Critic agents. LLM-facing only: nothing here computes a measurement or a decision."""
from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from .literature import TESTBED_DESCRIPTION, Snapshot
from .llm import LLM
from .schemas import (
    Analysis,
    CriticPostOutput,
    CriticPreOutput,
    ExperimentConfig,
    Fixture,
    ScientistOutput,
    apply_delta,
)

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def schema_summary(allowed: list[str]) -> str:
    props = ExperimentConfig.model_json_schema()["properties"]
    lines = []
    for f in allowed:
        p = props[f]
        dom = p.get("enum") or (f"{p.get('type')}, min {p.get('minimum')}, max {p.get('maximum')}")
        lines.append(f"- {f}: {dom} (default {p.get('default')})")
    return "\n".join(lines)


class Scientist:
    def __init__(self, llm: LLM, snap: Snapshot):
        self.llm, self.snap = llm, snap
        self.system = (PROMPTS / "scientist.md").read_text()

    def propose(self, fx: Fixture, baseline: ExperimentConfig, context_claims: list[str]
                ) -> tuple[ScientistOutput, ExperimentConfig, dict, str]:
        user = (f"{TESTBED_DESCRIPTION}\n\nFixture {fx.fixture_id}: {fx.title}\nHypothesis: {fx.statement}\n\n"
                f"Baseline configuration:\n{json.dumps(baseline.model_dump(), sort_keys=True, indent=1)}\n\n"
                f"ALLOWED FIELDS:\n{schema_summary(fx.allowed_fields)}\n\n"
                "Related claims from the curated snapshot (context only):\n"
                + "\n".join(f"- {c}" for c in context_claims) + "\n")
        holder: dict = {}

        def validate(o: ScientistOutput) -> None:
            if not o.config_changes:
                raise ValueError("config_changes must not be empty")
            fields = [c.field for c in o.config_changes]
            if len(set(fields)) != len(fields):
                raise ValueError("duplicate fields in config_changes")
            bad = [f for f in fields if f not in fx.allowed_fields]
            if bad:
                raise ValueError(f"fields {bad} not in ALLOWED FIELDS {fx.allowed_fields}")
            delta = {c.field: c.value for c in o.config_changes}
            try:
                cfg = apply_delta(baseline, delta)
            except ValidationError as e:
                raise ValueError(f"config rejected by validator: {e}") from e
            if cfg == baseline:
                raise ValueError("config_changes leave the baseline unchanged")
            holder["cfg"], holder["delta"] = cfg, baseline.diff(cfg)

        out, eid = self.llm.call("scientist", self.system, user, ScientistOutput, validate=validate)
        return out, holder["cfg"], holder["delta"], eid


class Critic:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.pre_system = (PROMPTS / "critic_pre.md").read_text()
        self.post_system = (PROMPTS / "critic_post.md").read_text()

    def review(self, fx: Fixture, prop: ScientistOutput, baseline: ExperimentConfig,
               cand: ExperimentConfig, delta: dict) -> tuple[CriticPreOutput, str]:
        user = (f"Hypothesis ({fx.fixture_id}): {prop.statement}\nMechanism: {prop.mechanism}\n"
                f"Predicted direction: {prop.predicted_direction}\nFalsification: {prop.falsification_rule}\n\n"
                f"Baseline config: {json.dumps(baseline.model_dump(), sort_keys=True)}\n"
                f"Config delta applied by code: {json.dumps(delta, sort_keys=True)}\n")
        return self.llm.call("critic_pre", self.pre_system, user, CriticPreOutput)

    def interpret(self, a: Analysis, branch_text: str) -> tuple[str, str, str]:
        """Returns (reading, source, event_id). `source` is 'llm' or 'template' (number guard)."""
        facts = analysis_facts(a)
        user = ("FACTS (computed by code; copy numbers exactly):\n"
                + "\n".join(f"- {k}: {v}" for k, v in facts.items())
                + f"\n\nDecision-rule outcome (fixed by code): {branch_text}\n")
        out, eid = self.llm.call("critic_post", self.post_system, user, CriticPostOutput)
        bad = number_violations(out.reading, facts, branch_text)
        if bad:
            return template_reading(a), "template", eid
        return out.reading, "llm", eid


# ---------------------------------------------------------------- number guard (invariant)
_NUM = re.compile(r"\d+(?:\.\d+)?")


def f4(x: float) -> str:
    return f"{x:.4f}"


def analysis_facts(a: Analysis) -> dict[str, str]:
    cf = a.counterfactual
    f = {
        "label": a.label,
        "noise floor (baseline seed-to-seed SD of val_loss)": f4(a.noise.sd),
        "baseline seeds in noise floor": str(a.noise.n),
        "paired seeds": ", ".join(str(p.seed) for p in a.pairs),
        "per-seed delta (candidate minus baseline, negative = better)":
            ", ".join(f"seed {p.seed}: {f4(p.delta)}" for p in a.pairs),
        "mean delta": f4(a.mean_delta),
        "mean improvement in noise SDs": f"{abs(a.improvement_in_noise_sd):.2f}"
            + (" (candidate better)" if a.improvement_in_noise_sd > 0 else " (candidate worse)"),
        "tokens seen, candidate / baseline": f"{a.token_ratio:.2f}",
        "mean delta at equal tokens": f4(a.mean_equal_token_delta) if a.mean_equal_token_delta is not None else "n/a",
        "single-run counterfactual": cf.summary,
        "single-run keep count": f"{cf.n_keep} of {cf.n_pairings}",
    }
    return f


def number_violations(text: str, facts: dict[str, str], extra: str = "") -> list[str]:
    allowed = set(_NUM.findall(" ".join(facts.values()) + " " + extra))
    return [n for n in _NUM.findall(text) if n not in allowed and not (len(n) == 1)]


def template_reading(a: Analysis) -> str:
    cf = a.counterfactual
    direction = "better" if a.mean_improvement > 0 else "worse"
    return (f"[deterministic summary] Across {len(a.pairs)} paired seeds the candidate's mean delta "
            f"was {f4(a.mean_delta)} (negative = better), i.e. {abs(a.improvement_in_noise_sd):.2f} "
            f"noise SDs {direction}, against a measured noise floor of {f4(a.noise.sd)}. "
            f"{cf.summary} {a.label}.")
