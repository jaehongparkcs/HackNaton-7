"""Offline scripted LLM for tests and the CPU smoke profile. NOT evidence; never a headline result.

Responses come from the `mock` hints in data/fixtures.json, so the pipeline can be exercised
end to end with no API key. A bundle produced this way is labeled `llm_mode: mock`.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel

from .schemas import (
    CriticPostOutput,
    CriticPreOutput,
    DeepFindings,
    ExtractionOutput,
    Fixture,
    GapScientistOutput,
    LiteratureOutput,
    LiteratureOutputV2,
    QueryPlan,
    ScientistOutput,
    ScoutOutput,
    StatedGapProposal,
)

# Keyword -> allowed config change, for the scripted extraction mock only.
_MOCK_KEYWORDS = [("rmsnorm", "norm=rmsnorm"), ("swiglu", "activation=swiglu"),
                  ("squared relu", "activation=relu2"), ("rotary", "pos_encoding=rope"),
                  ("dropout", "dropout=0.1"), ("post-ln", "norm_position=post")]


def _mock_mechanism(abstract: str) -> dict:
    m = re.search(r"(?:because it|by|through) ((?:improv|stabili|reduc|smooth)[^.,;]*)", abstract)
    quote = m.group(1) if m else ""
    cat = ("position_generalization" if "extrapolation" in quote else
           "loss_landscape_smoothness" if "loss landscape" in quote else
           "gradient_stability" if "gradient" in quote else "none")
    return {"mechanism_category": cat, "mechanism": quote if cat != "none" else ""}


def _mock_extraction(user: str) -> ExtractionOutput:
    """Scripted stand-in for extraction: quote the first sentence that names a known method."""
    claims = []
    for m in re.finditer(r"### PAPER (\S+)\nTitle: (.*)\nAbstract: (.*)", user):
        pid, _title, abstract = m.groups()
        low = abstract.lower()
        for kw, change in _MOCK_KEYWORDS:
            sent = next((s for s in re.split(r"(?<=\.)\s+", abstract) if kw in s.lower()), None)
            if sent:
                claims.append({
                    "paper_id": pid, "method": kw, "config_change": change,
                    "claim": f"(mock) the abstract reports a result for {kw}.", "source_span": sent,
                    "direction": "worse" if " worse" in sent.lower() else "improves",
                    **_mock_mechanism(abstract),
                    "setting": {"model_family": "transformer" if "transformer" in low else "unspecified",
                                "task": "language_modeling" if "language model" in low else "unspecified",
                                "scale": "unspecified", "evidence": "empirical"}})
                break
    return ExtractionOutput(claims=claims)


_DEEP_KEYWORDS = [*_MOCK_KEYWORDS, ("lion", "optimizer=lion")]


def _mock_deep(user: str) -> DeepFindings:
    """Scripted stand-in for the full-text read: quote whole sentences that match simple cues."""
    out: dict[str, list] = {k: [] for k in ("setting", "recipes", "results", "mechanisms", "limitations",
                                            "small_scale_evidence")}
    for name, text in re.findall(r"## SECTION (\w+)\n(.*?)(?=\n\n## SECTION |\n*$)", user, re.S):
        for sent in re.split(r"(?<=\.)\s+", text.strip()):
            low = sent.lower()
            method = next((c for kw, c in _DEEP_KEYWORDS if kw in low), "")
            params = re.search(r"(\d+(?:\.\d+)?\s*[MB])\s*parameters", sent)
            if params and not out["setting"] and name in ("experimental_setup", "appendix_setup", "method"):
                out["setting"].append({
                    "model_family": "transformer" if "transformer" in low else "rnn" if "rnn" in low else "unspecified",
                    "task": "char_language_modeling" if "character" in low else "language_modeling"
                    if "language model" in low else "translation" if "translation" in low else "unspecified",
                    "parameter_count": params.group(1), "dataset": "unspecified", "training_steps": "unspecified",
                    "batch_size": "unspecified", "quote": sent, "section": name})
            for hp, cue in (("lr", "learning rate"), ("weight_decay", "weight decay")):
                rng = re.search(cue + r"[^,;]*?(\d+)-(\d+)x (smaller|larger)", low)
                if method and rng:
                    a, b = int(rng.group(1)), int(rng.group(2))
                    lo, hi = (1 / b, 1 / a) if rng.group(3) == "smaller" else (a, b)
                    out["recipes"].append({"method": method, "hyperparameter": hp,
                                           "value_as_stated": rng.group(0)[len(cue):].strip(),
                                           "ratio_to_adamw_min": lo, "ratio_to_adamw_max": hi, "value": None,
                                           "adamw_value": None, "quote": sent, "section": name})
            if method and name == "results":
                out["results"].append({"method": method, "direction": "worse" if "worse" in low else "improves",
                                       "setting_note": "as in the paper", "quote": sent, "section": name})
            if method and name in ("limitations", "conclusion"):
                out["limitations"].append({"method": method, "quote": sent, "section": name})
            if method and "character-level" in low and name in ("results", "experimental_setup") and params:
                out["small_scale_evidence"].append({"method": method, "direction": "worse" if "worse" in low else "improves",
                                                    "parameter_count": params.group(1), "char_level": True,
                                                    "quote": sent, "section": name})
    return DeepFindings(**out)


def make_mock(fixtures: dict[str, Fixture]):
    """`fixtures` is shared with the session: generated candidates are registered there (with
    auto-filled `mock` hints) before they are evaluated."""
    def fid_of(user: str) -> str:
        m = re.search(r"(?:Hypothesis id: |Fixture |Hypothesis \()(\w+)", user)
        if not m or m.group(1) not in fixtures:
            raise KeyError(f"mock cannot identify fixture in prompt: {user[:200]!r}")
        return m.group(1)

    def fn(role: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        if schema is QueryPlan:
            return QueryPlan(dimensions=[
                {"name": "normalization", "precise": ['"RMSNorm" AND "Transformer"'], "broad": ['"layer normalization"']},
                {"name": "activation", "precise": ['"SwiGLU"'], "broad": ['"activation function" AND "Transformer"']},
                {"name": "position", "precise": ['"rotary position embedding"'], "broad": ['"positional encoding"']}])
        if schema is ScoutOutput:
            mk = lambda t, terms, blocks, titles=(): {  # noqa: E731
                "title": t, "idea": f"(mock) try {t}.", "mechanism": "(mock) scripted reason.",
                "search_terms": list(terms), "possibly_related_titles": list(titles), "building_blocks": list(blocks)}
            return ScoutOutput(directions=[
                mk("Gated feed-forward blocks", ["SwiGLU", "gated linear unit"], ["activation"],
                   ["SwiGLU feed-forward blocks for character-level Transformers", "A paper that does not exist"]),
                mk("Rotary positions", ["rotary position embedding"], ["pos_encoding"]),
                mk("Normalization choice", ["RMSNorm", "layer normalization"], ["norm", "norm_position"]),
                mk("Regularization", ["dropout"], ["dropout", "attention_sparsity"]),
                mk("Optimizer and schedule", ["learning rate schedule", "SGD momentum"], ["optimizer", "schedule"]),
                mk("Activation and position together", ["SwiGLU", "rotary position embedding"], ["activation", "pos_encoding"]),
                mk("Sparse attention", ["sparse attention"], []),
                mk("Curriculum over sequence length", ["curriculum learning"], [])])
        if schema is ExtractionOutput:
            return _mock_extraction(user)
        if schema is DeepFindings:
            return _mock_deep(user)
        if schema is StatedGapProposal:
            f, v = re.search(r"METHOD: (\w+)=(\S+)", user).groups()
            return StatedGapProposal(config_changes=[{"field": f, "value": v}],
                                     rationale="(mock) tests the stated limitation in our setting.")
        if schema is CriticPostOutput:
            return CriticPostOutput(reading="(mock) The measured deltas are compared with the noise "
                                            "floor above. This is a screening result, not a confirmation.")
        fx = fixtures[fid_of(user)]
        m = fx.mock
        if schema is LiteratureOutputV2:
            return LiteratureOutputV2(same_comparison_claim_ids=[m["claim_id"]] if m["same_comparison"] else [],
                                      rationale=m.get("rationale", "mock"))
        if schema is LiteratureOutput:
            return LiteratureOutput(same_comparison=m["same_comparison"], claim_id=m.get("claim_id", ""),
                                    rationale=m.get("rationale", "mock"))
        if schema is ScientistOutput:
            return ScientistOutput(
                statement=fx.statement, mechanism="(mock) scripted mechanism.",
                predicted_direction="lower_val_loss",
                falsification_rule="(mock) falsified if the paired deltas are not below the noise floor.",
                config_changes=m["config_changes"])
        if schema is GapScientistOutput:
            changes = m["config_changes"]
            if "REVISION." in user:                      # narrow to the first field of the previous delta
                prev = json.loads(re.search(r"Return a strict subset of (\{.*\})", user).group(1))
                changes = [{"field": f, "value": v} for f, v in sorted(prev.items())][:1]
            return GapScientistOutput(
                statement=fx.statement if "REVISION." not in user else f"(mock, narrowed) {changes[0]['field']} alone: {fx.statement}",
                mechanism="(mock) scripted mechanism.", predicted_direction="lower_val_loss",
                falsification_rule="(mock) falsified if the paired deltas are not below the noise floor.",
                config_changes=changes, grounded_in=m["grounded_in"])
        if schema is CriticPreOutput:
            return CriticPreOutput(verdict="accept", confounds=["(mock) throughput may differ"],
                                   required_controls=["report throughput separately"], reasons="(mock) ok")
        raise KeyError(schema)
    return fn
