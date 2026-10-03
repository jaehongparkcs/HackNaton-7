"""Frozen contracts shared by every component (BUILD_PLAN s10: only the core owner edits this file).

Two groups of models:
  * LLM-facing   - what an agent may return. Deliberately has no field for measurements,
                   status labels or decisions, so an LLM *cannot* emit them.
  * Deterministic - produced only by code from runs (RunResult, Analysis, NextAction, ...).
"""
from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Experiment language (typed, allowlisted). Every option is implemented in testbed/model.py.
# --------------------------------------------------------------------------------------
class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    optimizer: Literal["adamw", "sgd_momentum"] = "adamw"
    lr: float = Field(2e-3, ge=1e-5, le=1e-1)
    weight_decay: float = Field(0.01, ge=0.0, le=0.5)
    grad_clip: float = Field(1.0, ge=0.0, le=10.0)
    schedule: Literal["constant", "cosine"] = "cosine"
    warmup_frac: float = Field(0.05, ge=0.0, le=0.5)
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    norm_position: Literal["pre", "post"] = "pre"
    activation: Literal["gelu", "relu2", "swiglu"] = "gelu"
    pos_encoding: Literal["learned", "rope"] = "learned"
    n_layer: int = Field(4, ge=1, le=8)
    n_head: int = Field(4, ge=1, le=8)
    n_embd: int = Field(128, ge=16, le=512)
    dropout: float = Field(0.0, ge=0.0, le=0.5)
    batch_size: int = Field(64, ge=4, le=256)
    seq_len: int = Field(128, ge=16, le=512)

    @model_validator(mode="after")
    def _heads_divide(self) -> ExperimentConfig:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("head dim must be even (RoPE)")
        return self

    def config_hash(self) -> str:
        return sha256_hex(canonical_json(self.model_dump()))[:16]

    def diff(self, other: ExperimentConfig) -> dict[str, Any]:
        a, b = self.model_dump(), other.model_dump()
        return {k: b[k] for k in b if a[k] != b[k]}


def apply_delta(base: ExperimentConfig, delta: dict[str, Any]) -> ExperimentConfig:
    """Validated delta on a base config. Unknown fields / out-of-range values raise."""
    return ExperimentConfig.model_validate({**base.model_dump(), **delta})


class Profile(BaseModel):
    """Harness settings. Part of every run's cache key, so smoke and full never mix."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    deterministic: bool = False
    budget_mode: Literal["time", "steps"] = "time"
    train_seconds: float = 60.0
    max_steps: int = 0
    eval_every_s: float = 10.0
    eval_every_steps: int = 50
    eval_batches: int = 20
    eval_batch_size: int = 64
    prewarm_steps: int = 2
    baseline_overrides: dict[str, Any] = Field(default_factory=dict)

    def profile_hash(self) -> str:
        return sha256_hex(canonical_json(self.model_dump()))[:12]


# --------------------------------------------------------------------------------------
# Literature snapshot
# --------------------------------------------------------------------------------------
class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = []
    published: str = ""
    url: str
    abstract: str
    abstract_sha256: str


class Claim(BaseModel):
    claim_id: str
    paper_id: str
    dimension: str
    method: str
    setting: str                      # as stated in the abstract; used for setting matching
    claim: str
    source_span: str                  # verbatim substring of the abstract (checked at load)
    expected_outcome: Literal["improves", "no_worse", "context"]
    # How a measured result relates to this claim; set by a human, never by an LLM.
    # Human-curated, one written criterion for every claim (README "Setting coverage"):
    # covers = the abstract states the result for Transformer language models, or for Transformers
    # in general without restricting the task. Never judged by an LLM.
    covers_our_setting: bool = False
    coverage_note: str = ""
    config_change: dict[str, str] | None = None   # single-field delta this claim speaks to (None = context only)
    claims_speed: bool = False                    # the claim includes a speed / running-time component


class Fixture(BaseModel):
    """A predefined test hypothesis (NOT a measurement). Labeled 'fixture' in the UI."""
    fixture_id: str
    kind: Literal["known_in_corpus", "untested_setting", "generated"]
    title: str
    statement: str
    allowed_fields: list[str]         # the only config fields the Scientist may touch
    cited_claim_ids: list[str] = []   # claims a result is compared against (human-curated)
    origin: Literal["fixture", "generated"] = "fixture"   # generated = picked by code from the queue
    candidate_key: str | None = None  # "field=value" this hypothesis tests (excluded from the queue)
    queue_priority: list[Any] | None = None   # priority tuple that put it here (generated only)
    mock: dict[str, Any] = Field(default_factory=dict)   # used only by the offline mock LLM


class QueueItem(BaseModel):
    """One candidate experiment, ordered by `stats.candidate_queue` (pure code, no LLM)."""
    key: str                          # "field=value"
    field: str
    value: str
    priority: list[Any]               # [outcome rank, -n supporting claims, key]
    supporting_claim_ids: list[str]


# --------------------------------------------------------------------------------------
# LLM-facing outputs (strict JSON). None carries a measurement, a label or a decision.
# --------------------------------------------------------------------------------------
class PriorArtVerdictKind(StrEnum):
    known = "known_in_corpus"
    setting_untested = "method_known_setting_untested"
    not_found = "not_found_in_corpus"


class LiteratureOutput(BaseModel):
    """The LLM only judges whether a retrieved claim tests the same change. The verdict is derived
    by `stats.prior_art_verdict` from this bool and the human-curated `covers_our_setting`."""
    model_config = ConfigDict(extra="forbid")
    same_comparison: bool
    claim_id: str            # the retrieved claim that tests the same change ("" if none)
    rationale: str


class ConfigChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    value: str               # parsed and range-checked by code (apply_delta)


class ScientistOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statement: str
    mechanism: str
    predicted_direction: Literal["lower_val_loss", "higher_val_loss", "no_change"]
    falsification_rule: str
    config_changes: list[ConfigChange]


class CriticPreOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["accept", "reject"]
    confounds: list[str]
    required_controls: list[str]
    reasons: str


class CriticPostOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reading: str             # plain-language reading; may quote only numbers it was given


# --------------------------------------------------------------------------------------
# Deterministic records
# --------------------------------------------------------------------------------------
class CurvePoint(BaseModel):
    train_s: float
    step: int
    tokens_seen: int
    val_loss: float


class RunResult(BaseModel):
    run_id: str
    config_hash: str
    config: ExperimentConfig
    profile_hash: str
    seed: int
    status: Literal["ok", "failed"]
    error: str = ""
    val_loss: float | None = None
    tokens_seen: int = 0
    steps: int = 0
    train_seconds: float = 0.0
    n_params: int = 0
    curve: list[CurvePoint] = []
    env: dict[str, Any] = {}
    run_order: int = 0
    started_at: str = ""


class PriorArtResult(BaseModel):
    """Verdict + the passage. The passage text comes from the snapshot, never from the LLM."""
    verdict: PriorArtVerdictKind
    claim_id: str | None
    paper_id: str | None
    paper_title: str | None
    paper_url: str | None
    passage: str | None
    rationale: str
    snapshot_size: int
    label: str                # UI string, e.g. "not found in curated snapshot (10 papers)"
    event_id: str
    same_comparison: bool = False                 # the only LLM judgment behind the verdict
    covers_our_setting: bool | None = None        # human-curated, from the claim
    coverage_note: str | None = None              # human-curated, from the claim


class NoiseFloor(BaseModel):
    seeds: list[int]
    run_ids: list[str]
    val_losses: list[float]
    mean: float
    sd: float                 # sample SD (ddof=1)
    n: int


class SeedPair(BaseModel):
    seed: int
    baseline_run_id: str
    candidate_run_id: str
    baseline_val_loss: float
    candidate_val_loss: float
    delta: float              # candidate - baseline; NEGATIVE means the candidate is better
    baseline_tokens: int
    candidate_tokens: int
    equal_token_delta: float | None   # delta of interpolated curves at min(tokens); see stats.py
    baseline_train_seconds: float = 0.0
    candidate_train_seconds: float = 0.0


class Counterfactual(BaseModel):
    """What a single-run keep/revert loop (autoresearch-style) would have decided."""
    same_seed_decisions: dict[int, str]       # seed -> "keep" | "revert"  (the headline)
    n_pairings: int                           # candidate seed x baseline seed (secondary)
    n_keep: int
    keep_rate: float
    decision_depends_on_seed: bool            # True if keep and revert both occur across pairings
    wording: Literal["never", "almost_always", "luck"] = "never"
    same_seed_summary: str = ""
    cross_seed_summary: str = ""
    summary: str


class Analysis(BaseModel):
    analysis_id: str
    hypothesis_id: str
    candidate_config_hash: str
    noise: NoiseFloor
    pairs: list[SeedPair]
    mean_delta: float
    mean_improvement: float                   # = -mean_delta
    improvement_in_noise_sd: float            # mean_improvement / noise.sd
    all_seeds_improve: bool
    all_seeds_worse: bool
    token_ratio: float                        # mean candidate tokens / mean baseline tokens
    baseline_tokens_per_s: float | None = None    # secondary "speed cost/benefit" metric,
    candidate_tokens_per_s: float | None = None   # never used by the decision rule
    throughput_ratio: float | None = None         # candidate / baseline tokens per second
    mean_equal_token_delta: float | None
    counterfactual: Counterfactual
    branch: Literal["promising", "no_improvement", "harmful"]
    label: str                                # always "SCREENING RESULT — NOT CONFIRMED"
    seed_disagreement: bool                   # |improvement| beyond 1 SD but seeds disagree


class NextAction(BaseModel):
    branch: Literal["promising", "no_improvement", "harmful"]
    action: Literal["extra_seeds", "next_candidate", "literature_check"]
    detail: dict[str, Any] = {}
    rule_text: str
    rationale: str


class DerivedEvidence(BaseModel):
    evidence_id: str
    claim_id: str
    analysis_id: str
    relation: Literal["agrees", "contradicts", "inconclusive",
                      "consistent_in_our_setting", "not_reproduced_in_our_setting"]
    note: str
