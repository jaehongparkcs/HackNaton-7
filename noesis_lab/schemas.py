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
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _OmitEmpty(BaseModel):
    """Fields added after bundles were recorded are left out of the dump while they are empty, so
    a bundle recorded before the field existed still replays to the identical state digest."""
    _omit_if_empty: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def _ser(self, handler):
        data = handler(self)
        for k in self._omit_if_empty:
            if k in data and (data[k] is None or data[k] is False or data[k] in ("", [], {})):
                del data[k]
        return data


# --------------------------------------------------------------------------------------
# Experiment language (typed, allowlisted). Every option is implemented in testbed/model.py.
# --------------------------------------------------------------------------------------
class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    optimizer: Literal["adamw", "sgd_momentum", "lion"] = "adamw"
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
    # Building blocks added after the first bundles were recorded (FIXES2). At their default they
    # are left out of the dump, so config hashes, run ids and prompts of older bundles are unchanged.
    qk_norm: bool = False                              # L2-normalized queries/keys with a learned scale
    weight_tying: bool = False                         # tie the output head to the input embedding
    z_loss_coef: float = Field(0.0, ge=0.0, le=1e-2)   # training-only penalty on log Z of the output logits
    label_smoothing: float = Field(0.0, ge=0.0, le=0.3)   # training-only; val_loss stays plain cross-entropy
    init_scale: float = Field(1.0, ge=0.25, le=4.0)    # multiplies the initialization std
    # Method recipes from a paper's full text (DEEP_READ §4.2): multipliers on the baseline lr and
    # weight decay, one recipe per method, decided before any run (never swept). At (1, 1) Lion keeps
    # the harness's built-in default (lr ÷ 5, wd × 5); any other pair replaces it.
    lr_mult: float = 1.0
    wd_mult: float = 1.0

    _LATER: ClassVar[dict[str, Any]] = {"qk_norm": False, "weight_tying": False, "z_loss_coef": 0.0,
                                        "label_smoothing": 0.0, "init_scale": 1.0, "lr_mult": 1.0, "wd_mult": 1.0}
    LR_MULTS: ClassVar[tuple[float, ...]] = (0.1, 0.2, 0.33, 0.5, 1.0, 2.0)
    WD_MULTS: ClassVar[tuple[float, ...]] = (1.0, 3.0, 5.0, 10.0)

    @model_serializer(mode="wrap")
    def _ser(self, handler):
        data = handler(self)
        for k, default in self._LATER.items():
            if data.get(k) == default:
                del data[k]
        return data

    @model_validator(mode="after")
    def _heads_divide(self) -> ExperimentConfig:
        if self.lr_mult not in self.LR_MULTS:
            raise ValueError(f"lr_mult must be one of {self.LR_MULTS}")
        if self.wd_mult not in self.WD_MULTS:
            raise ValueError(f"wd_mult must be one of {self.WD_MULTS}")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("head dim must be even (RoPE)")
        return self

    def config_hash(self) -> str:
        return sha256_hex(canonical_json(self.model_dump()))[:16]

    def diff(self, other: ExperimentConfig) -> dict[str, Any]:
        return {k: getattr(other, k) for k in type(self).model_fields if getattr(self, k) != getattr(other, k)}


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
class Paper(_OmitEmpty):
    _omit_if_empty = ("directions",)

    paper_id: str
    title: str
    authors: list[str] = []
    published: str = ""
    url: str
    abstract: str
    abstract_sha256: str


Coverage = Literal["covers", "partial", "none"]
Tier = Literal["T1", "T2", "T3", "T4", "D-explore", "D-confirmed"]
# T1 curated / T2 verified-auto / T3 preprint / T4 unextracted / D-* = our own derived results
# (D-explore = a single-seed exploration, weak; D-confirmed = a paired screening). D-* claims
# shape the gap graph but never count as prior art and are never used as a literature search.


class RetrievedPaper(Paper):
    """A paper as frozen into a session corpus. Curated papers load with the defaults."""
    source: str = "curated"            # curated | arxiv
    venue: str | None = None
    peer_reviewed: bool | None = None  # None = review status unknown (OpenAlex off or no match)
    cited_by: int = 0
    doi: str | None = None
    categories: list[str] = []
    queries: list[str] = []            # which queries returned it
    directions: list[str] = []         # provenance: direction ids whose targeted search found it
    relevance: float | None = None
    relevance_components: dict[str, float] = {}
    tier: Tier = "T1"                  # paper-level: T4 = retrieved but no claim passed the quote check
    map_xy: list[float] | None = None  # TF-IDF -> PCA position (text similarity only)


# The fixed mechanism vocabulary (human-written). Extraction picks one and quotes the supporting
# text, so two methods that work "through the same thing" land on the same graph node. Free-text
# reasons almost never merge by text similarity; a closed list guarantees shared nodes.
MECHANISMS: dict[str, str] = {
    "gradient_stability": "stabilizes gradients or their norm; avoids vanishing / exploding gradients; removes the need for warm-up",
    "optimization_speed": "faster convergence; fewer steps or less compute to reach the same quality",
    "loss_landscape_smoothness": "smoother or better-conditioned loss landscape; easier optimization geometry",
    "implicit_regularization": "regularizes; reduces overfitting or co-adaptation; improves generalization from limited data",
    "position_generalization": "better use of positional information; relative position; extrapolation to other lengths",
    "long_range_dependency": "captures long-range or long-context dependencies",
    "expressivity": "increases model capacity or expressive power; richer functions per parameter",
    "representation_quality": "better internal representations; avoids rank or representation collapse",
    "attention_behavior": "changes attention patterns: entropy, sparsity, focus, collapse",
    "scale_invariance": "invariance to re-scaling or re-centering of weights or activations; implicit learning-rate adaptation",
    "computational_efficiency": "cheaper per step: lower running time or memory",
    "parameter_efficiency": "same quality with fewer parameters",
}
MechanismCategory = Literal[
    "none", "gradient_stability", "optimization_speed", "loss_landscape_smoothness", "implicit_regularization",
    "position_generalization", "long_range_dependency", "expressivity", "representation_quality",
    "attention_behavior", "scale_invariance", "computational_efficiency", "parameter_efficiency"]


class SettingFields(BaseModel):
    """What the abstract says about the setting. 'unspecified' is always a valid answer."""
    model_config = ConfigDict(extra="forbid")
    model_family: Literal["transformer", "rnn", "cnn", "mlp", "general", "unspecified"]
    task: Literal["language_modeling", "char_language_modeling", "translation", "classification",
                  "vision", "speech", "general", "unspecified"]
    scale: Literal["tiny", "small", "medium", "large", "unspecified"]
    evidence: Literal["empirical", "theoretical", "survey"]


class Claim(_OmitEmpty):
    _omit_if_empty = ("mechanism", "mechanism_category", "span_source", "setting_source", "section",
                      "superseded_by")

    claim_id: str
    paper_id: str
    dimension: str
    method: str
    setting: str                      # as stated in the abstract; used for setting matching
    claim: str
    source_span: str                  # verbatim substring of the abstract (checked at load)
    expected_outcome: Literal["improves", "no_worse", "worse", "context"]
    # How a measured result relates to this claim; set by a human, never by an LLM.
    # Human-curated, one written criterion for every claim (README "Setting coverage"):
    # covers = the abstract states the result for Transformer language models, or for Transformers
    # in general without restricting the task. Never judged by an LLM.
    covers_our_setting: bool = False
    coverage_note: str = ""
    config_change: dict[str, str] | None = None   # single-field delta this claim speaks to (None = context only)
    claims_speed: bool = False                    # the claim includes a speed / running-time component
    # Live-search additions. T1 = human-curated (the fields above). T2/T3 = auto-extracted: the LLM
    # transcribed `setting_fields`; code checked the quote and computed `coverage` by one rule.
    tier: Tier = "T1"
    coverage: Coverage | None = None              # None on T1: derived from covers_our_setting
    setting_fields: SettingFields | None = None
    peer_reviewed: bool | None = None
    published: str = ""
    extraction_event: str = ""                    # request key of the recorded extraction call
    mechanism: str = ""                           # verbatim quote of the stated reason ("" = none given)
    mechanism_category: str = ""                  # one of MECHANISMS; kept only if the quote is verbatim
    # Deep read (DEEP_READ.md). "full_text": source_span is a verbatim quote from the paper's full
    # text (section `section`), checked against the fetched HTML when it was read; the stored paper
    # record keeps the URL, version and SHA256 so anyone can re-check it (`verify-deep`).
    span_source: str = ""
    setting_source: str = ""                      # "full_text": setting_fields came from the full text
    section: str = ""
    superseded_by: str = ""                       # an abstract claim replaced by a full-text claim


class NicheSpec(BaseModel):
    """The PI's niche. A PI decision: the lab chooses experiments, not the niche."""
    model_config = ConfigDict(extra="forbid")
    title: str
    description: str
    include_terms: list[str] = []
    exclude_terms: list[str] = []
    arxiv_categories: list[str] = ["cs.LG", "cs.CL", "cs.NE"]
    date_from: str = "2017-01-01"
    date_to: str = ""                 # "" = the day of the search
    seed_papers: list[str] = []
    max_papers: int = Field(80, ge=1, le=300)
    testbed: Literal["tiny_char_lm"] = "tiny_char_lm"
    pi_overrides: list[str] = []      # "field=value" keys the PI runs despite a soft-reject
    pi_dismissed_holds: list[str] = []   # "field=value" keys whose T4 hold the PI dismissed


class Fixture(BaseModel):
    """A predefined test hypothesis (NOT a measurement). Labeled 'fixture' in the UI."""
    fixture_id: str
    kind: Literal["known_in_corpus", "untested_setting", "generated", "gap"]
    title: str
    statement: str
    allowed_fields: list[str]         # the only config fields the Scientist may touch
    cited_claim_ids: list[str] = []   # claims a result is compared against (human-curated)
    origin: Literal["fixture", "generated"] = "fixture"   # generated = picked by code from the queue
    candidate_key: str | None = None  # "field=value" this hypothesis tests (excluded from the queue)
    queue_priority: list[Any] | None = None   # priority tuple that put it here (generated only)
    # Gap hypotheses (kind == "gap"): the typed delta, label and explanation all come from code.
    required_delta: dict[str, str] | None = None
    novelty_type: str = ""            # transfer | combination | resolution (computed by code)
    gap_ids: list[str] = []
    gap: dict[str, Any] | None = None # best gap: type, score components, explanation path
    mock: dict[str, Any] = Field(default_factory=dict)   # used only by the offline mock LLM


class QueueItem(_OmitEmpty):
    """One candidate experiment, ordered by `stats.candidate_queue` (pure code, no LLM)."""
    _omit_if_empty = ("delta", "gap_ids", "gap_score", "novelty_type")

    key: str                          # "field=value"
    field: str
    value: str
    priority: list[Any]               # [outcome rank, -n supporting claims, key]
    supporting_claim_ids: list[str]
    # open = runnable now; soft_rejected = covered by an auto-extracted / preprint claim (end of the
    # queue, PI may override); held = possible overlap with an unextracted paper (PI review);
    # rejected_prior_art = covered by a curated (T1) claim.
    status: Literal["open", "soft_rejected", "held", "rejected_prior_art"] = "open"
    covering_claim_ids: list[str] = []
    note: str = ""
    # Gap queue (hypothesis engine): the delta may have two fields; order is by gap_score.
    delta: dict[str, str] = {}
    gap_ids: list[str] = []
    gap_score: float | None = None
    novelty_type: str = ""


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


class QueryDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    precise: list[str]       # term-level queries
    broad: list[str]         # area-level queries


class QueryPlan(BaseModel):
    """LLM-proposed search queries. Code caps the total and appends one query per runnable change."""
    model_config = ConfigDict(extra="forbid")
    dimensions: list[QueryDimension]


class Direction(BaseModel):
    """One research direction proposed by the scout. A search direction, never evidence: every
    field is used only to build searches. `possibly_related_titles` are search hints."""
    model_config = ConfigDict(extra="forbid")
    title: str
    idea: str
    mechanism: str
    search_terms: list[str]
    possibly_related_titles: list[str]
    building_blocks: list[str]         # block names it would touch; [] = outside the testbed


class ScoutOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    directions: list[Direction]


class DirectionRecord(BaseModel):
    """A direction as frozen into the corpus: the scout's text plus what code did with it."""
    direction_id: str                  # assigned by code
    origin: Literal["scout", "deterministic"]
    title: str
    idea: str = ""
    mechanism: str = ""
    search_terms: list[str] = []
    building_blocks: list[str] = []
    dropped_blocks: list[str] = []     # names the scout gave that are not building blocks
    queries: list[dict[str, Any]] = []         # [{q, n_results}]
    title_hints: list[dict[str, Any]] = []     # [{title, status: found|not_found, paper_id}]
    paper_ids: list[str] = []


class ExtractedClaim(BaseModel):
    """What the LLM transcribes from ONE abstract. Code verifies the quote and maps the rest."""
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    method: str
    config_change: str       # one of the allowed "field=value" pairs, or "" (context only)
    claim: str
    source_span: str         # must be a verbatim substring of the abstract, else dropped by code
    direction: Literal["improves", "no_worse", "worse", "context"]
    setting: SettingFields
    mechanism_category: MechanismCategory   # which listed mechanism the abstract gives as the reason; "none" if none
    mechanism: str           # the abstract's own words for that reason, quoted verbatim; "" if none


class ExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[ExtractedClaim]


# ------------------------------------------------------------------ deep read (full text)
DeepSection = Literal["abstract", "introduction", "method", "experimental_setup", "results", "limitations",
                      "conclusion", "appendix_setup"]


class DeepSetting(BaseModel):
    """Where the paper's experiments were run, as the full text states it."""
    model_config = ConfigDict(extra="forbid")
    model_family: Literal["transformer", "rnn", "cnn", "mlp", "general", "unspecified"]
    task: Literal["language_modeling", "char_language_modeling", "translation", "classification",
                  "vision", "speech", "general", "unspecified"]
    parameter_count: str      # as stated ("125M", "1.3 billion"), or "unspecified"; code maps it to a scale
    dataset: str
    training_steps: str
    batch_size: str
    quote: str
    section: DeepSection


class DeepRecipe(BaseModel):
    """A hyperparameter the paper used or recommends for a method, relative to AdamW when it says so."""
    model_config = ConfigDict(extra="forbid")
    method: str               # an allowed "field=value" change, copied exactly
    hyperparameter: Literal["lr", "weight_decay", "warmup", "betas", "schedule", "other"]
    value_as_stated: str
    ratio_to_adamw_min: float | None    # e.g. "3-10x smaller" -> 0.1 and 0.333; None if not stated
    ratio_to_adamw_max: float | None
    value: float | None                 # absolute value used for this method, if stated
    adamw_value: float | None           # the paper's own AdamW value for the same hyperparameter, if stated
    quote: str
    section: DeepSection


class DeepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str               # allowed "field=value" change, or "" if none fits
    direction: Literal["improves", "no_worse", "worse", "context"]
    setting_note: str
    quote: str
    section: DeepSection


class DeepMechanism(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str
    mechanism_category: MechanismCategory
    quote: str
    section: DeepSection


class DeepLimitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str               # allowed "field=value" change the limitation is about, or ""
    quote: str
    section: DeepSection


class DeepSmallScale(BaseModel):
    """An experiment at <= 10M parameters or on character-level language modeling."""
    model_config = ConfigDict(extra="forbid")
    method: str
    direction: Literal["improves", "no_worse", "worse", "context"]
    parameter_count: str
    char_level: bool
    quote: str
    section: DeepSection


class DeepFindings(BaseModel):
    """What the LLM transcribes from ONE paper's full text. Every item carries a verbatim quote and
    the section it came from; code drops any item whose quote is not in that section. No field is a
    measurement of ours, a verdict, a coverage label or a score."""
    model_config = ConfigDict(extra="forbid")
    setting: list[DeepSetting]          # 0 or 1
    recipes: list[DeepRecipe]
    results: list[DeepResult]
    mechanisms: list[DeepMechanism]
    limitations: list[DeepLimitation]
    small_scale_evidence: list[DeepSmallScale]


class LiteratureOutputV2(BaseModel):
    """Gate v2. The LLM lists EVERY retrieved claim that tests the same comparison; it does not
    pick a "nearest" one. `stats.gate_decision` then chooses the verdict deterministically."""
    model_config = ConfigDict(extra="forbid")
    same_comparison_claim_ids: list[str]     # ids copied from the retrieved list; [] if none
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


class GapScientistOutput(BaseModel):
    """A hypothesis written for a graph gap. The delta, the gap type, its score and the novelty
    label are code's; the LLM writes the text and names the claims it rests on."""
    model_config = ConfigDict(extra="forbid")
    statement: str
    mechanism: str
    predicted_direction: Literal["lower_val_loss", "higher_val_loss", "no_change"]
    falsification_rule: str
    config_changes: list[ConfigChange]
    grounded_in: list[str]   # claim ids from the gap's explanation path (validated by code)


class StatedGapProposal(BaseModel):
    """A typed delta that would test an author-stated limitation. Code validates it."""
    model_config = ConfigDict(extra="forbid")
    config_changes: list[ConfigChange]
    rationale: str


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


class PriorArtResult(_OmitEmpty):
    """Verdict + the passage. The passage text comes from the snapshot, never from the LLM."""
    _omit_if_empty = ("same_comparison_claim_ids", "partial_overlap_claim_ids", "contested")

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
    covers_our_setting: bool | None = None        # curated (T1) or computed by the coverage rule
    coverage_note: str | None = None
    coverage: Coverage | None = None
    tier: Tier | None = None                      # tier of the cited claim
    gate: Literal["reject", "soft_reject", "run"] = "run"   # derived by stats.gate_outcome
    # Gate v2: every retrieved claim the LLM says tests the same comparison; code picked the one
    # above (strongest first). `contested`: the only covering claims disagree in sign.
    same_comparison_claim_ids: list[str] = []
    # Claims that cover only PART of a combination (a strict-subset change). They never decide the
    # verdict — a claim about one method is not prior art for two methods tested together — but they
    # feed the tighten pass, which can narrow the combination to its untested field.
    partial_overlap_claim_ids: list[str] = []
    contested: bool = False


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
