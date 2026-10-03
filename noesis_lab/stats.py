"""Statistician / Decider. Pure functions of stored runs: no I/O, no LLM, no clock, no randomness.

`rederive` calls these same functions on the shipped runs and demands an exact match.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Collection, Iterable, Sequence
from typing import Any

import numpy as np

from .schemas import (
    Analysis,
    Claim,
    Counterfactual,
    Coverage,
    DerivedEvidence,
    ExperimentConfig,
    NextAction,
    NoiseFloor,
    PriorArtVerdictKind,
    QueueItem,
    RunResult,
    SeedPair,
    sha256_hex,
)
from .search.coverage import counts_as_covered

SCREENING_LABEL = "SCREENING RESULT — NOT CONFIRMED"

# Pre-registered next-action rule (BUILD_PLAN s6). Shown verbatim on the dashboard.
# Written before the first run; thresholds live in config.yaml `rule:`.
RULE_TEXT = (
    "Let SD = seed-to-seed standard deviation of the baseline (noise floor) and "
    "improvement = baseline val_loss − candidate val_loss, averaged over the paired seeds.\n"
    "1. HARMFUL: candidate is worse by more than 1×SD  → record a contradiction with the cited "
    "claim; next action = targeted Literature Agent check of whether the snapshot already reports this.\n"
    "2. PROMISING: improvement > 1×SD and every paired seed improves → next action = extra seeds "
    "for the same change (toward confirmation).\n"
    "3. NO IMPROVEMENT: otherwise (including improvement > 1×SD where the seeds disagree in sign, "
    "flagged as seed disagreement) → next action = the next candidate in the code-generated "
    "queue (ordered from the literature snapshot).\n"
    "The rule is applied to every screening result. Extra seeds run at most once per hypothesis; "
    "after any action the lab continues with the next queued candidate while the budget lasts.\n"
    "The LLM never selects the branch."
)


# Rule v2, written 2026-10-03 after the `explore1` session and before any v2 session was recorded.
# v1 required every paired seed to agree for PROMISING but not for HARMFUL, so one seed could
# decide "harmful" (explore1, RMSNorm: 1.07 SD worse with per-seed deltas -0.010 / +0.028 / +0.009).
# v2 makes the agreement requirement symmetric. It is not applied retroactively: a bundle is
# analysed with the version it was recorded under.
RULE_V2_TEXT = (
    "Rule version 2. Let SD = seed-to-seed standard deviation of the baseline (noise floor) and "
    "improvement = baseline val_loss − candidate val_loss, averaged over the paired seeds.\n"
    "1. HARMFUL: candidate is worse by more than 1×SD AND every paired seed is worse → record the "
    "result against the cited claims; next action = targeted Literature Agent check.\n"
    "2. PROMISING: improvement > 1×SD AND every paired seed improves → next action = extra seeds "
    "for the same change (toward confirmation).\n"
    "3. NO IMPROVEMENT: otherwise. A mean beyond 1×SD in either direction where the seeds disagree "
    "in sign is NO IMPROVEMENT, flagged as seed disagreement → next action = the next candidate "
    "in the code-generated queue.\n"
    "The rule is applied to every screening result. Extra seeds run at most once per hypothesis.\n"
    "The LLM never selects the branch."
)
RULE_TEXTS = {1: RULE_TEXT, 2: RULE_V2_TEXT}


def _r(x: float) -> float:
    """Canonical rounding so recomputation is bit-identical across platforms."""
    return float(f"{x:.10g}")


# --------------------------------------------------------------------------- prior-art verdict
def prior_art_verdict(same_comparison: bool, coverage: Coverage | bool | None) -> PriorArtVerdictKind:
    """Pure function of one LLM judgment (same comparison?) and the claim's coverage (a human flag
    on curated claims, the written rule on auto-extracted ones). For overlap, `covers` and
    `partial` both count as covered: novelty errors lean toward "covered"."""
    if not same_comparison:
        return PriorArtVerdictKind.not_found
    if coverage is True or coverage in ("covers", "partial"):
        return PriorArtVerdictKind.known
    return PriorArtVerdictKind.setting_untested


def gate_outcome(verdict: PriorArtVerdictKind, tier: str | None, overridden: bool = False) -> str:
    """What the gate does with a verdict. Only a curated (T1) claim can hard-reject; an
    auto-extracted or preprint claim (T2/T3) soft-rejects, which the PI may override."""
    if verdict != PriorArtVerdictKind.known:
        return "run"
    if tier in (None, "T1"):
        return "reject"
    return "run" if overridden else "soft_reject"


# --------------------------------------------------------------------------- candidate queue
# Single-field changes the lab may propose. Pure retunes (lr, batch size) are excluded.
QUEUE_FIELDS = ("norm", "norm_position", "activation", "pos_encoding", "schedule", "optimizer", "dropout",
                "qk_norm", "weight_tying", "z_loss_coef", "label_smoothing", "warmup_frac", "grad_clip",
                "init_scale")
OUTCOME_RANK = {"improves": 0, "no_worse": 1, "context": 2, "worse": 3}
# The testbed's runnable single-field changes ("field=value"). Extraction may map a claim only
# onto one of these; anything else is context.
ALLOWED_CHANGES = (
    "activation=relu2", "activation=swiglu", "dropout=0.1", "grad_clip=0", "init_scale=0.5", "init_scale=2.0",
    "label_smoothing=0.1", "norm=rmsnorm", "norm_position=post", "optimizer=lion", "optimizer=sgd_momentum",
    "pos_encoding=rope", "qk_norm=true", "schedule=constant", "warmup_frac=0", "warmup_frac=0.02",
    "warmup_frac=0.1", "weight_tying=true", "z_loss_coef=0.0001")


def gate_decision(same: Sequence[Claim], *, contested_ids: Collection[str] = (), overridden: bool = False
                  ) -> tuple[PriorArtVerdictKind, Claim | None, str, bool]:
    """Gate v2: (verdict, the claim that determined it, gate action, contested) from ALL claims the
    LLM says test the same comparison. Strongest first, ties by lowest claim id:
      1. a curated (T1) claim that covers our setting          → known, reject
      2. an auto-extracted (T2/T3) covering claim, uncontested → known, soft-reject (run if the PI overrides)
      2b. only contested covering claims (they disagree in sign) → known, run, contested
      3. a same-comparison claim that does not cover us        → method known, setting untested, run
      4. none                                                  → not found, run
    Which claim the LLM happens to mention first can no longer flip the outcome."""
    by_id = sorted(same, key=lambda c: c.claim_id)
    covering = [c for c in by_id if counts_as_covered(c)]
    t1 = [c for c in covering if c.tier == "T1"]
    if t1:
        return PriorArtVerdictKind.known, t1[0], "reject", False
    auto = [c for c in covering if c.claim_id not in set(contested_ids)]
    if auto:
        return PriorArtVerdictKind.known, auto[0], "run" if overridden else "soft_reject", False
    if covering:
        return PriorArtVerdictKind.known, covering[0], "run", True
    if by_id:
        return PriorArtVerdictKind.setting_untested, by_id[0], "run", False
    return PriorArtVerdictKind.not_found, None, "run", False


PARAM_VARIANT_FIELDS = ("cosine_final_frac",)   # planned, not built yet


def fmt_value(v: Any) -> str:
    """A config value as it appears in a "field=value" key: true / false for booleans, and the
    allowlist's own spelling for numbers (0, 0.02, 0.0001, 2.0)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        for spelled in (f"{v:g}", f"{v}", f"{v:.4f}".rstrip("0")):
            if any(k.split("=", 1)[1] == spelled for k in ALLOWED_CHANGES):
                return spelled
        return f"{v:g}"
    return str(v)


def delta_key(delta: dict[str, Any]) -> str:
    """Canonical key of a config delta: "a=b" or "a=b+c=d" (fields sorted)."""
    return "+".join(f"{f}={fmt_value(delta[f])}" for f in sorted(delta))


def parse_delta_key(key: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in key.split("+"))


def valid_delta(delta: dict[str, Any]) -> bool:
    """The experiment language: one allowlisted single change, or any two that touch different
    fields. Same-field pairs and anything outside the allowlist are rejected."""
    keys = [f"{f}={v}" for f, v in delta.items()]
    return 1 <= len(keys) <= 2 and all(k in ALLOWED_CHANGES for k in keys)


def combine(key_a: str, key_b: str) -> dict[str, str] | None:
    """Two single changes as one delta, or None if they touch the same field / are not allowlisted."""
    if key_a not in ALLOWED_CHANGES or key_b not in ALLOWED_CHANGES:
        return None
    (fa, va), (fb, vb) = key_a.split("=", 1), key_b.split("=", 1)
    return None if fa == fb else {fa: va, fb: vb}


def testability(delta: dict[str, Any]) -> float:
    """1 = expressible as a typed delta (1-2 fields); 0.3 = needs a parameterized variant that is
    not built yet; 0 = outside the testbed (shown on the map, never queued)."""
    if valid_delta(delta):
        return 1.0
    if delta and all(f in PARAM_VARIANT_FIELDS or f"{f}={v}" in ALLOWED_CHANGES for f, v in delta.items()) \
            and len(delta) <= 2:
        return 0.3
    return 0.0


def contested_claim_ids(claims_for_key: Sequence[Claim]) -> list[str]:
    """Auto-extracted claims that would count as covering our setting (`covers` or `partial`, one
    bucket for the overlap question) but disagree in sign (improves vs worse). The literature is
    contested there, not settled. Curated (T1) claims are never treated as contested."""
    cell = [c for c in claims_for_key if c.tier != "T1" and counts_as_covered(c)
            and c.expected_outcome in ("improves", "worse")]
    if {"improves", "worse"} <= {c.expected_outcome for c in cell}:
        return sorted(c.claim_id for c in cell)
    return []


def literature_status(key: str, claims_for_key: Sequence[Claim], soft_rejected: Collection[str] = (),
                      held: Collection[str] = (), overrides: Collection[str] = (), *,
                      contested_open: bool = False) -> tuple[str, list[str], str]:
    """(status, covering claim ids, note) of one candidate from the frozen claims that map onto it.
    Only directional claims cover (a context claim reports no result); `covers` and `partial` both
    count (novelty errors lean toward "covered"). T1 → rejected; T2/T3 or a gate soft-reject →
    soft_rejected unless the PI overrides; a T4 text-overlap hold → held.

    `contested_open` (the hypothesis engine): auto-extracted claims that disagree in sign within
    the covered setting bucket do not cover the topic. A disagreement is an open question, not prior art, so the
    candidate is "open (contested)" unless some other, uncontested claim covers it. The
    literature-ordered queue of earlier bundles keeps the old behavior (False)."""
    contested = set(contested_claim_ids(claims_for_key)) if contested_open else set()
    covering = sorted((c for c in claims_for_key if c.expected_outcome != "context" and counts_as_covered(c)
                       and c.claim_id not in contested), key=lambda c: c.claim_id)
    t1 = [c.claim_id for c in covering if c.tier == "T1"]
    auto = [c.claim_id for c in covering if c.tier != "T1"]
    if t1:
        return "rejected_prior_art", t1, "covered by a curated (T1) claim"
    if (auto or key in set(soft_rejected)) and key not in set(overrides):
        return "soft_rejected", auto, "covered by an auto-extracted / preprint claim; the PI may override"
    if key in set(held):
        return "held", [], "possible overlap with an unextracted paper; waiting for PI review"
    if auto or key in set(soft_rejected):
        return "open", auto, "soft-reject overridden by the PI"
    if contested:
        return "open", [], "open (contested): the claims for this change disagree in sign in the same setting"
    return "open", [], ""


QUEUE_STATUS_ORDER = {"open": 0, "soft_rejected": 1, "held": 2, "rejected_prior_art": 3}


def candidate_queue(baseline: ExperimentConfig, claims: Iterable[Claim], tested: Collection[str], *,
                    soft_rejected: Collection[str] = (), held: Collection[str] = (),
                    overrides: Collection[str] = ()) -> list[QueueItem]:
    """Every allowlisted single-field change from the baseline that at least one claim maps to,
    in deterministic order; no LLM, no I/O.

    Status, from the frozen claims (directional claims only; a context claim reports no result):
      (a) covered by a curated T1 claim                       -> rejected_prior_art (not run)
      (b) covered by a T2/T3 claim, or soft-rejected at the gate -> soft_rejected (end of the queue;
          the PI may override)
      (c) flagged by the T4 text-overlap check (`held`)        -> held for PI review
      (d) the rest are open, ordered by: best expected outcome among supporting claims
          (improves < no_worse < context < worse), more supporting claims first, alphabetical.
    `tested` holds "field=value" keys already tested or gated out this session.
    """
    base = baseline.model_dump()
    support: dict[str, list[Claim]] = {}
    for c in claims:
        ch = c.config_change
        if not ch or len(ch) != 1:
            continue
        (field, value), = ch.items()
        if field not in QUEUE_FIELDS or str(base.get(field)) == str(value):
            continue
        support.setdefault(f"{field}={value}", []).append(c)
    items = []
    for key, cs in support.items():
        if key in set(tested):
            continue
        field, value = key.split("=", 1)
        ids = sorted(c.claim_id for c in cs)
        rank = min(OUTCOME_RANK[c.expected_outcome] for c in cs)
        status, cov_ids, note = literature_status(key, cs, soft_rejected, held, overrides)
        items.append(QueueItem(key=key, field=field, value=value, priority=[rank, -len(ids), key],
                               supporting_claim_ids=ids, status=status, covering_claim_ids=cov_ids,
                               note=note))
    return sorted(items, key=lambda i: (QUEUE_STATUS_ORDER[i.status], *i.priority))


def open_items(queue: Sequence[QueueItem]) -> list[QueueItem]:
    return [q for q in queue if q.status == "open"]


# --------------------------------------------------------------------------- noise floor
def noise_floor(runs: Sequence[RunResult]) -> NoiseFloor:
    ok = sorted((r for r in runs if r.status == "ok"), key=lambda r: r.seed)
    if len(ok) < 2:
        raise ValueError("noise floor needs >= 2 successful baseline runs")
    vals = [r.val_loss for r in ok]
    return NoiseFloor(seeds=[r.seed for r in ok], run_ids=[r.run_id for r in ok],
                      val_losses=vals, mean=_r(statistics.fmean(vals)),
                      sd=_r(statistics.stdev(vals)), n=len(vals))


# --------------------------------------------------------------------------- equal tokens
def _interp_at(r: RunResult, tokens: int) -> float:
    xs = [p.tokens_seen for p in r.curve]
    ys = [p.val_loss for p in r.curve]
    return float(np.interp(tokens, xs, ys))


def equal_token_delta(base: RunResult, cand: RunResult) -> float | None:
    """candidate − baseline val_loss, both read off their eval curves at min(final tokens).

    Caveat (documented in README): with a cosine schedule, a mid-curve point is not a finished
    schedule, so this is a throughput-separation diagnostic, not a headline number.
    """
    if not base.curve or not cand.curve:
        return None
    t = min(base.tokens_seen, cand.tokens_seen)
    return _r(_interp_at(cand, t) - _interp_at(base, t))


# --------------------------------------------------------------------------- counterfactual
def counterfactual(cand_runs: Sequence[RunResult], base_runs: Sequence[RunResult]) -> Counterfactual:
    """An autoresearch-style loop keeps a change if ONE run beats the previous ONE run.

    Headline: the same-seed decisions (a loop with a fixed seed). Secondary: every (candidate seed,
    baseline seed) pairing, i.e. what happens if the seed varies between runs. Both use runs we
    already have, so this costs no extra compute.
    """
    cands = sorted(cand_runs, key=lambda r: r.seed)
    bases = sorted(base_runs, key=lambda r: r.seed)
    base_by_seed = {b.seed: b for b in bases}
    same = {c.seed: ("keep" if c.val_loss < base_by_seed[c.seed].val_loss else "revert")
            for c in cands if c.seed in base_by_seed}
    n = len(cands) * len(bases)
    keep = sum(1 for c in cands for b in bases if c.val_loss < b.val_loss)
    dep = 0 < keep < n
    flips = min(keep, n - keep)
    if not dep:
        wording = "never"
        cross = (f"If the seed varies between runs, a keep/revert loop would {'KEEP' if keep else 'REVERT'} "
                 f"this change in all {n} pairings: seed never changed the decision.")
    elif flips <= 2:
        wording = "almost_always"
        major = "KEEP" if keep > n - keep else "REVERT"
        cross = (f"If the seed varies between runs, a keep/revert loop would almost always {major} "
                 f"this change ({flips} of {n} pairings flip).")
    else:
        wording = "luck"
        cross = (f"If the seed varies between runs, the decision is decided by seed luck: KEEP in {keep} "
                 f"of {n} pairings, REVERT in {n - keep}.")
    head = ("A single-run keep/revert loop with a fixed seed would have decided: "
            + ", ".join(f"seed {k} {v}" for k, v in same.items()) + ".")
    return Counterfactual(same_seed_decisions=same, n_pairings=n, n_keep=keep,
                          keep_rate=_r(keep / n if n else 0.0), decision_depends_on_seed=dep,
                          wording=wording, same_seed_summary=head, cross_seed_summary=cross,
                          summary=f"{head} {cross}")


# --------------------------------------------------------------------------- throughput
def _tokens_per_s(tokens: Sequence[int], seconds: Sequence[float]) -> float | None:
    """Pooled tokens/s over the paired runs. Secondary metric: never enters the decision rule."""
    tot_s = sum(seconds)
    return _r(sum(tokens) / tot_s) if tot_s > 0 else None


def throughput_drift(runs: Sequence[RunResult], baseline_config_hash: str, *,
                     threshold: float = 0.10) -> dict:
    """Informational drift check: median tokens/s of the baseline-era runs (baseline config, run
    order first) vs the late-session runs. Different configs have different costs, so a flag is a
    prompt to look at the throughput plot, never a verdict."""
    pts = sorted(((r.run_order, r.config_hash, r.tokens_seen / r.train_seconds) for r in runs
                  if r.status == "ok" and r.train_seconds > 0), key=lambda t: t[0])
    early = [t for _, h, t in pts if h == baseline_config_hash]
    late = [t for _, h, t in pts if h != baseline_config_hash]
    if not early or not late:
        return {"early_median": None, "late_median": None, "relative_change": None, "flag": False}
    e, l_ = statistics.median(early), statistics.median(late)
    rel = (l_ - e) / e
    return {"early_median": _r(e), "late_median": _r(l_), "relative_change": _r(rel),
            "flag": abs(rel) > threshold}


# --------------------------------------------------------------------------- paired analysis
def analyze(hypothesis_id: str, cand_runs: Sequence[RunResult], base_runs: Sequence[RunResult],
            noise: NoiseFloor, *, all_base_runs: Sequence[RunResult] | None = None,
            promising_sd: float = 1.0, harmful_sd: float = 1.0, rule_version: int = 1) -> Analysis:
    """Paired by seed: delta = candidate − baseline on the same seed (negative = better).

    `all_base_runs` (default: base_runs) is every baseline run available, used only for the
    single-run counterfactual: a one-run loop could have compared against any of them.
    """
    cands = {r.seed: r for r in cand_runs if r.status == "ok"}
    bases = {r.seed: r for r in base_runs if r.status == "ok"}
    seeds = sorted(set(cands) & set(bases))
    if not seeds:
        raise ValueError("no paired seeds with successful runs on both sides")
    pairs, eq = [], []
    for s in seeds:
        c, b = cands[s], bases[s]
        e = equal_token_delta(b, c)
        eq.append(e)
        pairs.append(SeedPair(seed=s, baseline_run_id=b.run_id, candidate_run_id=c.run_id,
                              baseline_val_loss=b.val_loss, candidate_val_loss=c.val_loss,
                              delta=c.val_loss - b.val_loss, baseline_tokens=b.tokens_seen,
                              candidate_tokens=c.tokens_seen, equal_token_delta=e,
                              baseline_train_seconds=b.train_seconds,
                              candidate_train_seconds=c.train_seconds))
    deltas = [p.delta for p in pairs]
    mean_delta = statistics.fmean(deltas)
    imp = -mean_delta
    sd = noise.sd
    imp_sd = (imp / sd) if sd > 0 else (0.0 if imp == 0 else math.copysign(1e9, imp))
    all_imp, all_worse = all(d < 0 for d in deltas), all(d > 0 for d in deltas)

    seed_disagreement = False
    worse_beyond = mean_delta > harmful_sd * sd and mean_delta > 0
    if worse_beyond and (rule_version < 2 or all_worse):      # v2: HARMFUL also needs every seed worse
        branch = "harmful"
    elif worse_beyond:
        branch, seed_disagreement = "no_improvement", True
    elif imp > promising_sd * sd and imp > 0 and all_imp:
        branch = "promising"
    else:
        branch = "no_improvement"
        seed_disagreement = imp > promising_sd * sd and imp > 0 and not all_imp

    eq_vals = [e for e in eq if e is not None]
    mb = statistics.fmean(p.baseline_tokens for p in pairs)
    mc = statistics.fmean(p.candidate_tokens for p in pairs)
    tps_b, tps_c = _tokens_per_s([p.baseline_tokens for p in pairs], [p.baseline_train_seconds for p in pairs]), \
        _tokens_per_s([p.candidate_tokens for p in pairs], [p.candidate_train_seconds for p in pairs])
    aid = "ana_" + sha256_hex(hypothesis_id + "|" + ",".join(sorted(
        [p.candidate_run_id for p in pairs] + [p.baseline_run_id for p in pairs])))[:10]
    return Analysis(
        analysis_id=aid, hypothesis_id=hypothesis_id,
        candidate_config_hash=next(iter(cands.values())).config_hash, noise=noise, pairs=pairs,
        mean_delta=_r(mean_delta), mean_improvement=_r(imp), improvement_in_noise_sd=_r(imp_sd),
        all_seeds_improve=all_imp, all_seeds_worse=all_worse,
        token_ratio=_r(mc / mb if mb else 1.0),
        baseline_tokens_per_s=tps_b, candidate_tokens_per_s=tps_c,
        throughput_ratio=_r(tps_c / tps_b) if tps_b and tps_c else None,
        mean_equal_token_delta=_r(statistics.fmean(eq_vals)) if eq_vals else None,
        counterfactual=counterfactual([cands[s] for s in seeds],
                                      [r for r in (all_base_runs or base_runs) if r.status == "ok"]),
        branch=branch, label=SCREENING_LABEL, seed_disagreement=seed_disagreement)


# --------------------------------------------------------------------------- next action
def next_action(a: Analysis, *, extra_seeds: Sequence[int], queue: Sequence[QueueItem],
                rule_version: int = 1) -> NextAction:
    rule_text = RULE_TEXTS[rule_version]
    if a.branch == "promising":
        return NextAction(
            branch="promising", action="extra_seeds", detail={"seeds": list(extra_seeds)},
            rule_text=rule_text,
            rationale=(f"mean improvement is {a.improvement_in_noise_sd:.2f}× the noise SD and all "
                       f"{len(a.pairs)} seeds improve → run extra seeds for the same change."))
    if a.branch == "harmful":
        return NextAction(
            branch="harmful", action="literature_check", detail={},
            rule_text=rule_text,
            rationale=(f"candidate is worse by {-a.improvement_in_noise_sd:.2f}× the noise SD → "
                       "record a contradiction and ask the Literature Agent whether the snapshot "
                       "already reports this."))
    nxt = next((q.key for q in queue if q.status == "open"), None)
    why = ("seeds disagree in sign" if a.seed_disagreement
           else f"mean improvement is {a.improvement_in_noise_sd:.2f}× the noise SD (within ±1×)")
    return NextAction(
        branch="no_improvement", action="next_candidate", detail={"candidate": nxt},
        rule_text=rule_text,
        rationale=(f"{why} → next candidate in the generated queue: {nxt if nxt else 'none left'}."))


# --------------------------------------------------------------------------- derived evidence
def relate_to_claim(a: Analysis, claim: Claim) -> DerivedEvidence:
    """How the measured branch relates to a cited claim. Pure table lookup; no LLM.

    `agrees` / `contradicts` are used only when the claim covers our setting (human-curated).
    Otherwise the relation says what we saw *in our setting* and does not overclaim."""
    cov = claim.covers_our_setting
    if claim.expected_outcome == "context":
        rel, note = "inconclusive", "cited claim makes no directional prediction for val_loss"
    elif a.branch == "harmful":
        ok = claim.expected_outcome == "worse"
        rel = (("agrees" if ok else "contradicts") if cov
               else ("consistent_in_our_setting" if ok else "not_reproduced_in_our_setting"))
        note = "candidate measured worse than baseline beyond the noise floor"
    elif a.branch == "promising":
        ok = claim.expected_outcome != "worse"
        rel = (("agrees" if ok else "contradicts") if cov
               else ("consistent_in_our_setting" if ok else "not_reproduced_in_our_setting"))
        note = "candidate measured better than baseline beyond the noise floor"
    elif claim.expected_outcome == "no_worse":
        rel = "agrees" if cov else "consistent_in_our_setting"
        note = "no measurable difference beyond the noise floor (claim: comparable)"
    else:
        rel, note = "inconclusive", "claimed effect not detected beyond the noise floor"
    if claim.claims_speed and a.throughput_ratio is not None and a.throughput_ratio < 0.95:
        note += "; partial: speed claim not reproduced (implementation-dependent)"
    return DerivedEvidence(evidence_id="ev_" + sha256_hex(a.analysis_id + claim.claim_id)[:10],
                           claim_id=claim.claim_id, analysis_id=a.analysis_id, relation=rel,
                           note=note)


# --------------------------------------------------------------------------- scoring the engine
def prediction_outcome(predicted_direction: str, branch: str) -> str:
    """Did a gap's predicted direction match the measured branch?
      hit            the predicted sign matches ("+" and promising, "-" and harmful; "0" = no worse:
                     anything but harmful)
      miss           the measured branch has the opposite sign
      null           no improvement: the measurement decided nothing
      no_prediction  the gap predicts no direction (contradictions, combination hints)"""
    if predicted_direction == "0":
        return "miss" if branch == "harmful" else "hit"
    if predicted_direction not in ("+", "-"):
        return "no_prediction"
    if branch == "no_improvement":
        return "null"
    return "hit" if (predicted_direction == "+") == (branch == "promising") else "miss"


# --------------------------------------------------------------------------- explore cheaply, confirm rigorously
EXPLORATION_LABEL = "EXPLORATION — NOT A RESULT"
FINALIST_RULE_TEXT = (
    "Exploration runs one seed per hypothesis and is never a result. Finalists (code): keep the best "
    "single-seed improvement per mechanism cell, then take the top N by single-seed improvement among "
    "those that improved. If fewer than the configured minimum improved, the best remaining cell is "
    "confirmed anyway, so the session still measures how a single-seed estimate holds up. Ties: "
    "alphabetical. Only the paired screening that follows can set a label, a branch or the incumbent."
)


def single_seed_estimate(cand: RunResult, base: RunResult, noise_sd: float) -> dict[str, float]:
    """What a one-run keep/revert loop sees: candidate vs the incumbent on the SAME seed."""
    delta = cand.val_loss - base.val_loss
    imp_sd = (-delta / noise_sd) if noise_sd > 0 else 0.0
    return {"delta": _r(delta), "improvement": _r(-delta), "improvement_in_noise_sd": _r(imp_sd)}


def select_finalists(explored: Sequence[dict[str, Any]], top_n: int = 2, min_n: int = 1) -> list[str]:
    """Keys of the hypotheses that go on to paired confirmation. `explored` rows carry `key`,
    `cell` and `improvement_in_noise_sd`. See FINALIST_RULE_TEXT."""
    best: dict[str, dict] = {}
    for row in sorted(explored, key=lambda r: (-r["improvement_in_noise_sd"], r["key"])):
        best.setdefault(row["cell"], row)
    ranked = sorted(best.values(), key=lambda r: (-r["improvement_in_noise_sd"], r["key"]))
    chosen = [r for r in ranked if r["improvement_in_noise_sd"] > 0][:top_n]
    for r in ranked:
        if len(chosen) >= min(min_n, len(ranked)):
            break
        if r not in chosen:
            chosen.append(r)
    return [r["key"] for r in chosen]


def shrinkage(single_seed_sd: float, paired_sd: float) -> float:
    """Single-seed estimate minus the paired estimate, in noise SDs. Positive = the one-run number
    was optimistic. This is the autoresearch-vs-paired comparison, measured inside one session."""
    return _r(single_seed_sd - paired_sd)


def should_promote(a: Analysis, n_seeds_required: int) -> bool:
    """A finalist becomes the incumbent only if it is still PROMISING on the full seed set and
    every one of those seeds improves."""
    return a.branch == "promising" and a.all_seeds_improve and len(a.pairs) >= n_seeds_required


# --------------------------------------------------------------------------- the noise floor is an estimate
def sd_interval(values: Sequence[float], level: float = 0.95) -> tuple[float, float] | None:
    """Exhaustive bootstrap interval for the sample SD: every resample with replacement of the
    n values (n^n of them, so no randomness), percentile interval. With 5 seeds the SD is itself a
    rough estimate; this shows how rough. None for n < 3 or n > 6 (too many resamples)."""
    import itertools
    n = len(values)
    if not 3 <= n <= 6:
        return None
    sds = sorted(statistics.stdev(c) for c in itertools.product(values, repeat=n))
    lo = sds[int((1 - level) / 2 * (len(sds) - 1))]
    hi = sds[int((1 + level) / 2 * (len(sds) - 1))]
    return _r(lo), _r(hi)
