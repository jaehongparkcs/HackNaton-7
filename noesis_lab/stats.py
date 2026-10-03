"""Statistician / Decider. Pure functions of stored runs: no I/O, no LLM, no clock, no randomness.

`rederive` calls these same functions on the shipped runs and demands an exact match.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Collection, Iterable, Sequence

import numpy as np

from .schemas import (
    Analysis,
    Claim,
    Counterfactual,
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


def _r(x: float) -> float:
    """Canonical rounding so recomputation is bit-identical across platforms."""
    return float(f"{x:.10g}")


# --------------------------------------------------------------------------- prior-art verdict
def prior_art_verdict(same_comparison: bool, covers_our_setting: bool | None) -> PriorArtVerdictKind:
    """Pure function of one LLM judgment (same comparison?) and one human-curated flag (covers our
    setting?). The LLM never chooses between known / setting-untested."""
    if not same_comparison:
        return PriorArtVerdictKind.not_found
    if covers_our_setting:
        return PriorArtVerdictKind.known
    return PriorArtVerdictKind.setting_untested


# --------------------------------------------------------------------------- candidate queue
# Single-field changes the lab may propose. Pure retunes (lr, batch size) are excluded.
QUEUE_FIELDS = ("norm", "norm_position", "activation", "pos_encoding", "schedule", "optimizer", "dropout")
OUTCOME_RANK = {"improves": 0, "no_worse": 1, "context": 2}


def candidate_queue(baseline: ExperimentConfig, claims: Iterable[Claim],
                    tested: Collection[str]) -> list[QueueItem]:
    """Every allowlisted single-field change from the baseline that at least one claim maps to,
    in deterministic priority order; no LLM, no I/O.

    Priority: (a) best expected outcome among supporting claims (improves < no_worse < context),
    (b) more supporting claims first, (c) alphabetical by field=value.
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
        items.append(QueueItem(key=key, field=field, value=value, priority=[rank, -len(ids), key],
                               supporting_claim_ids=ids))
    return sorted(items, key=lambda i: tuple(i.priority))


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
            promising_sd: float = 1.0, harmful_sd: float = 1.0) -> Analysis:
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
    if mean_delta > harmful_sd * sd and mean_delta > 0:
        branch = "harmful"
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
def next_action(a: Analysis, *, extra_seeds: Sequence[int], queue: Sequence[QueueItem]) -> NextAction:
    if a.branch == "promising":
        return NextAction(
            branch="promising", action="extra_seeds", detail={"seeds": list(extra_seeds)},
            rule_text=RULE_TEXT,
            rationale=(f"mean improvement is {a.improvement_in_noise_sd:.2f}× the noise SD and all "
                       f"{len(a.pairs)} seeds improve → run extra seeds for the same change."))
    if a.branch == "harmful":
        return NextAction(
            branch="harmful", action="literature_check", detail={},
            rule_text=RULE_TEXT,
            rationale=(f"candidate is worse by {-a.improvement_in_noise_sd:.2f}× the noise SD → "
                       "record a contradiction and ask the Literature Agent whether the snapshot "
                       "already reports this."))
    nxt = queue[0].key if queue else None
    why = ("seeds disagree in sign" if a.seed_disagreement
           else f"mean improvement is {a.improvement_in_noise_sd:.2f}× the noise SD (within ±1×)")
    return NextAction(
        branch="no_improvement", action="next_candidate", detail={"candidate": nxt},
        rule_text=RULE_TEXT,
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
        rel = "contradicts" if cov else "not_reproduced_in_our_setting"
        note = "candidate measured worse than baseline beyond the noise floor"
    elif a.branch == "promising":
        rel = "agrees" if cov else "consistent_in_our_setting"
        note = "candidate measured better than baseline beyond the noise floor"
    elif claim.expected_outcome == "no_worse":
        rel = "agrees" if cov else "consistent_in_our_setting"
        note = "no measurable difference beyond the noise floor (claim: comparable)"
    else:
        rel, note = "inconclusive", "claimed improvement not detected beyond the noise floor"
    if claim.claims_speed and a.throughput_ratio is not None and a.throughput_ratio < 0.95:
        note += "; partial: speed claim not reproduced (implementation-dependent)"
    return DerivedEvidence(evidence_id="ev_" + sha256_hex(a.analysis_id + claim.claim_id)[:10],
                           claim_id=claim.claim_id, analysis_id=a.analysis_id, relation=rel,
                           note=note)
