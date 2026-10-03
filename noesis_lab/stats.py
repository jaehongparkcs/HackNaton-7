"""Statistician / Decider. Pure functions of stored runs: no I/O, no LLM, no clock, no randomness.

`rederive` calls these same functions on the shipped runs and demands an exact match.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

import numpy as np

from .schemas import (
    Analysis,
    Claim,
    Counterfactual,
    DerivedEvidence,
    NextAction,
    NoiseFloor,
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
    "flagged as seed disagreement) → next action = the next untested-setting candidate in the "
    "fixed literature-priority order.\n"
    "The LLM never selects the branch."
)


def _r(x: float) -> float:
    """Canonical rounding so recomputation is bit-identical across platforms."""
    return float(f"{x:.10g}")


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

    We replay that decision for every (candidate seed, baseline seed) pairing using runs we
    already have, so this costs no extra compute. If keep AND revert both occur, the seed alone
    decided the outcome.
    """
    cands = sorted(cand_runs, key=lambda r: r.seed)
    bases = sorted(base_runs, key=lambda r: r.seed)
    base_by_seed = {b.seed: b for b in bases}
    same = {c.seed: ("keep" if c.val_loss < base_by_seed[c.seed].val_loss else "revert")
            for c in cands if c.seed in base_by_seed}
    n = len(cands) * len(bases)
    keep = sum(1 for c in cands for b in bases if c.val_loss < b.val_loss)
    dep = 0 < keep < n
    if dep:
        summary = (f"A single-run keep/revert loop would KEEP this change in {keep} of {n} "
                   f"(candidate seed × baseline seed) pairings and REVERT it in {n - keep}: "
                   "the decision depended on seed luck, not on the change.")
    else:
        verdict = "keep" if keep == n else "revert"
        summary = (f"A single-run keep/revert loop would {verdict.upper()} this change in all {n} "
                   "pairings: the seed did not change the decision.")
    return Counterfactual(same_seed_decisions=same, n_pairings=n, n_keep=keep,
                          keep_rate=_r(keep / n if n else 0.0), decision_depends_on_seed=dep,
                          summary=summary)


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
                              candidate_tokens=c.tokens_seen, equal_token_delta=e))
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
    aid = "ana_" + sha256_hex(hypothesis_id + "|" + ",".join(sorted(
        [p.candidate_run_id for p in pairs] + [p.baseline_run_id for p in pairs])))[:10]
    return Analysis(
        analysis_id=aid, hypothesis_id=hypothesis_id,
        candidate_config_hash=next(iter(cands.values())).config_hash, noise=noise, pairs=pairs,
        mean_delta=_r(mean_delta), mean_improvement=_r(imp), improvement_in_noise_sd=_r(imp_sd),
        all_seeds_improve=all_imp, all_seeds_worse=all_worse,
        token_ratio=_r(mc / mb if mb else 1.0),
        mean_equal_token_delta=_r(statistics.fmean(eq_vals)) if eq_vals else None,
        counterfactual=counterfactual([cands[s] for s in seeds],
                                      [r for r in (all_base_runs or base_runs) if r.status == "ok"]),
        branch=branch, label=SCREENING_LABEL, seed_disagreement=seed_disagreement)


# --------------------------------------------------------------------------- next action
def next_action(a: Analysis, *, extra_seeds: Sequence[int], candidate_order: Sequence[str],
                already_considered: Sequence[str]) -> NextAction:
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
    nxt = next((f for f in candidate_order if f not in set(already_considered)), None)
    why = ("seeds disagree in sign" if a.seed_disagreement
           else f"mean improvement is {a.improvement_in_noise_sd:.2f}× the noise SD (within ±1×)")
    return NextAction(
        branch="no_improvement", action="next_candidate", detail={"fixture_id": nxt},
        rule_text=RULE_TEXT,
        rationale=(f"{why} → next untested-setting candidate in fixed order: "
                   f"{nxt if nxt else 'none left'}."))


# --------------------------------------------------------------------------- derived evidence
def relate_to_claim(a: Analysis, claim: Claim) -> DerivedEvidence:
    """How the measured branch relates to a cited claim. Pure table lookup; no LLM."""
    if claim.expected_outcome == "context":
        rel, note = "inconclusive", "cited claim makes no directional prediction for val_loss"
    elif a.branch == "harmful":
        rel, note = "contradicts", "candidate measured worse than baseline beyond the noise floor"
    elif a.branch == "promising":
        rel, note = "agrees", "candidate measured better than baseline beyond the noise floor"
    elif claim.expected_outcome == "no_worse":
        rel, note = "agrees", "no measurable difference beyond the noise floor (claim: comparable)"
    else:
        rel, note = "inconclusive", "claimed improvement not detected beyond the noise floor"
    return DerivedEvidence(evidence_id="ev_" + sha256_hex(a.analysis_id + claim.claim_id)[:10],
                           claim_id=claim.claim_id, analysis_id=a.analysis_id, relation=rel,
                           note=note)
