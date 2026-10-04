"""Command line: `python -m noesis_lab.cli <command>`; see the Makefile for the usual entry points."""
from __future__ import annotations

import argparse
import json
import sys
import time

from .config import ROOT, baseline_config, get_profile, load_config, path_of


def cmd_session(a) -> int:
    from .orchestrator import SessionOpts, run_session
    out = run_session(SessionOpts(session=a.session, profile=a.profile, llm_mode=a.llm, force=a.force,
                                  niche=a.niche or None, corpus=a.corpus or None))
    print(json.dumps({k: v for k, v in out.items() if k != "counts"}, indent=2))
    if a.llm == "mock" or a.profile == "smoke":
        print("\nWARNING: mock-LLM / smoke-profile output. Proves the pipeline only; "
              "never present these numbers as results.", file=sys.stderr)
    return 0 if out["status"] == "complete" else 2


def cmd_replay(a) -> int:
    """R2: rebuild the whole session from recorded LLM responses + recorded runs; compare digests."""
    from .orchestrator import SessionOpts, run_session
    bundle = ROOT / load_config()["paths"]["results"] / a.session
    state = json.loads((bundle / "state.json").read_text())
    out = run_session(SessionOpts(session=a.session, profile=state["profile"], llm_mode="replay"))
    ok = out["state_digest"] == state["state_digest"]
    print(f"recorded digest: {state['state_digest']}\nreplayed digest: {out['state_digest']}")
    print("REPLAY " + ("OK: dashboard state is identical to the original session"
                       if ok else "MISMATCH"))
    print(f"replayed notebook: {out['out']}/notebook.sqlite   (view: make app SESSION={a.session})")
    return 0 if ok else 1


def cmd_rederive(a) -> int:
    from .rederive import rederive
    problems = rederive(ROOT / load_config()["paths"]["results"] / a.session)
    for p in problems:
        print("MISMATCH:", p)
    print("REDERIVE " + ("OK: every number reproduced exactly" if not problems else "FAILED"))
    return 1 if problems else 0


def cmd_verify(a) -> int:
    from .bundle import verify_manifest
    problems = verify_manifest(ROOT / load_config()["paths"]["results"] / a.session)
    for p in problems:
        print("PROBLEM:", p)
    print("MANIFEST " + ("OK" if not problems else "FAILED"))
    return 1 if problems else 0


def cmd_search(a) -> int:
    """Live literature search for a niche, frozen to a corpus folder. No training."""
    import datetime as dt
    import shutil
    from pathlib import Path

    from .llm import LLM
    from .mock_llm import make_mock
    from .search.corpus import build_corpus, load_niche
    cfg = load_config()
    niche = load_niche(a.niche)
    out = Path(a.out) if a.out else ROOT / cfg["paths"]["work"] / ("corpus-" + Path(a.niche).stem)
    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():          # keep raw/ so a re-run resumes from cache; refresh everything else
        if child.name != "raw":
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    llm = LLM(a.llm, cfg["llm"], recordings_path=out / "llm.jsonl",
              mock_fn=make_mock({}) if a.llm == "mock" else None,
              max_cost_usd=cfg["lit_search"]["max_llm_cost_usd"])
    meta = build_corpus(out, niche, llm, today=dt.date.today().isoformat())
    print(json.dumps({k: meta.get(k) for k in ("label", "mode", "papers", "claims", "tiers", "counts", "explore",
                                              "extraction", "validation", "openalex", "degraded",
                                              "attempted", "failed", "reused", "failure_fraction",
                                              "llm_calls", "llm_cost_usd")}, indent=2, default=str))
    if meta.get("mode") == "aborted_degraded":
        print(f"\nSEARCH ABORTED: {meta['label']}. The raw/ cache is kept; wait ~15-30 min and re-run the same "
              "command to resume (only the failed requests are re-fetched). Nothing was frozen.")
        return 1
    shared = (meta.get("explore") or {}).get("shared_mechanisms")
    if shared is not None:
        need = cfg["explore"]["min_shared_mechanisms"]
        print(f"\nmechanism nodes linked to 2+ runnable methods: {len(shared)} "
              + ("(OK)" if len(shared) >= need else f"(THIN: want >= {need}; fix before recording)"))
        for k, ms in shared.items():
            print(f"  {k}: {', '.join(ms)}")
    print(f"\nfrozen corpus: {out}\nuse it:  make golden CORPUS={out}")
    return 0


def cmd_lit_dryrun(a) -> int:
    """Gate stability: run the Literature agent on every fixture + queue candidate `--repeat` times.
    A row is STABLE if every repeat gives the same verdict and the same gate action (the cited
    claim may differ only then). Exits non-zero if any row is UNSTABLE, so a recording can stop.
    Nothing is recorded or shipped; live mode costs cents."""
    from . import gaps, stats
    from .literature import LiteratureAgent, Snapshot
    from .llm import LLM
    from .mock_llm import make_mock
    from .orchestrator import gap_fixture, generate_fixture
    from .schemas import Fixture
    cfg = load_config()
    curated = Snapshot(path_of("papers"), path_of("claims"))
    snap = Snapshot.from_corpus(a.corpus) if a.corpus else curated
    baseline = baseline_config(get_profile("full"))
    fixtures = {f["fixture_id"]: Fixture(**f) for f in json.loads(path_of("fixtures").read_text())["fixtures"]}
    contested: dict[str, tuple[str, ...]] = {}
    require: dict[str, dict] = {}
    frozen = sorted(snap.claims.values(), key=lambda c: c.claim_id)
    explore_mode = bool(snap.directions) and cfg.get("explore", {}).get("enabled")
    if explore_mode:                                   # the hypothesis engine's own queue
        fixtures.pop("fixture_b_rmsnorm", None)        # the engine generates RMSNorm (FIXES3 P0-3)
        graph, found = gaps.find_gaps(frozen)
        for item in gaps.gap_queue(found, graph, frozen, top_n=cfg["explore"]["top_gaps_for_hypotheses"]):
            fx = gap_fixture(item, next(g for g in found if g.gap_id == item.gap_ids[0]), baseline)
            fixtures.setdefault(fx.fixture_id, fx)
            require[fx.fixture_id] = dict(item.delta)
            if len(item.delta) == 1:
                contested[fx.fixture_id] = tuple(stats.contested_claim_ids(
                    [c for c in frozen if gaps.method_key(c) == item.key]))
    else:
        for item in stats.candidate_queue(baseline, snap.claims.values(), []):
            fx = generate_fixture(item, baseline, snap)
            fixtures.setdefault(fx.fixture_id, fx)
            require[fx.fixture_id] = stats.parse_delta_key(fx.candidate_key)
    llm = LLM(a.llm, cfg["llm"], mock_fn=make_mock(fixtures) if a.llm == "mock" else None)
    agent, stable = LiteratureAgent(llm, snap), True
    base_agent = LiteratureAgent(llm, curated) if a.corpus else None
    for fid, fx in fixtures.items():
        runs = [agent.check(fx, contested_ids=contested.get(fid, ()), require_change=require.get(fid))
                for _ in range(a.repeat)]
        actions = {r.gate for r in runs}
        verdicts = {r.verdict.value for r in runs}
        cited = sorted({r.claim_id or "-" for r in runs})
        stable &= len(actions) == 1            # the ACTION must be stable; the cited claim may differ
        tag = "STABLE  " if len(actions) == 1 else "UNSTABLE"
        line = (f"{fid:44s} {tag} actions: {', '.join(sorted(actions))}; verdicts: {', '.join(sorted(verdicts))}"
                + f"   cited: {', '.join(cited)}")
        if base_agent and fx.origin == "fixture":      # leakage check: did the larger corpus change it?
            before = base_agent.check(fx)
            if before.gate not in actions:              # the action changed: real leakage
                line += f"   LEAKAGE: curated-only action was {before.gate}. Report it and pick a new fixture."
            elif before.verdict.value not in verdicts:  # only the label changed: report, do not abort
                line += f"   LEAKAGE (label only): curated-only verdict was {before.verdict.value}; action unchanged."
        print(line)
    print(f"calls={llm.n_calls} cost=${llm.cost_usd:.4f}  "
          + ("every row STABLE" if stable else
             "GATE UNSTABLE: at least one row changed its gate action between repeats. Do not record; "
             "fix the gate (or the corpus) first."))
    return 0 if stable else 1


def cmd_timing(a) -> int:
    """Throughput check on this machine (the budget is steps, so this only reports speed)."""
    from .schemas import apply_delta
    from .testbed.harness import load_dataset, resolve_device, train_one
    prof = get_profile(a.profile).model_copy(update={"max_steps": a.steps, "eval_every_steps": a.steps})
    data = load_dataset(path_of("data"), load_config()["paths"]["data_sha256"])
    dev = resolve_device(prof.device)
    base = baseline_config(prof)
    print(f"device={dev} profile={a.profile} budget={a.steps} steps")
    tps = {}
    for name, delta in [("baseline", {}), ("rmsnorm", {"norm": "rmsnorm"})]:
        t = time.time()
        r = train_one(apply_delta(base, delta), 0, prof, data, dev)
        tps[name] = r.tokens_seen / max(r.train_seconds, 1e-9)
        print(f"{name:9s} status={r.status} val_loss={r.val_loss} steps={r.steps} "
              f"tokens={r.tokens_seen} ({tps[name]:,.0f} tok/s) "
              f"params={r.n_params} wall={time.time() - t:.1f}s {r.error}")
    ratio = tps["rmsnorm"] / tps["baseline"]
    print(f"RMSNorm / LayerNorm throughput = {ratio:.3f} "
          + ("(OK: >= 0.98)" if ratio >= 0.98 else "(slower than LayerNorm; report as an honest implementation cost)"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="noesis_lab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("session", help="run the golden path and write results/<session>")
    s.add_argument("--session", default="golden")
    s.add_argument("--profile", default="full")
    s.add_argument("--llm", choices=["live", "mock"], default="live")
    s.add_argument("--force", action="store_true")
    s.add_argument("--niche", default="", help="niche.yaml: run the live literature search first")
    s.add_argument("--corpus", default="", help="frozen corpus folder from `search` (no network)")
    s.set_defaults(fn=cmd_session)
    for name, fn, h in [("replay", cmd_replay, "R2: replay recorded LLM + runs, no key, CPU"),
                        ("rederive", cmd_rederive, "R1: recompute all stats from stored runs"),
                        ("verify", cmd_verify, "check bundle SHA256 manifest")]:
        p = sub.add_parser(name, help=h)
        p.add_argument("--session", default="golden")
        p.set_defaults(fn=fn)
    t = sub.add_parser("timing", help="measure throughput (RMSNorm vs LayerNorm) on this machine")
    t.add_argument("--profile", default="full")
    t.add_argument("--steps", type=int, default=300)
    t.set_defaults(fn=cmd_timing)
    se = sub.add_parser("search", help="live literature search for a niche; freeze the corpus; no training")
    se.add_argument("--niche", default="niche.yaml")
    se.add_argument("--out", default="")
    se.add_argument("--llm", choices=["live", "mock"], default="live")
    se.set_defaults(fn=cmd_search)
    d = sub.add_parser("lit-dryrun", help="Literature agent only: verdict stability over repeats (cents)")
    d.add_argument("--llm", choices=["live", "mock"], default="live")
    d.add_argument("--repeat", type=int, default=2)
    d.add_argument("--corpus", default="", help="frozen corpus folder; also runs the fixture leakage check")
    d.set_defaults(fn=cmd_lit_dryrun)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
