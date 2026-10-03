"""Command line: `python -m noesis_lab.cli <command>`; see the Makefile for the usual entry points."""
from __future__ import annotations

import argparse
import json
import sys
import time

from .config import ROOT, baseline_config, get_profile, load_config, path_of


def cmd_session(a) -> int:
    from .orchestrator import SessionOpts, run_session
    out = run_session(SessionOpts(session=a.session, profile=a.profile, llm_mode=a.llm, force=a.force))
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


def cmd_lit_dryrun(a) -> int:
    """P0-4 acceptance: run the Literature agent on every fixture + queue candidate `--repeat` times
    and require identical verdicts. Nothing is recorded or shipped; live mode costs cents."""
    from . import stats
    from .literature import LiteratureAgent, Snapshot
    from .llm import LLM
    from .mock_llm import make_mock
    from .orchestrator import generate_fixture
    from .schemas import Fixture
    cfg, snap = load_config(), Snapshot(path_of("papers"), path_of("claims"))
    baseline = baseline_config(get_profile("full"))
    fixtures = {f["fixture_id"]: Fixture(**f) for f in json.loads(path_of("fixtures").read_text())["fixtures"]}
    for item in stats.candidate_queue(baseline, snap.claims.values(), []):
        fx = generate_fixture(item, baseline, snap)
        fixtures.setdefault(fx.fixture_id, fx)
    llm = LLM(a.llm, cfg["llm"], mock_fn=make_mock(fixtures) if a.llm == "mock" else None)
    agent, stable = LiteratureAgent(llm, snap), True
    for fid, fx in fixtures.items():
        runs = [agent.check(fx) for _ in range(a.repeat)]
        verdicts = {(r.verdict.value, r.claim_id) for r in runs}
        stable &= len(verdicts) == 1
        print(f"{fid:34s} {'STABLE ' if len(verdicts) == 1 else 'UNSTABLE'} "
              + " | ".join(f"{v} {c}" for v, c in sorted(verdicts)))
    print(f"calls={llm.n_calls} cost=${llm.cost_usd:.4f}  "
          + ("IDENTICAL verdicts across repeats" if stable else "VERDICTS DIFFER across repeats"))
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
    d = sub.add_parser("lit-dryrun", help="Literature agent only: verdict stability over repeats (cents)")
    d.add_argument("--llm", choices=["live", "mock"], default="live")
    d.add_argument("--repeat", type=int, default=2)
    d.set_defaults(fn=cmd_lit_dryrun)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
