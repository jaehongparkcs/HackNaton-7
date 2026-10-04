"""FINAL_FIXES: a bundle holds only what its manifest lists, and the polish / speed changes keep
old bundles replaying unchanged."""
import json

from noesis_lab import dashboard
from noesis_lab.bundle import verify_manifest, write_manifest
from noesis_lab.config import ROOT
from noesis_lab.store import Store


# --------------------------------------------------------------------------- 0. unexpected files
def _bundle(tmp_path):
    d = tmp_path / "b"
    (d / "corpus" / "raw").mkdir(parents=True)
    (d / "state.json").write_text("{}\n")
    (d / "corpus" / "raw" / "001_arxiv.xml").write_text("<feed/>")
    write_manifest(d)
    return d


def test_verify_passes_on_a_clean_bundle(tmp_path):
    assert verify_manifest(_bundle(tmp_path)) == []


def test_verify_fails_on_files_the_manifest_does_not_list(tmp_path):
    d = _bundle(tmp_path)
    (d / "corpus" / "raw" / "001_arxiv 2.xml").write_text("<feed/>")      # an iCloud conflict copy
    (d / "state 2.json").write_text("{}\n")
    problems = verify_manifest(d)
    assert problems == ["corpus/raw/001_arxiv 2.xml: not in MANIFEST.json (unexpected file)",
                        "state 2.json: not in MANIFEST.json (unexpected file)"]


def test_verify_ignores_finder_metadata_but_not_the_manifest_contents(tmp_path):
    d = _bundle(tmp_path)
    (d / ".DS_Store").write_bytes(b"\0")
    assert verify_manifest(d) == []
    files = json.loads((d / "MANIFEST.json").read_text())["files"]
    assert "MANIFEST.json" not in files and set(files) == {"state.json", "corpus/raw/001_arxiv.xml"}


# --------------------------------------------------------------------------- A1-A4 presentation
EXPLORE2 = ROOT / "results" / "explore2" / "notebook.sqlite"


def test_dashboard_opens_on_explore2_then_golden_and_labels_the_rest():
    keys = ["results/explore1/notebook.sqlite", "results/golden/notebook.sqlite", "results/smoke/notebook.sqlite",
            "work/replay-explore2/notebook.sqlite", "results/explore2/notebook.sqlite", "results/mps-mock/notebook.sqlite"]
    order = [dashboard.session_name(k) for k in dashboard.order_notebooks(keys)]
    assert order[:3] == ["explore2", "replay-explore2", "golden"]
    assert "superseded (scout schema bug, rule v1)" in dashboard.notebook_label(keys[0])
    for k in ("results/smoke/notebook.sqlite", "results/mps-mock/notebook.sqlite"):
        assert "pipeline checks: not results" in dashboard.notebook_label(k)


def test_a_sealed_bundle_states_its_limitation_instead_of_asking_to_fix_it():
    reason = "thin mechanism graph: only 2 node(s) (want >= 3); gaps will be sparse. Fix before recording."
    assert dashboard.degraded_note(reason, sealed=False) == f"Search degraded: {reason}"
    sealed = dashboard.degraded_note(reason, sealed=True)
    assert sealed.startswith("Recorded with this limitation: thin mechanism graph")
    assert "Fix before recording" not in sealed and "Search degraded" not in sealed


def _explore2():
    s = Store(EXPLORE2, readonly=True)
    d = dict(hyps=s.hypotheses(), analyses=s.analyses(), decisions=s.decisions(), evidence=s.derived_evidence(),
             claims={c["claim_id"]: c for c in s.claims()}, baseline=s.get_meta("baseline_config"))
    s.close()
    return d


def test_a_change_on_an_incumbent_is_credited_with_its_own_effect_only():
    d = _explore2()
    ana = {a["analysis_id"]: a for a in d["analyses"]}
    h = next(x for x in d["hyps"] if x["hypothesis_id"] == "hyp_gap_activation_relu2_norm_rmsnorm_i1")
    cr = dashboard.credit(h, ana, d["baseline"])
    assert cr["text"] == ("activation=relu2 + norm=rmsnorm on top of pos_encoding=rope: no improvement vs incumbent "
                          "(-0.67 SD). vs original +1.46 SD, which includes pos_encoding=rope.")
    assert dashboard.analysis_role(h, h["headline_analysis_id"], d["baseline"]).startswith("vs original baseline (cumulative")
    assert dashboard.analysis_role(h, h["latest_analysis_id"], d["baseline"]).startswith("vs incumbent pos_encoding=rope")
    rope = next(x for x in d["hyps"] if x["hypothesis_id"] == "hyp_gap_pos_encoding_rope")   # tested on the baseline
    assert dashboard.credit(rope, ana, d["baseline"]) is None
    assert dashboard.analysis_role(rope, rope["latest_analysis_id"], d["baseline"]) == ""


def test_tables_never_show_the_cumulative_gain_as_the_changes_result():
    from noesis_lab.evidence_chain import evidence_grid
    from noesis_lab.gap_views import our_results, prediction_rows
    d = _explore2()
    ana = {a["analysis_id"]: a for a in d["analyses"]}
    row = next(r for r in evidence_grid(d) if r["candidate"].startswith("activation=relu2 + norm=rmsnorm")
               and "incumbent" in r["candidate"])
    assert row["measured"].startswith("no_improvement (-0.67") and row["measured"].endswith("vs incumbent")
    assert row["vs original baseline (cumulative, includes the incumbent)"].startswith("promising (+1.46")
    res = {r["delta_key"]: r for r in our_results(d["hyps"], ana, d["baseline"])}
    assert res["activation=relu2+norm=rmsnorm"]["branch"] == "no_improvement"
    assert res["activation=relu2+norm=rmsnorm"]["incumbent"] == "pos_encoding=rope"
    assert res["pos_encoding=rope"]["incumbent"] == ""
    pr = {r["hypothesis"]: r for r in prediction_rows(d["hyps"], ana, d["baseline"])}
    assert pr["activation=relu2+norm=rmsnorm"]["measured vs"] == "incumbent pos_encoding=rope"


def test_lion_hypotheses_carry_the_scaling_caveat():
    d = _explore2()
    lion = [h for h in d["hyps"] if dashboard.is_lion(h)]
    assert lion and all("optimizer=lion" in h["candidate_key"] for h in lion)
    assert "not a test of Lion" in dashboard.LION_CAVEAT


def test_dashboard_renders_the_explore2_exhibit():
    import os

    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    os.environ["NOESIS_NOTEBOOK"] = str(EXPLORE2)
    try:
        at.run()
    finally:
        os.environ.pop("NOESIS_NOTEBOOK", None)
    assert not at.exception, [e.value for e in at.exception]
    md = [m.value for m in at.markdown]
    assert any("no improvement vs incumbent (-0.67 SD). vs original +1.46 SD, which includes pos_encoding=rope" in m
               for m in md)
    assert any(i.value.startswith("Recorded with this limitation") for i in at.info)
    assert not any("Fix before recording" in w.value for w in at.warning)
    assert any("not a test of Lion" in c.value for c in at.caption)


# --------------------------------------------------------------------------- A5 combination predictions
from noesis_lab import gaps  # noqa: E402

from .test_gaps import DROP, MECH, RMS, ROPE, SWIGLU, claim  # noqa: E402


def _combo_graph():
    return [claim("a", SWIGLU, "improves", "T3", "none", mech=MECH),
            claim("b", ROPE, "improves", "T3", "partial", mech=MECH),
            claim("c", RMS, "worse", "T3", "none", mech=MECH),
            claim("d", DROP, "no_worse", "T3", "none", mech=MECH)]


def test_combination_sign_is_the_sum_of_the_components_literature_signs():
    g = gaps.build_graph(_combo_graph())
    cd = gaps.combination_direction
    assert cd(g, ["activation=swiglu", "pos_encoding=rope"]) == "+"            # both + -> +
    assert cd(g, ["activation=swiglu", "norm=rmsnorm"]) == ""                  # mixed -> no prediction
    assert cd(g, ["activation=swiglu", "dropout=0.1"]) == "+"                  # + and 0 -> +
    assert cd(g, ["dropout=0.1", "norm=rmsnorm"]) == "-"
    assert cd(g, ["activation=swiglu", "optimizer=lion"]) == ""                # no claim -> no prediction


def test_our_own_results_never_set_a_components_sign():
    ours = gaps.derived_claim({"norm": "rmsnorm"}, "promising", tier="D-confirmed")
    g = gaps.build_graph([*_combo_graph(), ours, ours.model_copy(update={"claim_id": ours.claim_id + "b"})])
    assert gaps.literature_sign(g, "norm=rmsnorm") == "-"


def test_combination_predictions_only_under_prediction_version_2():
    cs = _combo_graph()
    old = [x for x in gaps.find_gaps(cs)[1] if x.novelty_type == "combination"]
    new = {x.gap_id: x for x in gaps.find_gaps(cs, predict_combinations=True)[1] if x.novelty_type == "combination"}
    assert old and all(x.predicted_direction == "" for x in old)              # how explore2 was recorded
    assert {x.gap_id for x in old} == set(new)                                # same gaps, same scores and order
    assert all(x.score == new[x.gap_id].score for x in old)
    assert new["link:activation=swiglu+pos_encoding=rope"].predicted_direction == "+"
    assert gaps.COMBINATION_RULE_TEXT.count("→") >= 3


def test_config_turns_on_prediction_version_2_and_old_bundles_stay_on_1():
    from noesis_lab.config import bundle_config, load_config, protocol
    assert protocol(load_config())["prediction_version"] == 2
    for s in ("golden", "explore1", "explore2"):
        assert protocol(bundle_config(ROOT / "results" / s))["prediction_version"] == 1


# --------------------------------------------------------------------------- A6 timestamps
def test_events_and_decisions_carry_a_timestamp_outside_the_digest(tmp_path):
    a, b = Store(tmp_path / "a.sqlite"), Store(tmp_path / "b.sqlite")
    for s in (a, b):
        s.add_event("llm_call", {"role": "x"})
        s.add_decision("noise_floor", "", {"sd": 0.01})
    b.db.execute("UPDATE decisions SET payload=json_set(payload, '$.ts', '1999-01-01T00:00:00.000+00:00')")
    assert a.events()[0]["ts"] and a.decisions()[0]["ts"].endswith("+00:00")
    assert a.state_digest() == b.state_digest()                               # wall time never enters the digest
    b.db.execute("UPDATE decisions SET payload=json_set(payload, '$.sd', 0.02)")
    assert a.state_digest() != b.state_digest()                               # content still does


def test_rows_recorded_before_timestamps_keep_their_digest(tmp_path):
    s = Store(tmp_path / "s.sqlite")
    s.add_decision("noise_floor", "", {"sd": 0.01})
    with_ts = s.state_digest()
    p = s.decisions()[0]
    old = json.dumps({k: v for k, v in p.items() if k not in ("ts", "kind", "hypothesis_id")}, sort_keys=True)
    s.db.execute("UPDATE decisions SET payload=?", (old,))                    # as Store wrote rows before A6
    assert s.state_digest() == with_ts


def test_timeline_flags_a_long_silence():
    ev = [{"event_id": "evt_0001", "role": "critic_pre", "ts": "2026-10-03T18:00:00.000+00:00"}]
    dec = [{"decision_id": "dec_001", "kind": "confirmation", "ts": "2026-10-03T18:00:30.000+00:00"},
           {"decision_id": "dec_002", "kind": "next_action", "ts": "2026-10-03T18:57:30.000+00:00"}]
    runs = [{"run_id": "r1", "seed": 0, "train_seconds": 36.0, "started_at": "2026-10-03T18:00:10+00:00"}]
    rows = dashboard.timeline_rows(ev, dec, runs)
    assert [r["kind"] for r in rows] == ["LLM call", "run", "decision", "decision"]
    assert rows[-1]["gap before (s)"] == 3420.0 and rows[-1]["stall"] and not any(r["stall"] for r in rows[:-1])


# --------------------------------------------------------------------------- B1 / B3 / B6
from pydantic import BaseModel  # noqa: E402

from noesis_lab.llm import LLM  # noqa: E402


class _Out(BaseModel):
    reading: str


def _mock(role, system, user, schema_cls):
    return _Out(reading="ok")


def test_effort_is_per_role_and_recorded_in_the_key(tmp_path):
    cfg = {"model": "m", "effort": "medium", "max_tokens": 10, "price_per_mtok": {"input": 0, "output": 0},
           "effort_by_role": {"literature": "low"}}
    rec = tmp_path / "llm.jsonl"
    llm = LLM("mock", cfg, recordings_path=rec, mock_fn=_mock)
    llm.call("literature", "s", "u", _Out)
    llm.call("scientist_gap", "s", "u", _Out)
    efforts = [json.loads(line)["effort"] for line in rec.read_text().splitlines()]
    assert efforts == ["low", "medium"]
    LLM("replay", cfg, recordings_path=rec).call("literature", "s", "u", _Out)       # same key on replay
    old = {k: v for k, v in cfg.items() if k != "effort_by_role"}                    # a bundle recorded before B3
    assert LLM("mock", old, mock_fn=_mock).effort_for("literature") == "medium"


def test_live_client_has_an_explicit_timeout(monkeypatch):
    import types

    import anthropic
    seen = {}

    class Fake:
        def __init__(self, **kw):
            seen.update(kw)
            resp = types.SimpleNamespace(stop_reason="end_turn", model="m", content=[types.SimpleNamespace(
                type="text", text='{"reading": "ok"}')], usage=types.SimpleNamespace(input_tokens=1, output_tokens=1))
            self.messages = types.SimpleNamespace(create=lambda **k: seen.setdefault("req", k) and resp)

    monkeypatch.setattr(anthropic, "Anthropic", Fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = {"model": "m", "effort": "medium", "max_tokens": 10, "price_per_mtok": {"input": 0, "output": 0},
           "server_side_fallback": False, "timeout_s": 120, "max_retries": 2, "effort_by_role": {"critic_post": "low"}}
    out, _ = LLM("live", cfg).call("critic_post", "s", "u", _Out)
    assert out.reading == "ok" and seen["timeout"] == 120.0 and seen["max_retries"] == 2
    assert seen["req"]["output_config"]["effort"] == "low"


def test_config_sets_cheap_routine_roles_and_keeps_the_scientist_at_medium():
    from noesis_lab.config import load_config
    llm = load_config()["llm"]
    for role in ("literature", "critic_post", "extract"):
        assert llm["effort_by_role"][role] == "low"
    assert "scientist_gap" not in llm["effort_by_role"] and "critic_pre" not in llm["effort_by_role"]
    assert llm["effort"] == "medium" and llm["timeout_s"] == 120 and llm["max_retries"] == 2


def test_critic_pre_cap_changes_the_prompt_only_when_configured():
    from noesis_lab.agents import Critic
    plain, capped = Critic(None), Critic(None, {"confounds": 5, "required_controls": 4})
    assert plain.pre_system == (ROOT / "prompts" / "critic_pre.md").read_text()      # recorded bundles replay
    assert capped.pre_system.startswith(plain.pre_system)
    assert "at most 5 confounds" in capped.pre_system and "at most 4 required controls" in capped.pre_system


def test_rehearse_profile_is_short_and_shrinks_the_loop():
    from noesis_lab.config import get_profile, load_config, with_profile_overrides
    p = get_profile("rehearse")
    assert p.max_steps == 500 and p.budget_mode == "steps"
    cfg = with_profile_overrides(load_config(), "rehearse")
    assert cfg["search_loop"]["explore_k"] == 3 and cfg["search_loop"]["cycles"] == 1
    assert cfg["search_loop"]["finalists_per_cycle"] == load_config()["search_loop"]["finalists_per_cycle"]
    assert with_profile_overrides(load_config(), "full") is load_config()
    assert "pipeline checks: not results" in dashboard.notebook_label("results/rehearse/notebook.sqlite") or \
        "rehearsal profile: not results" in dashboard.notebook_label("results/rehearse/notebook.sqlite")


def test_live_make_targets_run_under_caffeinate():
    mk = (ROOT / "Makefile").read_text()
    assert "caffeinate -i" in mk
    for target in ("golden:", "record:", "search:", "rehearse:"):
        body = mk.split("\n" + target, 1)[1].split("\n\n", 1)[0]
        assert "$(LIVE)" in body, target


# --------------------------------------------------------------------------- B2 concurrency
def _loop_run(name, monkeypatch, workers):
    from .test_explore import CFG, explore_session
    from .test_loop import OPEN, _promising_dropout
    monkeypatch.setitem(CFG["llm"], "concurrency", workers)
    s = explore_session(name, monkeypatch, overrides=OPEN, effect=_promising_dropout)
    assert s.llm.workers == workers
    out = s.run()
    d = ROOT / "results" / name
    return out, (d / "recordings" / "llm.jsonl").read_text(), s.llm


def test_concurrent_preparation_records_exactly_what_a_sequential_run_records(session_name, monkeypatch, capsys):
    from noesis_lab.cli import main
    seq, seq_rec, _ = _loop_run(session_name, monkeypatch, 1)
    par, par_rec, llm = _loop_run(session_name, monkeypatch, 5)
    assert llm.n_speculative > 0                                 # the cycle's candidates were prefetched
    assert par["state_digest"] == seq["state_digest"]            # same events, decisions, runs, links
    assert par_rec == seq_rec                                    # same recordings, in the same order
    assert par["llm_calls"] == seq["llm_calls"]
    assert main(["replay", "--session", session_name]) == 0 and "REPLAY OK" in capsys.readouterr().out


def test_concurrent_extraction_keeps_batch_order(monkeypatch):
    import time

    from noesis_lab.schemas import RetrievedPaper
    from noesis_lab.search import extract as ex
    papers = [RetrievedPaper(paper_id=f"p{i}", title=f"t{i}", published="2020-01-01", url="u",
                             abstract=f"We find RMSNorm helps {i}.", abstract_sha256="x", source="arxiv")
              for i in range(3 * ex.BATCH)]
    from noesis_lab.mock_llm import make_mock
    base = make_mock({})

    def slow_first(role, system, user, schema):             # the first batch returns last
        if "### PAPER p0\n" in user:
            time.sleep(0.2)
        return base(role, system, user, schema)
    cfg = {"model": "m", "effort": "medium", "max_tokens": 10, "price_per_mtok": {"input": 0, "output": 0}}
    one = ex.extract(LLM("mock", cfg, mock_fn=slow_first), papers)
    many = ex.extract(LLM("mock", {**cfg, "concurrency": 3}, mock_fn=slow_first), papers)
    assert [c.claim_id for c in one[0]] == [c.claim_id for c in many[0]] and len(one[0]) == len(papers)
