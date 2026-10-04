"""The schedule must scale each param group's own base lr. Until 2026-10-04 it reset every group to
c.lr × schedule, so Lion (built with lr / 5) trained at AdamW's learning rate, and lr_mult was dead."""
import types

import pytest
import torch

from noesis_lab.config import ROOT, baseline_config, get_profile, load_config, path_of
from noesis_lab.orchestrator import Session
from noesis_lab.schemas import ExperimentConfig
from noesis_lab.testbed import harness

DATA = harness.load_dataset(path_of("data"), load_config()["paths"]["data_sha256"])


def _lrs_during_training(c: ExperimentConfig, monkeypatch) -> list[tuple[float, list[float]]]:
    seen, real = [], harness.apply_schedule

    def spy(opt, progress, cfg):
        real(opt, progress, cfg)
        seen.append((progress, [g["lr"] for g in opt.param_groups]))
    monkeypatch.setattr(harness, "apply_schedule", spy)
    prof = get_profile("smoke")
    r = harness.train_one(c, 0, prof, DATA, torch.device("cpu"))
    assert r.status == "ok" and len(seen) == prof.max_steps
    return seen


def test_lion_trains_at_lr_mult_times_the_schedule_at_every_step(monkeypatch):
    c = baseline_config(get_profile("smoke")).model_copy(update={"optimizer": "lion", "lr_mult": 0.2, "wd_mult": 5.0})
    for progress, lrs in _lrs_during_training(c, monkeypatch):
        want = c.lr * 0.2 * harness.lr_multiplier(progress, c)
        assert lrs == [pytest.approx(want, rel=1e-12)] * 2
        if harness.lr_multiplier(progress, c) > 0:
            assert lrs[0] != pytest.approx(c.lr * harness.lr_multiplier(progress, c))   # the old bug: AdamW's lr


def test_adamw_schedule_is_unchanged(monkeypatch):
    c = baseline_config(get_profile("smoke"))
    for progress, lrs in _lrs_during_training(c, monkeypatch):
        assert lrs == [c.lr * harness.lr_multiplier(progress, c)] * 2         # bit-identical to the old formula


def test_baseline_config_hashes_of_recorded_bundles_are_unchanged():
    import json
    for s in ("golden", "explore2"):
        runs = [json.loads(x) for x in (ROOT / "results" / s / "runs.jsonl").read_text().splitlines() if x]
        base = baseline_config(get_profile("full"))
        assert base.config_hash() in {r["config_hash"] for r in runs}
        lion = [r for r in runs if r["config"].get("optimizer") == "lion"]
        for r in lion:                                                       # recorded Lion: no multipliers
            assert "lr_mult" not in r["config"] and ExperimentConfig(**r["config"]).config_hash() == r["config_hash"]


def _fake_session(recipes=None, defaults=True):
    cfg = load_config() if defaults else {k: v for k, v in load_config().items() if k != "method_defaults"}
    s = types.SimpleNamespace(snap=types.SimpleNamespace(recipes=recipes or {}), cfg=cfg)
    s._method_scaling = lambda m: Session._method_scaling(s, m)
    return s


def test_scaling_comes_from_a_recipe_or_a_labeled_stated_default_never_both():
    base = ExperimentConfig()
    cfg, rc = Session._with_recipe(_fake_session(), base, {"optimizer": "lion"})
    assert (cfg.lr_mult, cfg.wd_mult) == (0.2, 5.0) and rc["kind"] == "stated_default" and "2302.06675" in rc["source"]
    paper = {"optimizer=lion": {"lr_mult": 0.1, "wd_mult": 10.0, "paper_id": "2302.06675"}}
    cfg, rc = Session._with_recipe(_fake_session(paper), base, {"optimizer": "lion"})
    assert (cfg.lr_mult, cfg.wd_mult) == (0.1, 10.0) and rc["kind"] == "full_text_recipe"   # replaces, never stacks
    cfg2, rc2 = Session._with_recipe(_fake_session(), cfg, {"optimizer": "sgd_momentum"})
    assert (cfg2.lr_mult, cfg2.wd_mult) == (1.0, 1.0) and rc2 is None
    old = _fake_session(defaults=False)                                      # a bundle recorded before the fix
    assert Session._with_recipe(old, base, {"optimizer": "lion"}) == (base, None)
    assert Session._with_recipe(_fake_session(), base, {"norm": "rmsnorm"}) == (base, None)
