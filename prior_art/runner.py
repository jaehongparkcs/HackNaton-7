"""Runners: LiveRunner trains on the device; RecordedRunner serves stored results (replay).

Runs are cached by (config hash, seed, profile hash): baseline runs are reused as the paired
baseline for every candidate and are never re-run (BUILD_PLAN s5). Failures are returned as data.
"""
from __future__ import annotations

import json
from pathlib import Path

from .schemas import ExperimentConfig, Profile, RunResult
from .store import Store
from .testbed.harness import load_dataset, make_run_id, resolve_device, train_one


class RunBudgetExceeded(RuntimeError):
    pass


class BaseRunner:
    def __init__(self, store: Store, profile: Profile):
        self.store, self.profile = store, profile
        self.cache: dict[str, RunResult] = {}

    def get(self, cfg: ExperimentConfig, seed: int) -> RunResult:
        rid = make_run_id(cfg.config_hash(), seed, self.profile.profile_hash())
        if rid not in self.cache:
            r = self._produce(cfg, seed, rid)
            self.cache[rid] = r
            self.store.put_run(r)
            self._persist(r)
        return self.cache[rid]

    def _produce(self, cfg: ExperimentConfig, seed: int, rid: str) -> RunResult:
        raise NotImplementedError

    def _persist(self, r: RunResult) -> None:
        pass

    @property
    def n_executed(self) -> int:
        return len(self.cache)


class LiveRunner(BaseRunner):
    def __init__(self, store: Store, profile: Profile, data_path: Path, data_sha: str,
                 runs_path: Path, max_runs: int):
        super().__init__(store, profile)
        self.data = load_dataset(data_path, data_sha)
        self.device = resolve_device(profile.device)
        self.runs_path, self.max_runs = runs_path, max_runs
        runs_path.parent.mkdir(parents=True, exist_ok=True)

    def _produce(self, cfg, seed, rid):
        if len(self.cache) >= self.max_runs:
            raise RunBudgetExceeded(f"run budget {self.max_runs} exhausted")
        return train_one(cfg, seed, self.profile, self.data, self.device, run_order=len(self.cache) + 1)

    def _persist(self, r):
        with self.runs_path.open("a") as f:
            f.write(r.model_dump_json() + "\n")


class RecordedRunner(BaseRunner):
    def __init__(self, store: Store, profile: Profile, runs_path: Path):
        super().__init__(store, profile)
        self.recorded: dict[str, RunResult] = {}
        for line in Path(runs_path).read_text().splitlines():
            if line.strip():
                r = RunResult(**json.loads(line))
                self.recorded[r.run_id] = r

    def _produce(self, cfg, seed, rid):
        if rid not in self.recorded:
            raise KeyError(f"replay: run {rid} (config {cfg.config_hash()}, seed {seed}) "
                           "is not in the recorded bundle")
        return self.recorded[rid]
