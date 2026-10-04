"""Harness-owned: data, split, timer, metric, seeding, environment capture.

Agents never touch this file. Contract:
  * The timer starts after model construction + a throw-away prewarm, and excludes eval.
  * val_loss = mean cross-entropy (nats/char) on a FIXED set of validation windows
    (offsets drawn from a generator with a constant seed), identical for every run.
  * Same seed -> same init and same data order. MPS kernels are not guaranteed bit-deterministic;
    the cpu `deterministic` profile is (tested).
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ..schemas import CurvePoint, ExperimentConfig, Profile, RunResult, sha256_hex
from .model import GPT

EVAL_SEED = 12345


# ------------------------------------------------------------------------------ data
@dataclass
class Dataset:
    train: torch.Tensor
    val: torch.Tensor
    vocab_size: int
    sha256: str


def load_dataset(path: Path, expected_sha256: str | None = None) -> Dataset:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(f"dataset checksum mismatch: {digest} != {expected_sha256}")
    text = raw.decode("utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(ids))
    return Dataset(train=ids[:n], val=ids[n:], vocab_size=len(chars), sha256=digest)


# ------------------------------------------------------------------------------ device/seed
def resolve_device(pref: str) -> torch.device:
    if pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(pref)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def seed_everything(seed: int, deterministic: bool) -> None:
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)


# ------------------------------------------------------------------------------ schedule/opt
def lr_multiplier(progress: float, c: ExperimentConfig) -> float:
    progress = min(max(progress, 0.0), 1.0)
    if c.warmup_frac > 0 and progress < c.warmup_frac:
        return (progress + 1e-9) / c.warmup_frac
    if c.schedule == "constant":
        return 1.0
    decay = (progress - c.warmup_frac) / max(1e-9, 1.0 - c.warmup_frac)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(decay, 1.0)))   # cosine to 10%


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al., 2023): update = sign(β1·m + (1−β1)·g), decoupled weight decay,
    m ← β2·m + (1−β2)·g. Run at the baseline's learning rate (no retune: a retune is a confound)."""

    def __init__(self, params, lr: float, betas=(0.9, 0.99), weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                update = (m * b1 + p.grad * (1 - b1)).sign_()
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)


def build_optimizer(model: GPT, c: ExperimentConfig) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    recipe = (c.lr_mult, c.wd_mult) != (1.0, 1.0)      # a full-text recipe (DEEP_READ §4.2) replaces defaults
    lr, wd = c.lr * c.lr_mult, c.weight_decay * c.wd_mult
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]
    if c.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
    if c.optimizer == "lion" and recipe:
        return Lion(groups, lr=lr)
    if c.optimizer == "lion":
        # Lion's sign update takes a smaller step and a larger decay than Adam (Chen et al., 2023,
        # recommend lr/3-10 and wd x3-10). We fix lr/5, wd x5, so a "harmful" result is the method's,
        # not an artifact of reusing Adam's learning rate. The ratio is held across every Lion run.
        lion_groups = [{"params": decay, "weight_decay": c.weight_decay * 5},
                       {"params": no_decay, "weight_decay": 0.0}]
        return Lion(lion_groups, lr=c.lr / 5)
    return torch.optim.SGD(groups, lr=lr, momentum=0.9)


# ------------------------------------------------------------------------------ env
def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                              cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def environment_fingerprint(device: torch.device, profile: Profile, data_sha: str) -> dict:
    root = Path(__file__).resolve().parents[2]
    lock = root / "uv.lock"
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else "",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "os": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
        "profile": profile.name,
        "profile_hash": profile.profile_hash(),
        "data_sha256": data_sha,
    }


# ------------------------------------------------------------------------------ eval/train
class Evaluator:
    def __init__(self, data: Dataset, c: ExperimentConfig, profile: Profile, device):
        g = torch.Generator().manual_seed(EVAL_SEED)
        n = profile.eval_batches * profile.eval_batch_size
        offs = torch.randint(0, len(data.val) - c.seq_len - 1, (n,), generator=g)
        ar = torch.arange(c.seq_len)
        self.x = data.val[offs[:, None] + ar].to(device)
        self.y = data.val[offs[:, None] + ar + 1].to(device)
        self.bs = profile.eval_batch_size

    @torch.no_grad()
    def __call__(self, model: GPT) -> float:
        model.eval()
        tot, cnt = 0.0, 0
        for i in range(0, len(self.x), self.bs):
            _, loss = model(self.x[i:i + self.bs], self.y[i:i + self.bs])
            tot += float(loss.item()) * self.x[i:i + self.bs].size(0)
            cnt += self.x[i:i + self.bs].size(0)
        model.train()
        return tot / cnt


def make_run_id(config_hash: str, seed: int, profile_hash: str) -> str:
    return "run_" + sha256_hex(f"{config_hash}|{seed}|{profile_hash}")[:10]


def train_one(c: ExperimentConfig, seed: int, profile: Profile, data: Dataset,
              device: torch.device, run_order: int = 0) -> RunResult:
    """Run one config on one seed. Never raises for training failures: they are data."""
    chash, phash = c.config_hash(), profile.profile_hash()
    base = dict(run_id=make_run_id(chash, seed, phash), config_hash=chash, config=c,
                profile_hash=phash, seed=seed, run_order=run_order,
                started_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                env=environment_fingerprint(device, profile, data.sha256))
    try:
        seed_everything(seed, profile.deterministic)
        model = GPT(c, data.vocab_size).to(device)
        n_params = model.n_params()
        evaluate = Evaluator(data, c, profile, device)
        opt = build_optimizer(model, c)
        g = torch.Generator().manual_seed(seed)                 # data order
        ar = torch.arange(c.seq_len)

        def batch():
            offs = torch.randint(0, len(data.train) - c.seq_len - 1, (c.batch_size,), generator=g)
            idx = offs[:, None] + ar
            return data.train[idx].to(device), data.train[idx + 1].to(device)

        # prewarm on a throw-away copy so kernel compilation is outside the timed region
        if profile.prewarm_steps:
            warm_model, warm_opt = copy.deepcopy(model), None
            warm_opt = build_optimizer(warm_model, c)
            gw = torch.Generator().manual_seed(seed + 7919)
            for _ in range(profile.prewarm_steps):
                offs = torch.randint(0, len(data.train) - c.seq_len - 1, (c.batch_size,), generator=gw)
                x, y = data.train[offs[:, None] + ar].to(device), data.train[offs[:, None] + ar + 1].to(device)
                loss = warm_model.train_loss(x, y)
                loss.backward()
                warm_opt.step()
                warm_opt.zero_grad(set_to_none=True)
            _sync(device)
            del warm_model, warm_opt
        seed_everything(seed, profile.deterministic)             # dropout stream starts clean
        model.train()

        time_mode = profile.budget_mode == "time"
        budget = profile.train_seconds if time_mode else float(profile.max_steps)
        train_s, step, tokens, next_eval = 0.0, 0, 0, 0.0
        curve: list[CurvePoint] = []
        every = profile.eval_every_s if time_mode else float(profile.eval_every_steps)

        def log_point():
            curve.append(CurvePoint(train_s=round(train_s, 4), step=step, tokens_seen=tokens,
                                    val_loss=evaluate(model)))

        log_point()
        next_eval = every
        while (train_s if time_mode else step) < budget:
            progress = (train_s if time_mode else step) / budget
            for gr in opt.param_groups:
                gr["lr"] = c.lr * lr_multiplier(progress, c)
            x, y = batch()
            _sync(device)
            t0 = time.perf_counter()
            loss = model.train_loss(x, y)
            loss.backward()
            if c.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            lval = float(loss.item())                            # also syncs the device
            train_s += time.perf_counter() - t0
            step += 1
            tokens += c.batch_size * c.seq_len
            if not math.isfinite(lval):
                raise FloatingPointError(f"non-finite training loss at step {step}")
            if (train_s if time_mode else step) >= next_eval and (train_s if time_mode else step) < budget:
                log_point()                                      # eval time is NOT in train_s
                next_eval += every
        log_point()
        return RunResult(**base, status="ok", val_loss=curve[-1].val_loss, tokens_seen=tokens,
                         steps=step, train_seconds=round(train_s, 4), n_params=n_params, curve=curve)
    except Exception as e:  # noqa: BLE001 - failures are data (SPEC rule 5)
        return RunResult(**base, status="failed", error=f"{type(e).__name__}: {e}"[:2000])
    finally:
        if device.type == "mps":
            torch.mps.empty_cache()
