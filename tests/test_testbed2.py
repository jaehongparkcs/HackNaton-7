"""FIXES2 P1: the wider testbed. Every new building block runs on CPU (shape, finite loss,
gradients, determinism) and older bundles keep their config hashes."""
import pytest
import torch

from noesis_lab import stats
from noesis_lab.config import baseline_config, get_profile, load_config, path_of
from noesis_lab.schemas import ExperimentConfig, apply_delta
from noesis_lab.search.plan import CHANGE_QUERIES
from noesis_lab.search.scout import BLOCK_TERMS, BLOCKS
from noesis_lab.search.tighten import CHANGE_TERMS
from noesis_lab.testbed.harness import Lion, build_optimizer, load_dataset, train_one
from noesis_lab.testbed.model import GPT

SMALL = dict(n_layer=2, n_head=2, n_embd=32, seq_len=16, batch_size=4)
NEW = ["qk_norm=true", "weight_tying=true", "z_loss_coef=0.0001", "label_smoothing=0.1", "warmup_frac=0",
       "warmup_frac=0.02", "warmup_frac=0.1", "grad_clip=0", "init_scale=0.5", "init_scale=2.0", "optimizer=lion"]


def test_default_config_hash_and_dump_are_unchanged_by_the_new_fields():
    base = ExperimentConfig()
    assert not set(base.model_dump()) & {"qk_norm", "weight_tying", "z_loss_coef", "label_smoothing", "init_scale"}
    changed = apply_delta(base, {"qk_norm": "true"})
    assert changed.model_dump()["qk_norm"] is True and changed.config_hash() != base.config_hash()
    assert ExperimentConfig.model_validate(changed.model_dump()) == changed            # round-trips
    assert base.diff(changed) == {"qk_norm": True} and changed.diff(base) == {"qk_norm": False}


def test_recorded_run_ids_still_match_their_configs():
    """A stored run's config must hash to its stored config_hash, or replay could not find it."""
    import json

    from noesis_lab.config import ROOT
    for session in ("golden", "explore1"):
        f = ROOT / "results" / session / "runs.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            assert ExperimentConfig(**r["config"]).config_hash() == r["config_hash"]


@pytest.mark.parametrize("key", NEW)
def test_every_new_change_is_allowlisted_and_searchable(key):
    field = key.split("=")[0]
    assert key in stats.ALLOWED_CHANGES and field in stats.QUEUE_FIELDS and field in BLOCKS
    assert key in CHANGE_QUERIES and key in CHANGE_TERMS and BLOCK_TERMS[field]
    delta = stats.parse_delta_key(key)
    assert stats.valid_delta(delta) and stats.testability(delta) == 1.0
    cfg = apply_delta(ExperimentConfig(), delta)
    assert cfg != ExperimentConfig() and stats.delta_key(ExperimentConfig().diff(cfg)) == key   # typed value → same key


@pytest.mark.parametrize("key", NEW)
def test_every_new_option_forward_backward(key):
    c = apply_delta(ExperimentConfig(**SMALL), stats.parse_delta_key(key))
    torch.manual_seed(0)
    m = GPT(c, 65)
    x = torch.randint(0, 65, (4, 16))
    logits, val = m(x, x)
    loss = m.train_loss(x, x)
    assert logits.shape == (4, 16, 65) and torch.isfinite(val) and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    opt = build_optimizer(m, c)
    opt.step()
    assert torch.isfinite(m(x, x)[1])


def test_training_only_losses_do_not_change_the_reported_metric():
    x = torch.randint(0, 65, (4, 16))
    for delta in ({"label_smoothing": "0.1"}, {"z_loss_coef": "0.0001"}):
        torch.manual_seed(0)
        plain = GPT(ExperimentConfig(**SMALL), 65)
        torch.manual_seed(0)
        m = GPT(apply_delta(ExperimentConfig(**SMALL), delta), 65)
        assert torch.equal(m(x, x)[1], plain(x, x)[1])                 # val_loss: plain cross-entropy either way
        assert not torch.equal(m.train_loss(x, x), m(x, x)[1])         # the optimized loss carries the extra term
    assert torch.equal(plain.train_loss(x, x), plain(x, x)[1])         # defaults: identical to before


def test_weight_tying_shares_one_matrix_and_qk_norm_adds_one_scale():
    plain = GPT(ExperimentConfig(**SMALL), 65)
    tied = GPT(ExperimentConfig(**SMALL, weight_tying=True), 65)
    assert tied.head.weight is tied.tok.weight and tied.n_params() == plain.n_params() - 65 * 32
    qk = GPT(ExperimentConfig(**SMALL, qk_norm=True), 65)
    assert qk.n_params() == plain.n_params() + 2                       # one learned logit scale per layer
    assert all(b.attn.qk_scale is None for b in plain.blocks)


def test_init_scale_scales_the_initial_weights():
    def std(scale):
        torch.manual_seed(0)
        return GPT(ExperimentConfig(**SMALL, init_scale=scale), 65).tok.weight.std().item()
    assert std(2.0) == pytest.approx(2 * std(1.0), rel=1e-5) and std(0.5) == pytest.approx(0.5 * std(1.0), rel=1e-5)


def test_qk_norm_stays_causal():
    c = ExperimentConfig(**SMALL, qk_norm=True, pos_encoding="rope")
    m = GPT(c, 65).eval()
    x = torch.randint(0, 65, (1, 16))
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % 65
    assert torch.allclose(m(x)[0][:, :10], m(y)[0][:, :10], atol=1e-5)


def test_lion_moves_every_parameter_by_exactly_the_learning_rate():
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    opt = Lion([p], lr=0.1)
    (p * torch.tensor([1.0, -1.0, 2.0])).sum().backward()
    opt.step()
    assert torch.allclose(p, torch.tensor([0.9, -1.9, 2.9]))           # sign update: magnitude lr, direction −sign(grad)


@pytest.fixture(scope="module")
def data():
    return load_dataset(path_of("data"), load_config()["paths"]["data_sha256"])


@pytest.mark.parametrize("key", ["qk_norm=true", "weight_tying=true", "z_loss_coef=0.0001", "optimizer=lion", "grad_clip=0"])
def test_new_options_train_deterministically_on_the_cpu_profile(data, key):
    p = get_profile("smoke")
    c = apply_delta(baseline_config(p), stats.parse_delta_key(key))
    dev = torch.device("cpu")
    a, b = train_one(c, 1, p, data, dev), train_one(c, 1, p, data, dev)
    assert a.status == "ok" and a.val_loss == b.val_loss and a.val_loss < 4.2
    assert a.val_loss != train_one(baseline_config(p), 1, p, data, dev).val_loss
