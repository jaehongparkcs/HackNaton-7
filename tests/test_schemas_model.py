import itertools

import pytest
import torch
from pydantic import ValidationError

from noesis_lab.config import baseline_config, get_profile, load_config, path_of
from noesis_lab.schemas import (
    CriticPostOutput,
    CriticPreOutput,
    ExperimentConfig,
    LiteratureOutput,
    ScientistOutput,
    apply_delta,
)
from noesis_lab.testbed.harness import load_dataset, train_one
from noesis_lab.testbed.model import GPT


def test_delta_validation():
    b = ExperimentConfig()
    assert apply_delta(b, {"norm": "rmsnorm"}).norm == "rmsnorm"
    assert apply_delta(b, {"lr": "0.001"}).lr == 0.001          # LLM values arrive as strings
    for bad in ({"norm": "batchnorm"}, {"lr": "5"}, {"nonexistent": 1}, {"n_embd": 130}):
        with pytest.raises(ValidationError):
            apply_delta(b, bad)


def test_config_hash_is_stable():
    assert ExperimentConfig().config_hash() == ExperimentConfig().config_hash()
    assert ExperimentConfig().config_hash() != ExperimentConfig(norm="rmsnorm").config_hash()


def test_llm_schemas_cannot_carry_measurements_or_decisions():
    """The governing invariant, structurally: no LLM-facing field can hold a result or a label."""
    banned = {"val_loss", "delta", "label", "branch", "decision", "status", "p_value", "mean",
              "next_action", "tokens_seen"}
    for m in (LiteratureOutput, ScientistOutput, CriticPreOutput, CriticPostOutput):
        assert not (set(m.model_fields) & banned), m


@pytest.mark.parametrize("norm,pos,act,pe", list(itertools.product(
    ["layernorm", "rmsnorm"], ["pre", "post"], ["gelu", "relu2", "swiglu"], ["learned", "rope"])))
def test_every_option_combination_forward_backward(norm, pos, act, pe):
    c = ExperimentConfig(n_layer=2, n_head=2, n_embd=32, seq_len=16, batch_size=4, norm=norm,
                         norm_position=pos, activation=act, pos_encoding=pe, dropout=0.1)
    m = GPT(c, 65)
    x = torch.randint(0, 65, (4, 16))
    logits, loss = m(x, x)
    assert logits.shape == (4, 16, 65) and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())


def test_causality():
    c = ExperimentConfig(n_layer=2, n_head=2, n_embd=32, seq_len=16, pos_encoding="rope")
    m = GPT(c, 65).eval()
    x = torch.randint(0, 65, (1, 16))
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % 65
    a, _ = m(x)
    b, _ = m(y)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)       # earlier positions unaffected


@pytest.fixture(scope="module")
def data():
    return load_dataset(path_of("data"), load_config()["paths"]["data_sha256"])


def test_dataset_checksum_enforced():
    with pytest.raises(RuntimeError):
        load_dataset(path_of("data"), "0" * 64)


def test_cpu_profile_is_bit_deterministic_and_seed_sensitive(data):
    p = get_profile("smoke")
    c = baseline_config(p)
    dev = torch.device("cpu")
    a, b = train_one(c, 3, p, data, dev), train_one(c, 3, p, data, dev)
    assert a.status == "ok" and a.val_loss == b.val_loss and a.run_id == b.run_id
    assert train_one(c, 4, p, data, dev).val_loss != a.val_loss
    assert a.tokens_seen == p.max_steps * c.batch_size * c.seq_len


def test_failures_are_data_not_exceptions(data):
    p = get_profile("smoke")
    c = baseline_config(p).model_copy(update={"lr": float("nan")})
    r = train_one(c, 0, p, data, torch.device("cpu"))
    assert r.status == "failed" and r.error and r.val_loss is None
