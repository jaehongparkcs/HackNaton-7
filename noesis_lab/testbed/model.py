"""Tiny char-level GPT. Every allowlisted option in ExperimentConfig is implemented here."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schemas import ExperimentConfig


class RMSNorm(nn.Module):
    """Zhang & Sennrich (2019): x / rms(x) * g, no re-centering and no bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def make_norm(kind: str, dim: int) -> nn.Module:
    return nn.LayerNorm(dim) if kind == "layernorm" else RMSNorm(dim)


def rope_tables(seq_len: int, head_dim: int, device, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv)  # (T, hd/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, hd). Rotate consecutive (even, odd) pairs.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None, : x.size(2)], sin[None, None, : x.size(2)]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c: ExperimentConfig):
        super().__init__()
        self.n_head, self.hd, self.rope = c.n_head, c.n_embd // c.n_head, c.pos_encoding == "rope"
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.p = c.dropout
        self.resid = nn.Dropout(c.dropout)

    def forward(self, x, cos=None, sin=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.hd).transpose(1, 2) for t in (q, k, v))
        if self.rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.p if self.training else 0.0)
        return self.resid(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))


class MLP(nn.Module):
    """gelu / relu2 (squared ReLU, Primer) use a 4x hidden dim; swiglu uses ~8/3x to match params."""

    def __init__(self, c: ExperimentConfig):
        super().__init__()
        self.act = c.activation
        if self.act == "swiglu":
            hidden = max(8, int(round(8 * c.n_embd / 3 / 8)) * 8)
            self.w_gate = nn.Linear(c.n_embd, hidden, bias=False)
            self.w_up = nn.Linear(c.n_embd, hidden, bias=False)
            self.w_down = nn.Linear(hidden, c.n_embd, bias=False)
        else:
            self.w_up = nn.Linear(c.n_embd, 4 * c.n_embd, bias=False)
            self.w_down = nn.Linear(4 * c.n_embd, c.n_embd, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        if self.act == "swiglu":
            h = F.silu(self.w_gate(x)) * self.w_up(x)
        elif self.act == "relu2":
            h = F.relu(self.w_up(x)).square()
        else:
            h = F.gelu(self.w_up(x))
        return self.drop(self.w_down(h))


class Block(nn.Module):
    def __init__(self, c: ExperimentConfig):
        super().__init__()
        self.pre = c.norm_position == "pre"
        self.n1, self.n2 = make_norm(c.norm, c.n_embd), make_norm(c.norm, c.n_embd)
        self.attn, self.mlp = Attention(c), MLP(c)

    def forward(self, x, cos=None, sin=None):
        if self.pre:   # Pre-LN: norm inside the residual branch
            x = x + self.attn(self.n1(x), cos, sin)
            return x + self.mlp(self.n2(x))
        x = self.n1(x + self.attn(x, cos, sin))   # Post-LN: norm after the residual add
        return self.n2(x + self.mlp(x))


class GPT(nn.Module):
    def __init__(self, c: ExperimentConfig, vocab_size: int):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(vocab_size, c.n_embd)
        self.pos = nn.Embedding(c.seq_len, c.n_embd) if c.pos_encoding == "learned" else None
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.final = make_norm(c.norm, c.n_embd) if c.norm_position == "pre" else nn.Identity()
        self.head = nn.Linear(c.n_embd, vocab_size, bias=False)
        self.apply(self._init)
        for n, p in self.named_parameters():   # GPT-2 style residual scaling
            if n.endswith("proj.weight") or n.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))
        self._rope_cache: dict = {}

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _rope(self, T: int, device):
        key = (T, str(device))
        if key not in self._rope_cache:
            self._rope_cache[key] = rope_tables(T, self.c.n_embd // self.c.n_head, device)
        return self._rope_cache[key]

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx)
        cos = sin = None
        if self.pos is not None:
            x = x + self.pos(torch.arange(T, device=idx.device))[None]
        else:
            cos, sin = self._rope(T, idx.device)
        x = self.drop(x)
        for b in self.blocks:
            x = b(x, cos, sin)
        logits = self.head(self.final(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
