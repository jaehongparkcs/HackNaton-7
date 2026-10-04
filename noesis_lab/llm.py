"""Anthropic wrapper with record / replay (BUILD_PLAN s8).

Modes
  live    real API call. If `recordings_path` is set, every request/response is appended to it.
  mock    offline scripted responses (tests / smoke only; never headline results).
  replay  serve from recordings keyed by a hash of (model, effort, role, system, user, schema).
          A miss is an error: replay must be exact, never silently live.

Effort is per role (`llm.effort_by_role`, falling back to `llm.effort`) and is part of the key, so a
bundle recorded before per-role effort replays under its own config with its single effort.
Live calls have an explicit timeout (`llm.timeout_s`, `llm.max_retries`): a stalled request fails
and goes through the crash-seal path instead of hanging the session.

Concurrency (`llm.concurrency` workers; 1 = sequential, the default for older configs). Calls are
I/O-bound, so callers may issue them from a thread pool; this class is thread-safe for that.
Callers apply results in a fixed order, so decisions never depend on which call returned first.
`speculate()` runs calls ahead of the sequential path (live mode only): their responses are held,
not recorded, and are served to the sequential path only for a byte-identical request (same key),
which then records them and writes the event in its usual order. A speculative response the
sequential path never asks for is dropped (its cost still counts towards the budget).

Rules enforced here (SPEC): strict JSON validated by Pydantic; invalid output -> one retry with the
validation error -> fail loudly. Refusals and truncation also fail loudly.
"""
from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from .schemas import canonical_json, sha256_hex

T = TypeVar("T", bound=BaseModel)
Mode = Literal["live", "mock", "replay"]
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    pass


class ReplayMiss(LLMError):
    pass


class BudgetExceeded(RuntimeError):
    pass


def strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema with additionalProperties:false on every object (structured outputs).

    Pydantic's `title` metadata is stripped from schema nodes, but never from a `properties`
    (or `$defs`) dict, whose keys are field / model names: a field called `title` must survive."""
    def walk(node: Any, names: bool = False) -> Any:
        if isinstance(node, dict):
            node = {k: walk(v, names=k in ("properties", "$defs")) for k, v in node.items()
                    if names or k != "title"}
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node.setdefault("properties", {})
                node["required"] = list(node["properties"])
            return node
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk(model.model_json_schema())


def request_key(model: str, effort: str, role: str, system: str, user: str,
                schema: dict) -> str:
    return sha256_hex(canonical_json({"model": model, "effort": effort, "role": role,
                                      "system": system, "user": user, "schema": schema}))


class LLM:
    def __init__(self, mode: Mode, cfg: dict, *, recordings_path: str | Path | None = None,
                 store=None, mock_fn: Callable[[str, str, str, type[BaseModel]], BaseModel] | None = None,
                 max_cost_usd: float | None = None):
        self.mode, self.cfg, self.store, self.mock_fn = mode, cfg, store, mock_fn
        self.model, self.effort = cfg["model"], cfg["effort"]
        self.recordings_path = Path(recordings_path) if recordings_path else None
        self.max_cost = max_cost_usd
        self.cost_usd = 0.0
        self.n_calls = 0
        self._replay: dict[str, dict] = {}
        self._client = None
        self.workers = max(1, int(cfg.get("concurrency", 1)))
        self._lock = threading.Lock()
        self._local = threading.local()
        self._prefetched: dict[str, dict] = {}
        self.n_speculative = 0
        if mode == "replay":
            if not self.recordings_path or not self.recordings_path.exists():
                raise LLMError(f"replay needs recordings at {self.recordings_path}")
            for line in self.recordings_path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._replay[rec["key"]] = rec
        elif self.recordings_path:
            self.recordings_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ transport
    @contextmanager
    def speculate(self) -> Iterator[None]:
        """Calls made in this block (on this thread) are prefetches: not recorded, no event."""
        self._local.speculative = True
        try:
            yield
        finally:
            self._local.speculative = False

    @property
    def speculative(self) -> bool:
        return getattr(self._local, "speculative", False)

    def effort_for(self, role: str) -> str:
        return self.cfg.get("effort_by_role", {}).get(role, self.effort)

    def _connect(self) -> None:
        if self._client is None:
            import anthropic
            from dotenv import load_dotenv
            load_dotenv()
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise LLMError("ANTHROPIC_API_KEY not set (live/record mode only; replay needs none)")
            self._client = anthropic.Anthropic(timeout=float(self.cfg.get("timeout_s", 120)),
                                               max_retries=int(self.cfg.get("max_retries", 2)))

    def _live(self, role: str, system: str, user: str, schema: dict) -> dict:
        with self._lock:
            self._connect()
        kw: dict[str, Any] = dict(
            model=self.model, max_tokens=self.cfg["max_tokens"], system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort_for(role), "format": {"type": "json_schema", "schema": schema}})
        if self.cfg.get("server_side_fallback", False):
            resp = self._client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kw)
        else:
            resp = self._client.messages.create(**kw)
        if resp.stop_reason == "refusal":
            raise LLMError(f"model refused: {getattr(resp, 'stop_details', None)}")
        if resp.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return {"response_text": text, "served_model": resp.model,
                "usage": {"input_tokens": resp.usage.input_tokens,
                          "output_tokens": resp.usage.output_tokens},
                "stop_reason": resp.stop_reason}

    def _cost(self, usage: dict) -> float:
        p = self.cfg["price_per_mtok"]
        return (usage.get("input_tokens", 0) * p["input"] + usage.get("output_tokens", 0) * p["output"]) / 1e6

    def _raw(self, role: str, system: str, user: str, schema_cls: type[BaseModel],
             schema: dict) -> tuple[str, dict, bool]:
        """Returns (key, record, paid): `paid` is False when the response was prefetched and its
        cost was already counted when the speculative call was made."""
        effort = self.effort_for(role)
        key = request_key(self.model, effort, role, system, user, schema)
        if self.mode == "replay":
            rec = self._replay.get(key)
            if rec is None:
                raise ReplayMiss(f"no recording for {role} request {key[:12]} "
                                 "(prompt or schema changed since recording?)")
            return key, rec, True
        with self._lock:
            held = None if self.speculative else self._prefetched.pop(key, None)
            if held is None and self.max_cost is not None and self.cost_usd >= self.max_cost:
                raise BudgetExceeded(f"LLM budget ${self.max_cost:.2f} exhausted")
        if held is not None:
            rec = held
        elif self.mode == "mock":
            out = self.mock_fn(role, system, user, schema_cls)       # type: ignore[misc]
            rec = {"response_text": out.model_dump_json(), "served_model": "mock",
                   "usage": {"input_tokens": 0, "output_tokens": 0}, "stop_reason": "end_turn"}
        else:
            rec = self._live(role, system, user, schema)
        rec = {"key": key, "role": role, "model": self.model, "effort": effort,
               "system": system, "user": user, "schema_name": schema_cls.__name__,
               **{k: v for k, v in rec.items() if k not in ("key", "role", "model", "effort", "system", "user",
                                                            "schema_name")}}
        with self._lock:
            if self.speculative:
                self._prefetched[key] = rec
                self.n_speculative += 1
            elif self.recordings_path:
                with self.recordings_path.open("a") as f:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
        return key, rec, held is None

    # ------------------------------------------------------------------ public
    def call(self, role: str, system: str, user: str, schema_cls: type[T], *,
             validate: Callable[[T], None] | None = None) -> tuple[T, str]:
        """Returns (parsed, event_id). One retry on invalid output, then raises."""
        schema = strict_schema(schema_cls)
        prompt, last_err = user, ""
        for attempt in (1, 2):
            key, rec, paid = self._raw(role, system, prompt, schema_cls, schema)
            usage = rec.get("usage", {})
            cost = self._cost(usage)
            with self._lock:
                if paid:
                    self.cost_usd += cost
                if not self.speculative:
                    self.n_calls += 1
            try:
                parsed = schema_cls.model_validate_json(rec["response_text"])
                if validate:
                    validate(parsed)
                err = ""
            except (ValidationError, ValueError) as e:
                parsed, err = None, str(e)[:1500]
            eid = (f"speculative-{key[:8]}" if self.speculative else
                   self._event(role, key, rec, attempt, usage, cost, err))
            if parsed is not None:
                return parsed, eid
            last_err = err
            prompt = (f"{user}\n\nYour previous output was rejected:\n{err}\n"
                      "Return corrected JSON only.")
        raise LLMError(f"{role}: invalid output after retry: {last_err}")

    def _event(self, role: str, key: str, rec: dict, attempt: int, usage: dict, cost: float,
               err: str) -> str:
        if self.store is None:
            return f"unrecorded-{key[:8]}"
        return self.store.add_event("llm_call", {
            "role": role, "request_hash": key, "model_requested": self.model,
            "model_served": rec.get("served_model"), "effort": self.effort_for(role), "attempt": attempt,
            "usage": usage, "cost_usd": round(cost, 6), "response_text": rec["response_text"],
            "system": rec.get("system", ""), "user": rec.get("user", ""),
            "valid": not err, "validation_error": err})
