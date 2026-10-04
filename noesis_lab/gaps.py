"""Gaps from graph structure (GRAPH_GAPS Part A). Pure functions over frozen claims: no LLM, no
I/O, no randomness. Scores are structural hints, never probabilities; the prior-art gate and the
measured result still decide.

Graph: methods M (runnable "field=value" changes), mechanisms K (a fixed vocabulary; each use is
backed by a verbatim quote),
setting buckets S (covers / partial / none), one outcome O (model quality). Every directional claim
is a signed edge (method, setting, sign); a claim with a mechanism adds method→mechanism and
mechanism→outcome edges.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from . import stats
from .schemas import MECHANISMS, Claim, QueueItem
from .search.coverage import claim_coverage

TIER_WEIGHT = {"T1": 1.0, "T2": 0.8, "T3": 0.6, "T4": 0.0, "D-confirmed": 1.0, "D-explore": 0.3}
SIGN = {"improves": "+", "no_worse": "0", "worse": "-"}        # "context" claims carry no sign
BUCKETS = ("covers", "partial", "none")

# Fixed similarity of a claim's setting to ours (transformer, char-level LM, tiny).
_FAMILY_SIM = {"transformer": 1.0, "general": 0.8, "unspecified": 0.5, "rnn": 0.3, "cnn": 0.3, "mlp": 0.3}
_TASK_SIM = {"char_language_modeling": 1.0, "language_modeling": 0.9, "general": 0.7, "unspecified": 0.5,
             "translation": 0.5, "classification": 0.4, "vision": 0.2, "speech": 0.2}
_SCALE_SIM = {"tiny": 1.0, "small": 0.9, "unspecified": 0.6, "medium": 0.5, "large": 0.4}
CURATED_SIM = {True: 1.0, False: 0.4}       # curated claims have a human covers flag, not fields


# --------------------------------------------------------------------------- mechanisms
def mechanism_nodes(claims: Sequence[Claim]) -> dict[str, dict]:
    """Mechanism nodes = the fixed categories that at least one claim uses (each with a verbatim
    quote). mechanism_id -> {label, members (the quotes)}."""
    out: dict[str, dict] = {}
    for c in sorted(claims, key=lambda c: c.claim_id):
        k = c.mechanism_category
        if k in MECHANISMS and c.mechanism:
            out.setdefault(k, {"label": k.replace("_", " "), "members": []})["members"].append(c.mechanism)
    return dict(sorted(out.items()))


def shared_mechanisms(g: GapGraph) -> dict[str, list[str]]:
    """Mechanism nodes connected to 2 or more runnable methods. ABC closure, link prediction and
    bridges depend on these; check the count right after a search (fewer than ~3 = thin graph)."""
    by: dict[str, set[str]] = {}
    for e in g.edges:
        if e.method and e.mechanism_id:
            by.setdefault(e.mechanism_id, set()).add(e.method)
    return {k: sorted(v) for k, v in sorted(by.items()) if len(v) >= 2}


# --------------------------------------------------------------------------- graph
class GEdge(BaseModel):
    """One claim as graph edges."""
    claim_id: str
    paper_id: str
    tier: str
    weight: float                 # tier weight
    method: str | None            # runnable "field=value", or None (method outside the testbed)
    mechanism_id: str | None
    sign: str | None              # "+", "0", "-" or None (context)
    bucket: str                   # covers / partial / none
    similarity: float             # of the claim's setting to ours (fixed table)


class GapGraph(BaseModel):
    methods: list[str]
    mechanisms: dict[str, dict[str, Any]]
    edges: list[GEdge]


def method_key(c: Claim) -> str | None:
    if c.config_change and len(c.config_change) == 1:
        (f, v), = c.config_change.items()
        return f"{f}={v}"
    return None


def setting_similarity(c: Claim) -> float:
    s = c.setting_fields
    if s is None:
        return CURATED_SIM[bool(c.covers_our_setting)]
    return round(_FAMILY_SIM[s.model_family] * _TASK_SIM[s.task] * _SCALE_SIM[s.scale], 6)


def build_graph(claims: Sequence[Claim]) -> GapGraph:
    claims = sorted(claims, key=lambda c: c.claim_id)
    mechs = mechanism_nodes(claims)
    edges = [GEdge(claim_id=c.claim_id, paper_id=c.paper_id, tier=c.tier, weight=TIER_WEIGHT[c.tier],
                   method=method_key(c),
                   mechanism_id=c.mechanism_category if c.mechanism_category in mechs and c.mechanism else None,
                   sign=SIGN.get(c.expected_outcome), bucket=claim_coverage(c),
                   similarity=setting_similarity(c)) for c in claims]
    return GapGraph(methods=sorted({e.method for e in edges if e.method}), mechanisms=mechs, edges=edges)


# --------------------------------------------------------------------------- coverage tensor
def coverage_tensor(g: GapGraph) -> dict[str, dict[str, dict[str, float]]]:
    """T[method][bucket][sign] = tier-weighted number of directional claims (T1 1.0, T2 0.8, T3 0.6)."""
    t: dict[str, dict[str, dict[str, float]]] = {
        m: {b: {"+": 0.0, "0": 0.0, "-": 0.0} for b in BUCKETS} for m in g.methods}
    for e in g.edges:
        if e.method and e.sign:
            t[e.method][e.bucket][e.sign] = round(t[e.method][e.bucket][e.sign] + e.weight, 6)
    return t


def strongly_covered(g: GapGraph, method: str) -> bool:
    """Is this method already covered in our setting by something that closes the gap — a literature
    claim or our own paired (D-confirmed) result? A single-seed exploration (D-explore) only lowers
    the gap score (via `coverage_in_our_setting`); it never closes the gap on its own."""
    return any(e.bucket == "covers" and e.sign and e.tier != "D-explore"
               for e in g.edges if e.method == method)


def coverage_in_our_setting(tensor: dict, method: str) -> float:
    """In [0, 1]: tier-weighted claims that cover our setting (partial counts half), capped at 1."""
    cell = tensor.get(method)
    if not cell:
        return 0.0
    return round(min(sum(cell["covers"].values()) + 0.5 * sum(cell["partial"].values()), 1.0), 6)


# --------------------------------------------------------------------------- gaps
class Gap(BaseModel):
    """A structural hint, with the components of its score and the path that explains it."""
    gap_id: str                    # "<type>:<delta key>[:<mechanism id>]"
    gap_type: str                  # coverage | abc | link | contradiction | bridge
    novelty_type: str              # transfer | combination | resolution
    delta_key: str
    delta: dict[str, str]
    plausibility: float            # the gap type's own formula
    coverage: float                # coverage_in_our_setting, in [0, 1]
    testability: float
    evidence_quality: float        # mean tier weight along the explanation path
    score: float                   # plausibility × (1 − coverage) × testability × evidence_quality
    path: list[dict[str, Any]]     # [{"from", "to", "claim_id"}]: why the gap exists
    claim_ids: list[str]
    detail: dict[str, Any] = {}
    # Sign of the method's effect on quality that the path predicts for our setting, from the
    # signs of the claims on it: "+" better, "-" worse, "0" no worse, "" = the gap predicts none
    # (a contradiction has no direction; a combination has one only under prediction_version 2,
    # from its components' literature signs).
    predicted_direction: str = ""


NOVELTY_ORDER = {"resolution": 0, "combination": 1, "transfer": 2}
GAP_SCORE_TEXT = (
    "gap_score = plausibility × (1 − coverage_in_our_setting) × testability × evidence_quality.\n"
    "plausibility: the gap type's formula (transfer consistency × setting similarity; ABC path support; "
    "normalized Adamic–Adar × complementarity; contradiction balance × setting similarity; bridge).\n"
    "coverage_in_our_setting: tier-weighted claims covering our setting (partial counts half), capped at 1.\n"
    "testability: 1 for a typed delta of 1–2 fields, 0.3 for a parameterized variant not built yet, 0 outside the testbed.\n"
    "evidence_quality: mean tier weight (T1 1.0, T2 0.8, T3 0.6) along the explanation path.\n"
    "Ties: resolution > combination > transfer, then alphabetical. Scores are structural hints, not probabilities."
)


COMBINATION_RULE_TEXT = (
    "Combination prediction (protocol prediction_version 2): each component's literature sign is the "
    "tier-weighted majority sign of its directional literature claims (our own derived results excluded; "
    "ties: + over 0 over −). Predicted sign = the sum of the components' signs: both + → +; + and 0 → +; "
    "both − → −; − and 0 → −; both 0 → 0; + and − (mixed), or a component with no directional claim → ? "
    "(no prediction)."
)


def literature_sign(g: GapGraph, method: str) -> str:
    """Tier-weighted majority sign of a method's directional LITERATURE claims ("" if none)."""
    edges = [e for e in g.edges if e.method == method and e.sign and not e.tier.startswith("D-")]
    if not edges:
        return ""
    by_sign = {s: sum(e.weight for e in edges if e.sign == s) for s in "+0-"}
    return max("+0-", key=lambda s: (by_sign[s], -"+0-".index(s)))


def combination_direction(g: GapGraph, methods: Sequence[str]) -> str:
    """The sum of the components' literature signs; "" (no prediction) if mixed or unknown."""
    signs = [literature_sign(g, m) for m in methods]
    if "" in signs or {"+", "-"} <= set(signs):
        return ""
    total = sum({"+": 1, "0": 0, "-": -1}[s] for s in signs)
    return "+" if total > 0 else "-" if total < 0 else "0"


def gap_score(plausibility: float, coverage: float, testability: float, evidence_quality: float) -> float:
    return round(plausibility * (1.0 - coverage) * testability * evidence_quality, 6)


def _mk(gap_type: str, novelty: str, delta: dict[str, str], plaus: float, cov: float,
        path: list[dict], g: GapGraph, detail: dict, suffix: str = "", direction: str = "") -> Gap:
    delta_key, testability = stats.delta_key, stats.testability
    w = {e.claim_id: e.weight for e in g.edges}
    ids = sorted({p["claim_id"] for p in path if p.get("claim_id")})
    quality = round(sum(w[i] for i in ids) / len(ids), 6) if ids else 0.0
    key = delta_key(delta)
    test = testability(delta)
    return Gap(gap_id=f"{gap_type}:{key}{suffix}", gap_type=gap_type, novelty_type=novelty, delta_key=key,
               delta=delta, plausibility=round(plaus, 6), coverage=round(cov, 6), testability=test,
               evidence_quality=quality, score=gap_score(plaus, cov, test, quality), path=path,
               claim_ids=ids, detail=detail, predicted_direction=direction)


def _delta(method: str) -> dict[str, str]:
    f, v = method.split("=", 1)
    return {f: v}


def coverage_gaps(g: GapGraph, tensor: dict) -> list[Gap]:
    """Missing cell: no directional claim for method m covers our setting, but claims exist
    elsewhere. Plausibility = share of m's tier-weighted claims with the majority sign × the
    weighted mean similarity of those claims' settings to ours."""
    out = []
    for m in g.methods:
        edges = [e for e in g.edges if e.method == m and e.sign]
        if not edges or strongly_covered(g, m):     # a D-explore edge lowers the score but does not close the gap
            continue
        by_sign = {s: sum(e.weight for e in edges if e.sign == s) for s in "+0-"}
        sign = max("+0-", key=lambda s: (by_sign[s], -"+0-".index(s)))
        major = [e for e in edges if e.sign == sign]
        frac = by_sign[sign] / sum(by_sign.values())
        sim = sum(e.weight * e.similarity for e in major) / by_sign[sign]
        path = [{"from": m, "to": f"quality ({e.sign}) in a {e.bucket}-coverage setting", "claim_id": e.claim_id}
                for e in major]
        out.append(_mk("coverage", "transfer", _delta(m), frac * sim, coverage_in_our_setting(tensor, m),
                       path, g, {"expected_sign": sign, "same_sign_fraction": round(frac, 6),
                                 "setting_similarity": round(sim, 6)}, direction=sign))
    return out


def abc_gaps(g: GapGraph, tensor: dict) -> list[Gap]:
    """Swanson ABC closure: method m → mechanism k (claim c1) and k → better quality (claim c2,
    about another method), but no claim m → quality in our setting. Score = support(m→k) ×
    support(k→quality) / (number of mechanisms on the path); supports are tier-weighted and capped
    at 1, and k→quality is also weighted by setting similarity.

    The path respects signs. A method→mechanism edge counts only if its claim presents the
    mechanism as the reason for a BENEFIT (improves or no_worse): a method that is "worse via k"
    does not gain k, so it must not inherit k's benefit. The predicted direction is the sign of the
    method's effect along the path, which with benefit-only edges and a "+" second leg is "+"."""
    out = []
    for m in g.methods:
        if strongly_covered(g, m):
            continue
        for k in sorted({e.mechanism_id for e in g.edges if e.method == m and e.mechanism_id}):
            c1 = [e for e in g.edges if e.method == m and e.mechanism_id == k and e.sign in ("+", "0")]
            if not c1:
                continue
            c2 = [e for e in g.edges if e.mechanism_id == k and e.sign == "+" and e.method != m]
            if not c2:
                continue
            s_mk = min(sum(e.weight for e in c1), 1.0)
            s_ko = min(sum(e.weight * e.similarity for e in c2), 1.0)
            label = g.mechanisms[k]["label"]
            path = ([{"from": m, "to": f"mechanism: {label}", "claim_id": e.claim_id} for e in c1]
                    + [{"from": f"mechanism: {label}", "to": "quality (+)", "claim_id": e.claim_id} for e in c2])
            out.append(_mk("abc", "transfer", _delta(m), s_mk * s_ko / 1, coverage_in_our_setting(tensor, m),
                           path, g, {"mechanism_id": k, "mechanism": label, "support_method_mechanism": round(s_mk, 6),
                                     "support_mechanism_outcome": round(s_ko, 6), "mechanisms_on_path": 1},
                           suffix=f":{k}", direction="+"))
    return out


def _neighbors(g: GapGraph) -> dict[str, set[str]]:
    """Neighbors of each method: its mechanisms, the setting buckets it was tested in, and methods
    tested in the same paper."""
    by_paper: dict[str, set[str]] = {}
    for e in g.edges:
        if e.method:
            by_paper.setdefault(e.paper_id, set()).add(e.method)
    n: dict[str, set[str]] = {m: set() for m in g.methods}
    for e in g.edges:
        if not e.method:
            continue
        if e.mechanism_id:
            n[e.method].add(f"K:{e.mechanism_id}")
        if e.sign:
            n[e.method].add(f"S:{e.bucket}")
        n[e.method] |= {f"M:{o}" for o in by_paper[e.paper_id] if o != e.method}
    return n


def _pairs(g: GapGraph) -> list[tuple[str, str]]:
    """Method pairs that can be combined (different fields) and that no single paper mentions together."""
    papers = {m: {e.paper_id for e in g.edges if e.method == m} for m in g.methods}
    return [(a, b) for i, a in enumerate(g.methods) for b in g.methods[i + 1:]
            if a.split("=")[0] != b.split("=")[0] and not papers[a] & papers[b]]


def link_scores(g: GapGraph) -> dict[tuple[str, str], dict[str, float]]:
    """Common neighbors, Adamic–Adar (Σ 1/log|N(z)|) and resource allocation (Σ 1/|N(z)|) for every
    combinable, not-yet-co-mentioned method pair. |N(z)| = number of methods adjacent to z."""
    import math
    n = _neighbors(g)
    degree: dict[str, int] = {}
    for m in g.methods:
        for z in n[m]:
            degree[z] = degree.get(z, 0) + 1
    out = {}
    for a, b in _pairs(g):
        shared = sorted(n[a] & n[b])
        out[(a, b)] = {"common_neighbors": len(shared),
                       "adamic_adar": round(sum(1.0 / math.log(degree[z]) for z in shared if degree[z] > 1), 6),
                       "resource_allocation": round(sum(1.0 / degree[z] for z in shared), 6),
                       "shared": shared}
    return out


def link_gaps(g: GapGraph, predict: bool = False) -> list[Gap]:
    """Combination gaps by link prediction. Score = Adamic–Adar normalized by the best pair ×
    complementarity (1.0 if both methods have mechanisms and they differ: likely additive; 0.5 if
    they share one: likely redundant; 0.75 if a mechanism is unknown). A pair needs at least one
    shared mechanism or co-tested method; a shared setting bucket alone only adds to the score."""
    scores = link_scores(g)
    top = max((s["adamic_adar"] for s in scores.values()), default=0.0)
    mechs = {m: {e.mechanism_id for e in g.edges if e.method == m and e.mechanism_id} for m in g.methods}
    out = []
    for (a, b), s in sorted(scores.items()):
        if top == 0 or not any(z[0] in "KM" for z in s["shared"]):
            continue        # sharing only a setting bucket is not a hint: need a mechanism or a co-tested method
        comp = 0.75 if not (mechs[a] and mechs[b]) else (0.5 if mechs[a] & mechs[b] else 1.0)
        path = []
        for z in s["shared"]:
            for m in (a, b):
                e = _edge_to(g, m, z)
                if e:
                    path.append({"from": m, "to": _label(g, z), "claim_id": e.claim_id})
        out.append(_mk("link", "combination", {**_delta(a), **_delta(b)}, s["adamic_adar"] / top * comp, 0.0,
                       path, g, {k: v for k, v in s.items() if k != "shared"} | {
                           "shared_neighbors": [_label(g, z) for z in s["shared"]], "complementarity": comp},
                       direction=combination_direction(g, (a, b)) if predict else ""))
    return out


def _label(g: GapGraph, z: str) -> str:
    kind, name = z.split(":", 1)
    return {"K": lambda: f"mechanism: {g.mechanisms[name]['label']}", "S": lambda: f"tested in a {name}-coverage setting",
            "M": lambda: f"co-tested with {name}"}[kind]()


def _edge_to(g: GapGraph, method: str, z: str) -> GEdge | None:
    """The strongest claim connecting a method to neighbor z (ties by claim id)."""
    kind, name = z.split(":", 1)
    papers_with = {e.paper_id for e in g.edges if e.method == name} if kind == "M" else set()
    cands = [e for e in g.edges if e.method == method and (
        (kind == "K" and e.mechanism_id == name) or (kind == "S" and e.sign and e.bucket == name)
        or (kind == "M" and e.paper_id in papers_with))]
    return min(cands, key=lambda e: (-e.weight, e.claim_id)) if cands else None


def contradiction_gaps(g: GapGraph, tensor: dict) -> list[Gap]:
    """The same (method, setting bucket) has claims with opposite signs. Score = min(support+,
    support−) / max(...) (1 = evenly split) × weighted mean setting similarity. Contested evidence
    does not count as coverage: the coverage component is scaled by (1 − balance)."""
    out = []
    for m in g.methods:
        for b in BUCKETS:
            pos = [e for e in g.edges if e.method == m and e.bucket == b and e.sign == "+"]
            neg = [e for e in g.edges if e.method == m and e.bucket == b and e.sign == "-"]
            if not pos or not neg:
                continue
            sp, sn = sum(e.weight for e in pos), sum(e.weight for e in neg)
            balance = min(sp, sn) / max(sp, sn)
            sim = sum(e.weight * e.similarity for e in pos + neg) / (sp + sn)
            path = [{"from": m, "to": f"quality ({e.sign}) in a {b}-coverage setting", "claim_id": e.claim_id}
                    for e in pos + neg]
            out.append(_mk("contradiction", "resolution", _delta(m), balance * sim,
                           coverage_in_our_setting(tensor, m) * (1 - balance), path, g,
                           {"bucket": b, "support_plus": round(sp, 6), "support_minus": round(sn, 6),
                            "balance": round(balance, 6), "setting_similarity": round(sim, 6)}, suffix=f":{b}"))
    return out


def communities(g: GapGraph) -> dict[str, int]:
    """Greedy-modularity communities of the method–mechanism graph. Node names: "M:<method>" and
    "K:<mechanism id>". Isolated methods get their own community. Deterministic (sorted insertion)."""
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities
    graph = nx.Graph()
    graph.add_nodes_from([f"M:{m}" for m in g.methods] + [f"K:{k}" for k in sorted(g.mechanisms)])
    for e in sorted(g.edges, key=lambda e: e.claim_id):
        if e.method and e.mechanism_id:
            graph.add_edge(f"M:{e.method}", f"K:{e.mechanism_id}")
    if graph.number_of_edges() == 0:
        return {n: i for i, n in enumerate(sorted(graph.nodes))}
    comms = sorted((sorted(c) for c in greedy_modularity_communities(graph)), key=lambda c: c[0])
    return {n: i for i, c in enumerate(comms) for n in c}


def bridge_gaps(g: GapGraph, predict: bool = False) -> list[Gap]:
    """Structural holes: method pairs from different communities, joined by a path of length 2
    through a mechanism, with no paper mentioning both. Score = (1 − fraction of the two
    communities' edges that already cross between them) × path support (the weaker of the two
    method→mechanism supports, capped at 1)."""
    comm = communities(g)
    links = sorted({(f"M:{e.method}", f"K:{e.mechanism_id}") for e in g.edges if e.method and e.mechanism_id})
    out = []
    for a, b in _pairs(g):
        ca, cb = comm[f"M:{a}"], comm[f"M:{b}"]
        if ca == cb:
            continue
        shared = sorted({k for m, k in links if m == f"M:{a}"} & {k for m, k in links if m == f"M:{b}"})
        if not shared:
            continue
        inside = [(m, k) for m, k in links if {comm[m], comm[k]} <= {ca, cb}]
        cross = [(m, k) for m, k in inside if comm[m] != comm[k]]
        frac = len(cross) / len(inside) if inside else 0.0
        best = None
        for k in shared:
            sup = min(*(min(sum(e.weight for e in g.edges if e.method == m and f"K:{e.mechanism_id}" == k), 1.0)
                        for m in (a, b)))
            if best is None or sup > best[0]:
                best = (sup, k)
        sup, k = best
        path = [{"from": m, "to": _label(g, k), "claim_id": _edge_to(g, m, k).claim_id} for m in (a, b)]
        out.append(_mk("bridge", "combination", {**_delta(a), **_delta(b)}, (1 - frac) * sup, 0.0, path, g,
                       {"communities": [ca, cb], "cross_edge_fraction": round(frac, 6),
                        "path_support": round(sup, 6), "mechanism": g.mechanisms[k.split(':', 1)[1]]["label"]},
                       direction=combination_direction(g, (a, b)) if predict else ""))
    return out


def rank_gaps(found: Sequence[Gap]) -> list[Gap]:
    """One ranked list. Ties: resolution > combination > transfer, then alphabetical by delta key."""
    return sorted(found, key=lambda x: (-x.score, NOVELTY_ORDER[x.novelty_type], x.delta_key, x.gap_id))


def find_gaps(claims: Sequence[Claim], predict_combinations: bool = False) -> tuple[GapGraph, list[Gap]]:
    """`predict_combinations` (protocol prediction_version >= 2) gives combination gaps a predicted
    direction (COMBINATION_RULE_TEXT); earlier bundles recorded them with none and replay so."""
    g = build_graph(claims)
    t = coverage_tensor(g)
    p = predict_combinations
    return g, rank_gaps([*coverage_gaps(g, t), *abc_gaps(g, t), *link_gaps(g, p),
                         *contradiction_gaps(g, t), *bridge_gaps(g, p)])


# --------------------------------------------------------------------------- novelty type + queue
def novelty_type(delta: dict[str, str], g: GapGraph, tensor: dict) -> str:
    """Label computed by code, never by an LLM (EXPLORE §5):
      new_mechanism  not expressible as a typed delta → outside the testbed, map only
      resolution     1 field; the method has claims with opposite signs in one setting bucket
      combination    2 fields; each method has claims; no paper mentions both
      transfer       1 field; the method has claims elsewhere; none covers our setting
      none           anything else (e.g. already covered, or no claims at all)"""
    if not stats.valid_delta(delta):
        return "new_mechanism"
    methods = [f"{f}={v}" for f, v in sorted(delta.items())]
    has = {m: [e for e in g.edges if e.method == m] for m in methods}
    if len(methods) == 2:
        together = {e.paper_id for e in has[methods[0]]} & {e.paper_id for e in has[methods[1]]}
        return "combination" if all(has.values()) and not together else "none"
    m = methods[0]
    for b in BUCKETS:
        cell = tensor.get(m, {}).get(b, {})
        if cell.get("+", 0) > 0 and cell.get("-", 0) > 0:
            return "resolution"
    directional = [e for e in has[m] if e.sign]
    return "transfer" if directional and not strongly_covered(g, m) else "none"


def gap_queue(found: Sequence[Gap], g: GapGraph, claims: Sequence[Claim], tested: Sequence[str] = (), *,
              soft_rejected: Sequence[str] = (), held: Sequence[str] = (), overrides: Sequence[str] = (),
              top_n: int = 8) -> list[QueueItem]:
    """The candidate queue of the hypothesis engine: one item per config delta, ordered by
    gap_score (ties: resolution > combination > transfer, then alphabetical). Gaps with the same
    delta are merged into one item linked to all of them. Already-tested deltas are removed first,
    then only the top_n remaining deltas get hypotheses, so a tested one never uses up a slot.
    Claims that disagree in sign within a setting bucket do not cover a topic (open, contested).
    Literature status (rejected / soft-rejected / held) is applied exactly as in
    `stats.literature_status`; a combination is covered only by a claim about the combination,
    which by construction does not exist, so it is open unless the gate soft-rejected it."""
    tensor = coverage_tensor(g)
    by_delta: dict[str, list[Gap]] = {}
    for x in rank_gaps(found):
        if x.score > 0 and x.testability == 1.0:
            by_delta.setdefault(x.delta_key, []).append(x)
    by_method: dict[str, list[Claim]] = {}
    for c in claims:
        k = method_key(c)
        if k:
            by_method.setdefault(k, []).append(c)
    items = []
    untested = [(k, grp) for k, grp in by_delta.items() if k not in set(tested)]   # filter first, then cut
    for key, group in untested[:top_n]:
        best = group[0]
        label = novelty_type(best.delta, g, tensor)
        label = label if label in NOVELTY_ORDER else best.novelty_type
        status, cov, note = stats.literature_status(key, by_method.get(key, []) if len(best.delta) == 1 else [],
                                                    soft_rejected, held, overrides, contested_open=True)
        items.append(QueueItem(
            key=key, field="+".join(sorted(best.delta)), value="+".join(best.delta[f] for f in sorted(best.delta)),
            priority=[-best.score, NOVELTY_ORDER[label], key],
            supporting_claim_ids=sorted({c for x in group for c in x.claim_ids}), status=status,
            covering_claim_ids=cov, note=note, delta=best.delta, gap_ids=[x.gap_id for x in group],
            gap_score=best.score, novelty_type=label))
    return sorted(items, key=lambda i: (stats.QUEUE_STATUS_ORDER[i.status], *i.priority))


# --------------------------------------------------------------------------- archive + write-back
def archive_cell(delta: dict[str, str], g: GapGraph) -> str:
    """Quality-diversity cell of a hypothesis: its method family (the fields it changes) × the
    mechanisms the graph links to its methods. The archive keeps the best explored run per cell."""
    methods = {f"{f}={v}" for f, v in delta.items()}
    mechs = sorted({e.mechanism_id for e in g.edges if e.method in methods and e.mechanism_id})
    return "+".join(sorted(delta)) + " | " + (", ".join(mechs) or "no mechanism")


DERIVED_PAPER = "noesis_lab_derived"


def derived_claim(delta: dict[str, str], branch: str, tier: str = "D-confirmed") -> Claim | None:
    """Our own result written back as a graph edge. It is tagged derived (tier D-explore for a
    single-seed exploration, D-confirmed for a paired screening): it shapes the gap graph — once we
    have measured a change, its cell is less of a gap — but it is NEVER shown to the gate, used as a
    literature search, or counted as prior art (`counts_as_covered` excludes it). Only a single-field
    result maps to one method; a combination result adds no edge."""
    if len(delta) != 1:
        return None
    key = stats.delta_key(delta)
    kind = {"D-explore": "single-seed exploration", "D-confirmed": "paired screening"}.get(tier, "measurement")
    outcome = {"promising": "improves", "harmful": "worse", "no_improvement": "no_worse"}[branch]
    return Claim(claim_id="ours_" + key.replace("=", "_").replace(".", "_"), paper_id=DERIVED_PAPER,
                 dimension=next(iter(delta)), method=key, setting=f"our setting ({kind} in this session)",
                 claim=f"derived: {kind} measured {branch}", source_span="",
                 expected_outcome=outcome, covers_our_setting=True, coverage="covers", tier=tier,
                 config_change=dict(delta))
