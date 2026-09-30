"""Evidence ranking: hard applicability rules first, then trust-weighted Personalized PageRank.

Pipeline:
  1. retrieve   - sources tagged with the question's topic
  2. applicable - country / client / employee scope / in force on the question date
  3. outdated   - superseded by an in-force source, or citing only outdated sources (stale dependency)
  4. rank       - Personalized PageRank over 'cites' edges, teleport vector = trust prior
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .knowledge_base import KnowledgeBase
from .models import ANY_EMPLOYEE, GENERIC_CLIENT, Question, Source

TIER_PRIOR = {1: 1.0, 2: 0.7, 3: 0.5, 4: 0.25, 5: 0.1, 6: 0.05}
VALIDATED_BONUS = 2.0
CLIENT_SPECIFIC_BONUS = 1.5
DAMPING = 0.85


@dataclass
class RankedSource:
    source: Source
    score: float
    notes: list[str] = field(default_factory=list)
    needs_context: list[str] = field(default_factory=list)   # assumptions the consultant must confirm


@dataclass
class RankingResult:
    ranked: list[RankedSource]
    outdated: dict[str, str]          # source id -> reason
    not_applicable: dict[str, str]    # source id -> reason


def in_force(source: Source, q: Question) -> bool:
    return source.effective_from <= q.on_date and (
        source.effective_to is None or q.on_date <= source.effective_to)


def applicability_problem(source: Source, q: Question) -> str | None:
    """Return why a source does not apply to the question, or None if it applies."""
    if source.country != q.country:
        return f"country {source.country} != {q.country}"
    if source.client not in (GENERIC_CLIENT, q.client):
        return "belongs to another client"  # client isolation: never leak across clients
    # An unspecified employee context does not exclude narrower sources; it is flagged in ranking instead.
    if (q.employee_ctx != ANY_EMPLOYEE and ANY_EMPLOYEE not in source.employee_scope
            and q.employee_ctx not in source.employee_scope):
        return f"employee scope {source.employee_scope} excludes '{q.employee_ctx}'"
    if not in_force(source, q):
        return f"not in force on {q.on_date.isoformat()}"
    return None


def trust_prior(source: Source, q: Question) -> float:
    p = TIER_PRIOR[source.tier]
    if source.validated and (source.review_by is None or q.on_date <= source.review_by):
        p *= VALIDATED_BONUS
    if source.client != GENERIC_CLIENT and source.client == q.client:
        p *= CLIENT_SPECIFIC_BONUS
    return p


def personalized_pagerank(nodes, edges, teleport, d=DAMPING, iters=100):
    """Power iteration. edges: list of (u, v). teleport: node -> probability (sums to 1)."""
    out_deg = {u: 0 for u in nodes}
    for u, _ in edges:
        out_deg[u] += 1
    rank = dict(teleport)
    for _ in range(iters):
        new = {v: (1 - d) * teleport[v] for v in nodes}
        dangling = sum(rank[u] for u in nodes if out_deg[u] == 0)
        for v in nodes:
            new[v] += d * dangling * teleport[v]
        for u, v in edges:
            new[v] += d * rank[u] / out_deg[u]
        rank = new
    return rank


def rank_evidence(kb: KnowledgeBase, q: Question) -> RankingResult:
    retrieved = [s for s in kb.sources.values() if q.topic in s.topics]

    not_applicable, outdated, applicable = {}, {}, []
    for s in retrieved:
        problem = applicability_problem(s, q)
        if problem is None:
            applicable.append(s)
        elif problem.startswith("not in force"):
            outdated[s.id] = problem
        else:
            not_applicable[s.id] = problem

    retrieved_ids = {s.id for s in retrieved}

    # Supersession: an in-force replacement makes the old source outdated.
    for e in kb.edges:
        if (e.relation == "supersedes" and e.dst in retrieved_ids and e.dst not in not_applicable
                and e.src in kb.sources):
            if in_force(kb.sources[e.src], q):
                outdated.setdefault(e.dst, f"superseded by {e.src}")

    # Stale dependency: a source whose on-topic citations ALL point at outdated sources is itself
    # suspect. Citations to other topics are ignored (a guide's old min-wage section says nothing
    # about its holiday-allowance section). Iterate to a fixed point along citation chains.
    cites = {}
    for e in kb.edges:
        if e.relation == "cites" and e.dst in retrieved_ids:
            cites.setdefault(e.src, []).append(e.dst)
    changed = True
    while changed:
        changed = False
        for s in applicable:
            targets = cites.get(s.id, [])
            if s.id not in outdated and targets and all(t in outdated for t in targets):
                outdated[s.id] = "relies only on outdated sources: " + ", ".join(targets)
                changed = True

    candidates = [s for s in applicable if s.id not in outdated]
    if not candidates:
        return RankingResult([], outdated, not_applicable)

    ids = {s.id for s in candidates}
    prior = {s.id: trust_prior(s, q) for s in candidates}
    total = sum(prior.values())
    teleport = {k: v / total for k, v in prior.items()}
    edges = [(e.src, e.dst) for e in kb.edges if e.relation == "cites" and e.src in ids and e.dst in ids]
    scores = personalized_pagerank(list(ids), edges, teleport)

    ranked = []
    for s in candidates:
        notes = [f"authority tier {s.tier} ({s.type})"]
        if s.validated:
            if s.review_by and q.on_date > s.review_by:
                notes.append(f"validation by {s.validated_by} is past its review date {s.review_by}")
            else:
                notes.append(f"validated by {s.validated_by}")
        if s.client != GENERIC_CLIENT:
            notes.append(f"client-specific ({s.client})")
        if s.origin == "web":
            host = s.url.split("/")[2] if s.url.count("/") >= 2 else s.url
            notes.append(f"from the web ({host}, retrieved {s.retrieved_at}), not yet validated by an expert")
        needs = []
        if q.employee_ctx == ANY_EMPLOYEE and ANY_EMPLOYEE not in s.employee_scope:
            needs.append(f"employee is {' or '.join(s.employee_scope)}")
            notes.append(f"only for employee scope {s.employee_scope}")
        ranked.append(RankedSource(s, scores[s.id], notes, needs))
    ranked.sort(key=lambda r: (-r.score, r.source.tier))
    return RankingResult(ranked, outdated, not_applicable)
