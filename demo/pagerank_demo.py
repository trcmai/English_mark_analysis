"""Toy demo: naive PageRank vs trust-adapted Personalized PageRank for Payroll Passport.

All documents and numbers are fictional and illustrative only.
Question: "Hourly minimum wage, adult employee, country NL, pay date 2026-03-15?"
"""
from datetime import date

# ---- Toy knowledge graph -------------------------------------------------
DOCS = {
    # id: (type, authority_tier 1=highest, effective_from, effective_to, claimed_value, validated)
    "LAW_2025":   ("law",       1, date(2025, 1, 1), date(2025, 12, 31), 13.00, False),
    "LAW_2026":   ("law",       1, date(2026, 1, 1), None,               14.00, False),
    "GUIDE_2024": ("int_guide", 4, date(2024, 1, 1), None,               13.00, False),
    "EMAIL_A":    ("email",     6, date(2025, 3, 1), None,               13.00, False),
    "EMAIL_B":    ("email",     6, date(2025, 6, 1), None,               13.00, False),
    "WIKI":       ("wiki",      5, date(2025, 2, 1), None,               13.00, False),
    "KI_2026":    ("knowledge", 3, date(2026, 1, 5), None,               14.00, True),
}

# (from, to, relation): "cites" = from references to; "supersedes" = from replaces to
EDGES = [
    ("GUIDE_2024", "LAW_2025", "cites"),
    ("EMAIL_A", "LAW_2025", "cites"),
    ("EMAIL_B", "GUIDE_2024", "cites"),
    ("WIKI", "LAW_2025", "cites"),
    ("WIKI", "GUIDE_2024", "cites"),
    ("EMAIL_A", "WIKI", "cites"),
    ("KI_2026", "LAW_2026", "cites"),
    ("LAW_2026", "LAW_2025", "supersedes"),
]


def pagerank(nodes, weighted_edges, teleport=None, d=0.85, iters=100):
    """Power-iteration PageRank. weighted_edges: list of (u, v, w). teleport: dict node->prob."""
    n = len(nodes)
    teleport = teleport or {v: 1 / n for v in nodes}
    out_w = {u: 0.0 for u in nodes}
    for u, _, w in weighted_edges:
        out_w[u] += w
    rank = {v: 1 / n for v in nodes}
    for _ in range(iters):
        new = {v: (1 - d) * teleport[v] for v in nodes}
        dangling = sum(rank[u] for u in nodes if out_w[u] == 0)
        for v in nodes:  # dangling mass follows the teleport vector
            new[v] += d * dangling * teleport[v]
        for u, v, w in weighted_edges:
            new[v] += d * rank[u] * w / out_w[u]
        rank = new
    return rank


def show(title, rank):
    print(f"\n{title}")
    for doc, score in sorted(rank.items(), key=lambda x: -x[1]):
        print(f"  {doc:<11} {score:.3f}  value={DOCS[doc][4]}")


# ---- 1. Naive PageRank: every edge = endorsement -------------------------
nodes = list(DOCS)
naive = pagerank(nodes, [(u, v, 1.0) for u, v, _ in EDGES])
show("1) NAIVE PageRank (all docs, all edges = +1)", naive)

# ---- 2. Adapted: filter -> trust-weighted personalized PageRank ----------
Q_DATE = date(2026, 3, 15)


def in_force(doc):
    _, _, start, end, _, _ = DOCS[doc]
    return start <= Q_DATE and (end is None or Q_DATE <= end)


superseded = {v for u, v, r in EDGES if r == "supersedes" and in_force(u)}
candidates = [d for d in nodes if in_force(d) and d not in superseded]

# Stale-dependency propagation: a doc whose cited sources are ALL outdated is itself suspect.
# Repeat until stable, because staleness flows along citation chains (EMAIL_B -> GUIDE -> LAW_2025).
changed = True
while changed:
    changed = False
    for d in list(candidates):
        cited = [v for u, v, r in EDGES if u == d and r == "cites"]
        if cited and all(c not in candidates for c in cited):
            candidates.remove(d)
            changed = True
flagged_outdated = sorted(set(nodes) - set(candidates))

# Trust prior (teleport): higher authority tier + expert validation => more prior mass
TIER_PRIOR = {1: 1.0, 2: 0.7, 3: 0.5, 4: 0.25, 5: 0.1, 6: 0.05}
prior = {d: TIER_PRIOR[DOCS[d][1]] * (2.0 if DOCS[d][5] else 1.0) for d in candidates}
total = sum(prior.values())
prior = {d: p / total for d, p in prior.items()}

# Only "cites" edges between surviving docs carry endorsement
kept_edges = [(u, v, 1.0) for u, v, r in EDGES
              if r == "cites" and u in candidates and v in candidates]
adapted = pagerank(candidates, kept_edges, teleport=prior)
show("2) ADAPTED: applicability filter + supersession + trust-prior Personalized PageRank", adapted)
print(f"  flagged as outdated/not in force: {flagged_outdated}")

# ---- 3. Decision gate on the ranking -------------------------------------
ranked = sorted(adapted.items(), key=lambda x: -x[1])
top, second = ranked[0], ranked[1] if len(ranked) > 1 else (None, 0.0)
values = {DOCS[d][4] for d, s in ranked if s >= 0.1}  # values held by meaningfully-ranked docs
margin = top[1] - second[1]
best_tier = min(DOCS[d][1] for d, _ in ranked)

if len(values) > 1:
    decision = "ESCALATE_TO_EXPERT (ranked sources disagree)"
elif best_tier <= 2 and top[1] >= 0.3:
    decision = "ANSWER (LLM composes answer from top sources, with citations)"
elif best_tier <= 4:
    decision = "ANSWER_WITH_GAPS (LLM answers + lists what is unverified; asks consultant for missing context)"
else:
    decision = "ESCALATE_TO_EXPERT (only low-authority evidence)"
print(f"\n3) DECISION: top={top[0]} score={top[1]:.3f} margin={margin:.3f} values={values}\n   -> {decision}")
