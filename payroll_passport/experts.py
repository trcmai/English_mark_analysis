"""Expert routing and escalation briefs."""
from __future__ import annotations

from .decision import Decision
from .models import Expert, Question
from .ranking import RankingResult


def route(experts: list[Expert], q: Question) -> list[Expert]:
    """Experts covering the country, topic matches first, then the least loaded."""
    eligible = [e for e in experts if q.country in e.countries]
    return sorted(eligible, key=lambda e: (q.topic not in e.topics, e.open_cases))


def brief(q: Question, result: RankingResult, decision: Decision) -> str:
    """Pre-packaged case, so the expert reviews instead of researching from scratch."""
    lines = [
        f"Question: {q.text}",
        f"Context: country={q.country} client={q.client} employee={q.employee_ctx} "
        f"date={q.on_date.isoformat()} topic={q.topic}",
        "Why escalated: " + "; ".join(decision.reasons),
    ]
    for c in decision.conflicts:
        vals = ", ".join(f"{v} <- {ids}" for v, ids in c.values.items())
        lines.append(f"Conflict on {c.rule}: {vals}")
    lines.append("Ranked evidence:")
    lines += [f"  {r.score:.3f} {r.source.id} [{r.source.type}] {r.source.title}" for r in result.ranked] or ["  (none)"]
    if result.outdated:
        lines.append("Flagged outdated:")
        lines += [f"  {sid}: {why}" for sid, why in result.outdated.items()]
    return "\n".join(lines)
