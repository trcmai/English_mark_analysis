"""Knowledge capture: turn an expert-validated answer into a reusable knowledge item."""
from __future__ import annotations

from datetime import date, timedelta

from .knowledge_base import KnowledgeBase
from .models import Claim, Question, Source

REVIEW_PERIOD = timedelta(days=365)   # payroll rules change yearly; validated knowledge decays too


def capture(kb: KnowledgeBase, q: Question, expert_id: str, answer: str,
            claims: list[Claim], evidence_ids: list[str], supersedes: list[str] = (),
            today: date | None = None) -> Source:
    today = today or date.today()
    if expert_id not in {e.id for e in kb.experts}:
        raise ValueError(f"unknown expert: {expert_id}")
    missing = [i for i in evidence_ids if i not in kb.sources]
    if missing:
        raise ValueError(f"unknown evidence ids: {missing}")
    if not evidence_ids:
        raise ValueError("a knowledge item must cite at least one evidence source")

    n = sum(1 for s in kb.sources.values() if s.type == "knowledge") + 1
    item = Source(
        id=f"KI_{q.country}_{q.topic}_{n}",
        title=f"Validated answer: {q.text}",
        type="knowledge",
        country=q.country,
        topics=[q.topic],
        effective_from=q.on_date,
        client=q.client,
        employee_scope=[q.employee_ctx],
        claims=claims,
        text=answer,
        validated_by=expert_id,
        review_by=today + REVIEW_PERIOD,
    )
    kb.add_source(item, cites=evidence_ids, supersedes=list(supersedes))
    return item
