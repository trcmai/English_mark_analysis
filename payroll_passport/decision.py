"""Decision gate: turn a ranking into ANSWER / ANSWER_WITH_GAPS / ESCALATE.

Structural signals only - the LLM is never asked to grade its own confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .models import Question
from .ranking import RankingResult

MIN_EVIDENCE_SCORE = 0.10   # ranked sources below this are ignored for conflict checks
MIN_TOP_SCORE = 0.30
HIGH_RISK_TOPICS = {"termination", "cross_border_social_security"}


class Action(str, Enum):
    ANSWER = "ANSWER"
    ANSWER_WITH_GAPS = "ANSWER_WITH_GAPS"
    ESCALATE = "ESCALATE"


@dataclass
class Conflict:
    rule: str
    values: dict[str, list[str]]   # value -> source ids claiming it


@dataclass
class Decision:
    action: Action
    reasons: list[str]
    conflicts: list[Conflict] = field(default_factory=list)


def find_conflicts(result: RankingResult) -> list[Conflict]:
    """Same rule, different value, among meaningfully-ranked sources."""
    by_rule: dict[str, dict[str, list[str]]] = {}
    for r in result.ranked:
        if r.score < MIN_EVIDENCE_SCORE:
            continue
        for c in r.source.claims:
            by_rule.setdefault(c.rule, {}).setdefault(f"{c.value} {c.unit}".strip(), []).append(r.source.id)
    return [Conflict(rule, vals) for rule, vals in by_rule.items() if len(vals) > 1]


def decide(q: Question, result: RankingResult) -> Decision:
    if not result.ranked:
        return Decision(Action.ESCALATE, ["no applicable source in force on the question date"])

    conflicts = find_conflicts(result)
    top = result.ranked[0]
    best_tier = min(r.source.tier for r in result.ranked)
    top_is_current_validated = top.source.validated and (
        top.source.review_by is None or q.on_date <= top.source.review_by)

    if q.topic in HIGH_RISK_TOPICS:
        return Decision(Action.ESCALATE, [f"'{q.topic}' is a high-risk topic: expert sign-off required"], conflicts)
    if conflicts:
        return Decision(Action.ESCALATE, ["ranked sources disagree"], conflicts)
    if top.score >= MIN_TOP_SCORE and (best_tier <= 2 or top_is_current_validated):
        return Decision(Action.ANSWER, [f"top source {top.source.id} is authoritative and uncontested"])
    if best_tier <= 4:
        return Decision(Action.ANSWER_WITH_GAPS, ["only mid-authority evidence; answer must list what is unverified"])
    return Decision(Action.ESCALATE, ["only low-authority evidence (emails/notes/wiki)"])
