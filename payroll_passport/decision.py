"""Decision gate: turn a ranking into ANSWER / ANSWER_WITH_GAPS / ESCALATE.

Structural signals only - the LLM is never asked to grade its own confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .models import ANY_EMPLOYEE, Question
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


def _scopes_overlap(a: list[str], b: list[str]) -> bool:
    return ANY_EMPLOYEE in a or ANY_EMPLOYEE in b or bool(set(a) & set(b))


def find_conflicts(result: RankingResult) -> list[Conflict]:
    """Same rule, different value, among meaningfully-ranked sources covering the same employees.

    Different values for different employee groups (e.g. adult vs youth rates) are not a conflict;
    they are missing context, handled by the needs_context check.
    """
    by_rule: dict[str, list] = {}
    for r in result.ranked:
        if r.score < MIN_EVIDENCE_SCORE:
            continue
        for c in r.source.claims:
            by_rule.setdefault(c.rule, []).append((f"{c.value} {c.unit}".strip(), r.source))
    conflicts = []
    for rule, items in by_rule.items():
        clashing = set()
        for i, (va, sa) in enumerate(items):
            for vb, sb in items[i + 1:]:
                if va != vb and _scopes_overlap(sa.employee_scope, sb.employee_scope):
                    clashing |= {(va, sa.id), (vb, sb.id)}
        if clashing:
            values: dict[str, list[str]] = {}
            for v, sid in sorted(clashing):
                values.setdefault(v, []).append(sid)
            conflicts.append(Conflict(rule, values))
    return conflicts


def decide(q: Question, result: RankingResult) -> Decision:
    if not result.ranked:
        return Decision(Action.ESCALATE, ["no source applies to this country, client, employee and date"])

    conflicts = find_conflicts(result)
    top = result.ranked[0]
    best_tier = min(r.source.tier for r in result.ranked)
    top_is_current_validated = top.source.validated and (
        top.source.review_by is None or q.on_date <= top.source.review_by)

    if q.topic in HIGH_RISK_TOPICS:
        return Decision(Action.ESCALATE, [f"'{q.topic}' is a high-risk topic: expert sign-off required"], conflicts)
    if q.topic_confidence == "very_low":
        return Decision(Action.ESCALATE, [f"the question does not clearly match any payroll topic "
                                          f"(closest: {q.topic.replace('_', ' ')})"], conflicts)
    if conflicts:
        return Decision(Action.ESCALATE, ["ranked sources disagree"], conflicts)
    if top.needs_context:
        decision = Decision(Action.ANSWER_WITH_GAPS,
                            [f"top evidence only applies if {' and '.join(top.needs_context)}: confirm with the consultant"])
    elif top.score >= MIN_TOP_SCORE and (best_tier <= 2 or top_is_current_validated):
        decision = Decision(Action.ANSWER, [f"top source {top.source.id} is authoritative and uncontested"])
    elif best_tier <= 4:
        decision = Decision(Action.ANSWER_WITH_GAPS, ["only mid-authority evidence; answer must list what is unverified"])
    else:
        return Decision(Action.ESCALATE, ["only low-authority evidence (emails/notes/wiki)"])
    # Web sources are unreviewed: an expert validates before the answer counts as plain.
    if decision.action == Action.ANSWER and top.source.origin == "web":
        decision.action = Action.ANSWER_WITH_GAPS
        decision.reasons.append("based on web sources that no expert has validated yet")
    # An inferred or mixed topic can never produce a plain answer: the consultant confirms the topic.
    if q.topic_confidence != "high":
        decision.action = Action.ANSWER_WITH_GAPS
        decision.reasons.append(f"topic '{q.topic.replace('_', ' ')}' was inferred ({q.topic_note}): confirm it fits the question")
    return decision
