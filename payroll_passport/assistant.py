"""End-to-end pipeline: question -> ranked evidence -> gate -> answer or expert escalation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .decision import Action, Decision, decide
from .experts import brief, route
from .knowledge_base import KnowledgeBase
from .llm import DraftAnswer, draft_answer
from .models import Expert, Question
from .ranking import RankingResult, rank_evidence
from .web_research import WebResearch, research


@dataclass
class Outcome:
    question: Question
    ranking: RankingResult
    decision: Decision
    draft: Optional[DraftAnswer] = None
    draft_mode: str = ""
    experts: list[Expert] = None
    expert_brief: str = ""
    web: Optional[WebResearch] = None      # set when a web search was run


def ask(kb: KnowledgeBase, q: Question, use_llm: bool = True, use_web: bool = False, web_client=None) -> Outcome:
    """use_web: when no source applies, search the web, verify what is found, and rank it with the rest."""
    ranking = rank_evidence(kb, q)
    web = None
    if not ranking.ranked and use_web:
        web = research(kb, q, web_client)
        if web.accepted:
            ranking = rank_evidence(kb, q)
    decision = decide(q, ranking)
    out = Outcome(q, ranking, decision, experts=[], web=web)
    if decision.action in (Action.ANSWER, Action.ANSWER_WITH_GAPS):
        out.draft, out.draft_mode = draft_answer(q, ranking.ranked, decision.action, use_llm)
    else:
        out.experts = route(kb.experts, q)
        out.expert_brief = brief(q, ranking, decision)
    return out


def render(out: Outcome) -> str:
    lines = [f"DECISION: {out.decision.action.value} - " + "; ".join(out.decision.reasons), ""]
    lines.append("Ranked evidence (trust-weighted PageRank):")
    for r in out.ranking.ranked:
        lines.append(f"  {r.score:.3f}  {r.source.id:<22} {r.source.title}  [{'; '.join(r.notes)}]")
    if not out.ranking.ranked:
        lines.append("  (none)")
    if out.ranking.outdated:
        lines.append("Flagged outdated:")
        lines += [f"  {sid}: {why}" for sid, why in out.ranking.outdated.items()]
    if out.ranking.not_applicable:
        lines.append("Not applicable:")
        lines += [f"  {sid}: {why}" for sid, why in out.ranking.not_applicable.items()]
    if out.web:
        lines.append("Web research:" + (f" {out.web.error}" if out.web.error else ""))
        for v in out.web.verdicts:
            lines.append(f"  {'ACCEPTED' if v.accepted else 'REJECTED'} {v.candidate.url}"
                         + (f" -> {v.source_id}" if v.source_id else ""))
            lines += [f"    {c}" for c in v.checks]
        if not out.web.verdicts and not out.web.error:
            lines.append("  nothing usable found")
    lines.append("")
    if out.draft:
        lines.append(f"ANSWER ({out.draft_mode}):")
        lines.append(f"  {out.draft.answer}")
        lines.append(f"  Sources: {', '.join(out.draft.cited_source_ids) or '(none)'}")
        for p in out.draft.unverified_points:
            lines.append(f"  UNVERIFIED: {p}")
        for p in out.draft.missing_context_questions:
            lines.append(f"  ASK CONSULTANT: {p}")
    else:
        lines.append("ROUTE TO EXPERT: " + (", ".join(f"{e.name} ({e.id})" for e in out.experts)
                                            or "no expert covers this country - add one"))
        lines.append("")
        lines.append(out.expert_brief)
    return "\n".join(lines)
