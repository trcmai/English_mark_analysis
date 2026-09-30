"""AI wrapper: phrase an answer strictly from ranked evidence.

The model may only use the evidence passed to it. It must not supply payroll facts from its own
knowledge; gaps are returned as `unverified_points` / `missing_context_questions` instead.
Without credentials (or with use_llm=False) a deterministic template answer is produced.
"""
from __future__ import annotations

import json

from pydantic import BaseModel

from .decision import Action
from .models import GENERIC_CLIENT, Question
from .ranking import RankedSource

MODEL = "claude-opus-5-5"
MAX_EVIDENCE = 6

SYSTEM_PROMPT = """You are Payroll Passport, a drafting assistant for payroll consultants.
You write the answer to a consultant's question using ONLY the evidence provided, which has already
been filtered for applicability and ranked by trust (highest first).

Rules:
- Every factual statement must come from the evidence. Cite the source ids you used.
- Never add rates, thresholds, deadlines or rules from your own knowledge. If something needed for a
  complete answer is not in the evidence, list it under unverified_points instead of filling it in.
- If the answer depends on context the consultant did not give (e.g. employee age, sector agreement),
  list the question to ask under missing_context_questions.
- Prefer the highest-ranked source; mention briefly why it is trusted (authority, date, validation).
- Be concise and practical: the reader is a payroll professional."""


class DraftAnswer(BaseModel):
    answer: str
    cited_source_ids: list[str]
    unverified_points: list[str]
    missing_context_questions: list[str]


def _evidence_block(ranked: list[RankedSource]) -> str:
    items = []
    for r in ranked[:MAX_EVIDENCE]:
        s = r.source
        items.append({
            "id": s.id, "title": s.title, "type": s.type, "authority_tier": s.tier,
            "rank_score": round(r.score, 3), "effective_from": s.effective_from.isoformat(),
            "effective_to": s.effective_to.isoformat() if s.effective_to else None,
            "claims": [{"rule": c.rule, "value": c.value, "unit": c.unit} for c in s.claims],
            "text": s.text, "notes": r.notes,
        })
    return json.dumps(items, indent=2)


def template_answer(q: Question, ranked: list[RankedSource], action: Action) -> DraftAnswer:
    """Deterministic fallback: restate the top source's claims. No generation involved."""
    def facts(s):
        return "; ".join(f"{c.rule} = {c.value} {c.unit}".strip() for c in s.claims) or s.text

    top = ranked[0].source
    answer = f"According to {top.title} ({top.id}): {facts(top)}."
    cited = [top.id]
    # Client-specific terms must be surfaced even when general guidance ranks higher:
    # citation flow pushes score toward the general source the contract cites.
    for r in ranked[1:]:
        s = r.source
        if s.client == q.client and s.client != GENERIC_CLIENT:
            answer += f" Client-specific terms in {s.title} ({s.id}): {facts(s)}."
            cited.append(s.id)
    # Employee group not given: show what applies to the other groups too.
    if ranked[0].needs_context:
        for r in ranked[1:]:
            s = r.source
            if r.needs_context and s.employee_scope != top.employee_scope and s.claims:
                answer += f" For {' or '.join(s.employee_scope)} employees, {s.title} ({s.id}): {facts(s)}."
                cited.append(s.id)
    unverified = []
    missing = [f"Confirm that the {n}." for n in ranked[0].needs_context]
    if action == Action.ANSWER_WITH_GAPS and not missing:
        unverified.append(f"Top evidence is only authority tier {top.tier} ({top.type}); "
                          "confirm against legislation or an expert.")
    return DraftAnswer(
        answer=answer,
        cited_source_ids=cited,
        unverified_points=unverified,
        missing_context_questions=missing,
    )


def draft_answer(q: Question, ranked: list[RankedSource], action: Action, use_llm: bool = True) -> tuple[DraftAnswer, str]:
    """Return (draft, mode) where mode is 'llm' or 'template'."""
    if not use_llm:
        return template_answer(q, ranked, action), "template"
    try:
        import anthropic
        client = anthropic.Anthropic()
        user = (
            f"Question: {q.text}\n"
            f"Context: country={q.country}, client={q.client}, employee={q.employee_ctx}, "
            f"date={q.on_date.isoformat()}, topic={q.topic}\n"
            f"Gate decision: {action.value}\n\n"
            f"Evidence (ranked, highest trust first):\n{_evidence_block(ranked)}"
        )
        response = client.beta.messages.parse(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
            output_format=DraftAnswer,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            return template_answer(q, ranked, action), "template"
        draft = response.parsed_output
    except Exception as exc:  # no credentials, network, API error -> stay usable offline
        draft = template_answer(q, ranked, action)
        draft.unverified_points.append(f"(Claude unavailable - {type(exc).__name__}: {str(exc)[:120]}; showing template answer)")
        return draft, "template"

    # Citation guard: the model may only cite evidence it was given.
    allowed = {r.source.id for r in ranked[:MAX_EVIDENCE]}
    bogus = [c for c in draft.cited_source_ids if c not in allowed]
    if bogus or not draft.cited_source_ids:
        draft.cited_source_ids = [c for c in draft.cited_source_ids if c in allowed]
        draft.unverified_points.append(
            "Answer contained citations outside the evidence set or no citations; verify before use.")
    return draft, "llm"


class TopicChoice(BaseModel):
    topic: str


def classify_topic(question: str, topics: list[str]) -> str | None:
    """Ask Claude to map a question to one known topic. Returns None if unavailable or unsure."""
    try:
        import anthropic
        response = anthropic.Anthropic().beta.messages.parse(
            model=MODEL,
            max_tokens=1000,
            system=("Classify a payroll consultant's question into exactly one topic from the list, or answer "
                    "'none' if it fits none of them or covers several. Reply with the topic id only."),
            messages=[{"role": "user", "content": f"Topics: {', '.join(topics)}\n\nQuestion: {question}"}],
            output_format=TopicChoice,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except Exception:
        return None
    if response.stop_reason == "refusal" or response.parsed_output is None:
        return None
    topic = response.parsed_output.topic.strip()
    return topic if topic in topics else None
