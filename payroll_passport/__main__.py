"""Command line interface.

  python -m payroll_passport ask "What is the adult minimum wage?" --country NL --date 2026-03-15
  python -m payroll_passport validate --kb my_kb.json --expert EXP_NL_1 --question "..." --country NL \
      --topic travel_allowance --date 2026-03-15 --answer "..." --claim travel_allowance_per_km=0.23:EUR \
      --evidence NL_TAX_TRAVEL_2026
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .assistant import ask, render
from .capture import capture
from .knowledge_base import DEFAULT_KB, KnowledgeBase
from .models import Claim, Question
from .topics import describe, resolve_topic


def _question(a, use_llm=False) -> Question:
    topic = a.topic
    if not topic:
        kb = KnowledgeBase.load(a.kb)
        match = resolve_topic(a.question, kb.sources.values(), a.country, use_llm)
        topic, confidence, note = match.topic, match.confidence, describe(match)
        alts = ", ".join(t.replace("_", " ") for t in match.alternatives)
        print(f"Topic: {topic.replace('_', ' ')} ({confidence.replace('_', ' ')} confidence) - {note}" + (f"\n  Other close topics: {alts} (use --topic)" if confidence != "high" else "") + "\n")
    else:
        confidence, note = "high", "chosen with --topic"
    return Question(text=a.question, country=a.country, topic=topic, topic_confidence=confidence, topic_note=note,
                    on_date=date.fromisoformat(a.date), client=a.client, employee_ctx=a.employee)


def _claim(spec: str) -> Claim:
    rule, _, rest = spec.partition("=")
    value, _, unit = rest.partition(":")
    try:
        parsed: float | str = float(value)
    except ValueError:
        parsed = value
    return Claim(rule, parsed, unit)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="payroll_passport")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, positional_question):
        if positional_question:
            sp.add_argument("question")
        else:
            sp.add_argument("--question", required=True)
        sp.add_argument("--kb", default=str(DEFAULT_KB))
        sp.add_argument("--country", required=True)
        sp.add_argument("--topic", help="topic id; detected from the question when omitted")
        sp.add_argument("--date", default=date.today().isoformat())
        sp.add_argument("--client", default="GENERIC")
        sp.add_argument("--employee", default="all")

    a_ask = sub.add_parser("ask", help="answer a question or route it to an expert")
    common(a_ask, True)
    a_ask.add_argument("--no-llm", action="store_true", help="use the deterministic template answer")

    a_val = sub.add_parser("validate", help="store an expert-validated answer as knowledge")
    common(a_val, False)
    a_val.add_argument("--expert", required=True)
    a_val.add_argument("--answer", required=True)
    a_val.add_argument("--claim", action="append", default=[], help="rule=value[:unit]")
    a_val.add_argument("--evidence", action="append", required=True, help="source id backing the answer")
    a_val.add_argument("--supersedes", action="append", default=[])

    a = p.parse_args(argv)
    if a.cmd == "ask":
        kb = KnowledgeBase.load(a.kb)
        print(render(ask(kb, _question(a, use_llm=not a.no_llm), use_llm=not a.no_llm)))
    else:
        kb_path = Path(a.kb)
        if kb_path.resolve() == DEFAULT_KB.resolve():
            raise SystemExit("refusing to modify the bundled sample; copy it and pass --kb <your file>")
        kb = KnowledgeBase.load(kb_path)
        item = capture(kb, _question(a), a.expert, a.answer,
                       [_claim(c) for c in a.claim], a.evidence, a.supersedes)
        kb.save(kb_path)
        print(f"Stored {item.id} (validated by {item.validated_by}, review by {item.review_by})")


if __name__ == "__main__":
    main()
