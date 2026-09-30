"""Detect a question's payroll topic from its wording.

Keyword matching, restricted to the topics that have sources for the question's country, so the same
words can resolve differently per country (e.g. "minimum wage" is monthly in BE/VN, hourly in NL/UK).
Longer matched phrases count more. If nothing matches, or two topics tie, no topic is returned and the
consultant is asked to rephrase: guessing a topic would pull the wrong evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# topic -> (terms, modifiers). Terms identify the topic; modifiers only add weight once a term matched.
TOPIC_TERMS: dict[str, tuple[list[str], list[str]]] = {
    "min_wage_hourly": (["minimum wage", "min wage", "minimum hourly", "national minimum wage", "nmw",
                         "living wage", "smic", "mindestlohn", "minimumloon", "wettelijk minimumloon"],
                        ["hour", "hourly", "per hour"]),
    "min_wage_monthly": (["minimum wage", "min wage", "minimum monthly", "regional minimum", "minimum salary",
                          "luong toi thieu", "lương tối thiểu", "minimumloon"],
                         ["month", "monthly", "per month", "region"]),
    "travel_allowance": (["travel allowance", "travel", "commut", "mileage", "per km", "kilomet",
                          "reiskosten", "fahrtkosten"], []),
    "holiday_allowance": (["holiday allowance", "holiday pay", "vacation pay", "vacation allowance",
                           "vakantiegeld", "vakantietoeslag"], []),
    "bonus_tax_timing": (["bonus", "tax table", "bijzonder tarief", "special rate"], ["december", "tax"]),
    "termination": (["terminat", "dismiss", "severance", "transition payment", "redundan", "fired",
                     "notice period", "layoff", "lay off", "rupture", "transitievergoeding", "kündigung"], []),
    "overtime_premium": (["overtime", "extra hours", "overwerk", "überstunden", "heures supplémentaires"], ["premium"]),
    "social_security_employer": (["social security", "social insurance", "employer contribution",
                                  "sozialversicherung", "bhxh", "cotisations"], ["employer"]),
    "meal_vouchers": (["meal voucher", "meal vouchers", "lunch voucher", "maaltijdcheque", "chèque-repas",
                       "cheque-repas", "ticket restaurant"], ["voucher"]),
    "year_end_bonus": (["year-end bonus", "year end bonus", "end-of-year bonus", "end of year bonus",
                        "eindejaarspremie", "prime de fin d'année"], []),
    "sick_pay": (["sick pay", "sick leave", "sickness", "sick", "illness", "waiting day", "statutory sick",
                  "ssp", "ziekte", "maladie", "carence"], []),
    "pension_contribution": (["pension", "auto-enrol", "auto enrol", "retirement", "pensioen"], ["employer"]),
    "thirteenth_month": (["13th month", "13th-month", "thirteenth month", "13e maand", "tet bonus", "lương tháng 13"], []),
}


@dataclass
class TopicMatch:
    topic: str | None
    matched: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)   # topics involved when no single topic was found
    reason: str = ""          # "", "no_match", "several_topics", "tie"
    method: str = "keywords"


def _hits(text: str, terms: list[str]) -> list[str]:
    # a term must start at a word boundary; stems like "terminat" or "commut" may continue
    return [t for t in terms if re.search(r"(?<![\w])" + re.escape(t), text)]


def detect_topic(question: str, available: list[str] | set[str]) -> TopicMatch:
    """Pick the question's topic. `available` = topics with sources for the question's country.

    All known topics are considered, so a question about a topic the country has no sources for is
    still recognised (and then escalated). Availability only breaks ties between topics that share
    wording, such as hourly vs monthly minimum wage.
    """
    text = " " + question.lower() + " "
    available = set(available)
    scored = []
    for topic in sorted(set(TOPIC_TERMS) | available):
        terms, modifiers = TOPIC_TERMS.get(topic, ([topic.replace("_", " ")], []))
        hits = _hits(text, terms)
        if not hits:
            continue
        mods = _hits(text, modifiers)
        scored.append((sum(len(h) for h in hits) + sum(len(m) for m in mods), topic in available, topic, hits, mods))
    if not scored:
        return TopicMatch(None, reason="no_match", candidates=sorted(available))
    scored.sort(key=lambda x: (-x[0], not x[1], x[2]))
    score, avail, topic, hits, mods = scored[0]

    # Another topic matched on its own words (not shared or overlapping wording, e.g. "bonus" inside
    # "year-end bonus"): the question mixes topics.
    def own(h):
        return [x for x in h if not any(x in y or y in x for y in hits)]
    others = [t for _, _, t, h, _ in scored[1:] if own(h)]
    if others:
        return TopicMatch(None, hits + mods, [topic] + others, reason="several_topics")
    tied = [t for s_, a, t, _, _ in scored[1:] if s_ == score and a == avail]
    if tied:
        return TopicMatch(None, hits + mods, [topic] + tied, reason="tie")
    return TopicMatch(topic, hits + mods)


def topics_for_country(sources, country: str) -> list[str]:
    return sorted({t for s in sources if s.country == country for t in s.topics})


def all_topics(sources) -> list[str]:
    return sorted(set(TOPIC_TERMS) | {t for s in sources for t in s.topics})


def resolve_topic(question: str, sources, country: str, use_llm: bool = False) -> TopicMatch:
    """Keywords first; if they find nothing, optionally ask Claude. Mixed-topic questions are never guessed."""
    match = detect_topic(question, topics_for_country(sources, country))
    if match.topic is None and match.reason == "no_match" and use_llm:
        from .llm import classify_topic
        topic = classify_topic(question, all_topics(sources))
        if topic:
            return TopicMatch(topic, [], method="claude")
    return match


def explain_unresolved(match: TopicMatch) -> str:
    names = ", ".join(t.replace("_", " ") for t in match.candidates)
    if match.reason == "several_topics":
        return f"This question covers several topics ({names}). Ask about one at a time."
    if match.reason == "tie":
        return f"The question could be about {names}. Add a word that makes the topic clear."
    return f"Could not tell the topic from the question. Mention it, for example: {names}."
