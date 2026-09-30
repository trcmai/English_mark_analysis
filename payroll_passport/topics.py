"""Detect a question's payroll topic from its wording.

Every question is related to its most similar topic, with a confidence level:
  high   - payroll terms of exactly one topic appear in the question (longer phrases count more; the
           question's country breaks ties such as hourly vs monthly minimum wage)
  medium - the question mixes topics or ties; the strongest match is used and the others are named
  medium/low - no terms match: the closest topic by word-fragment similarity to each topic's description
           (tolerates typos and related wording). Low confidence answers must not be taken at face value.
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
                     "notice period", "layoff", "lay off", "let go", "let someone go", "rupture", "transitievergoeding", "kündigung"], []),
    "overtime_premium": (["overtime", "extra hours", "overwerk", "überstunden", "heures supplémentaires"], ["premium"]),
    "social_security_employer": (["social security", "social insurance", "employer contribution",
                                  "sozialversicherung", "bhxh", "cotisations"], ["employer"]),
    "meal_vouchers": (["meal voucher", "meal vouchers", "lunch voucher", "maaltijdcheque", "chèque-repas",
                       "cheque-repas", "ticket restaurant", "lunch"], ["voucher"]),
    "year_end_bonus": (["year-end bonus", "year end bonus", "end-of-year bonus", "end of year bonus",
                        "eindejaarspremie", "prime de fin d'année"], []),
    "sick_pay": (["sick pay", "sick leave", "sickness", "sick", "illness", "waiting day", "statutory sick",
                  "ssp", "is ill", "falls ill", "fell ill", "off sick", "ziekte", "maladie", "carence"], []),
    "pension_contribution": (["pension", "auto-enrol", "auto enrol", "retirement", "pensioen"], ["employer"]),
    "thirteenth_month": (["13th month", "13th-month", "thirteenth month", "13e maand", "tet bonus", "lương tháng 13"], []),
}


# Plain-language descriptions used for similarity matching when no term matches.
TOPIC_DESCRIPTIONS: dict[str, str] = {
    "min_wage_hourly": "lowest legal pay per hour, statutory minimum hourly rate, how little can we pay an hour, wage floor",
    "min_wage_monthly": "lowest legal salary per month, statutory minimum monthly pay, regional wage floor",
    "travel_allowance": "reimburse travel costs, commuting, driving to work, car kilometres, business trip, tax-free travel expense",
    "holiday_allowance": "holiday money, vacation bonus paid once a year, percentage of annual salary for holidays",
    "bonus_tax_timing": "how a bonus or one-off payment is taxed, which withholding table or rate, special payments",
    "termination": "end of employment, let someone go, fire, quit, resign, leaving employee, exit payment, final pay, notice",
    "overtime_premium": "extra pay for working more hours, weekend or night work surcharge, additional hours rate",
    "social_security_employer": "employer contributions, payroll taxes, national insurance, health and unemployment insurance charges",
    "meal_vouchers": "lunch vouchers, food benefit, meal cheques, employer share per working day",
    "year_end_bonus": "end of year premium, december bonus required by sector agreement",
    "sick_pay": "employee is ill or unwell, off work with flu or a cold, sickness absence, continued pay while ill, "
                "waiting days, doctor note, medical certificate",
    "pension_contribution": "retirement savings, pension scheme, employer pension percentage, workplace pension",
    "thirteenth_month": "extra month of salary, thirteenth salary, lunar new year payment",
}

STOPWORDS = {"a", "an", "the", "is", "are", "was", "be", "to", "of", "in", "on", "for", "and", "or", "what", "how",
             "do", "does", "we", "i", "our", "my", "it", "this", "that", "with", "can", "should", "much", "many",
             "which", "when", "who", "there", "any", "at", "by", "per", "from", "about", "me", "you", "if", "as",
             # payroll words that appear in almost every question and say nothing about the topic
             "employee", "employees", "employer", "employers", "staff", "worker", "workers", "member",
             "pay", "paid", "payroll", "company", "client"}
SIMILAR_MEDIUM = 0.30     # similarity at or above this is medium confidence
SIMILAR_MIN = 0.10        # below this the question resembles no topic: very_low, escalated


@dataclass
class TopicMatch:
    topic: str | None
    matched: list[str] = field(default_factory=list)     # words that decided it (keyword method)
    alternatives: list[str] = field(default_factory=list)  # next most similar topics, to switch to
    method: str = "keywords"      # keywords | similarity | claude | chosen
    confidence: str = "high"      # high | medium | low | very_low
    similarity: float = 0.0
    also_mentions: list[str] = field(default_factory=list)  # other topics the question touches


def _hits(text: str, terms: list[str]) -> list[str]:
    # a term must start at a word boundary; stems like "terminat" or "commut" may continue
    return [t for t in terms if re.search(r"(?<![\w])" + re.escape(t), text)]


def _trigrams(text: str) -> dict[str, int]:
    grams: dict[str, int] = {}
    for tok in re.findall(r"[^\W_]+", text.lower()):
        if tok in STOPWORDS:
            continue
        padded = f" {tok} "
        for i in range(len(padded) - 2):
            g = padded[i:i + 3]
            grams[g] = grams.get(g, 0) + 1
    return grams


def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _profile(topic: str) -> str:
    terms, mods = TOPIC_TERMS.get(topic, ([], []))
    return " ".join([topic.replace("_", " "), *terms, *mods, TOPIC_DESCRIPTIONS.get(topic, "")])


def similarity_ranking(question: str, topics) -> list[tuple[float, str]]:
    q = _trigrams(question)
    return sorted(((round(_cosine(q, _trigrams(_profile(t))), 6), t) for t in topics), key=lambda x: (-x[0], x[1]))


def detect_topic(question: str, available: list[str] | set[str]) -> TopicMatch:
    """Relate the question to its most similar topic. `available` = topics with sources for its country.

    All known topics are considered, so a question about a topic the country has no sources for is
    still recognised (and then escalated). Availability breaks ties between topics that share wording.
    """
    text = " " + question.lower() + " "
    available = set(available)
    universe = sorted(set(TOPIC_TERMS) | available)
    sims = similarity_ranking(question, universe)
    sim_of = {t: v for v, t in sims}
    scored = []
    for topic in universe:
        terms, modifiers = TOPIC_TERMS.get(topic, ([topic.replace("_", " ")], []))
        hits = _hits(text, terms)
        if not hits:
            continue
        mods = _hits(text, modifiers)
        scored.append((sum(len(h) for h in hits) + sum(len(m) for m in mods), topic in available, topic, hits, mods))

    if not scored:
        # No payroll terms: fall back to the closest description. Prefer topics the country has sources for
        # only when similarity is equal.
        sims.sort(key=lambda x: (-x[0], x[1] not in available, x[1]))
        best_sim, best = sims[0]
        conf = "medium" if best_sim >= SIMILAR_MEDIUM else "low" if best_sim >= SIMILAR_MIN else "very_low"
        return TopicMatch(best, [], [t for _, t in sims[1:3]], "similarity", conf, best_sim)

    # Strongest keyword match; ties broken by country availability, then similarity.
    scored.sort(key=lambda x: (-x[0], not x[1], -sim_of[x[2]], x[2]))
    score, avail, topic, hits, mods = scored[0]

    def own(h):  # words not shared with / contained in the winner's words ("bonus" in "year-end bonus")
        return [x for x in h if not any(x in y or y in x for y in hits)]
    others = [t for _, _, t, h, _ in scored[1:] if own(h)]
    tied = [t for s_, a, t, _, _ in scored[1:] if s_ == score and a == avail]
    alternatives = [t for _, t in sims if t != topic][:2]
    confidence = "medium" if (others or tied) else "high"
    return TopicMatch(topic, hits + mods, alternatives, "keywords", confidence, sim_of[topic], others)


def topics_for_country(sources, country: str) -> list[str]:
    return sorted({t for s in sources if s.country == country for t in s.topics})


def all_topics(sources) -> list[str]:
    return sorted(set(TOPIC_TERMS) | {t for s in sources for t in s.topics})


def resolve_topic(question: str, sources, country: str, use_llm: bool = False) -> TopicMatch:
    """Keywords first. Without a keyword match, Claude (if enabled) classifies; otherwise similarity decides."""
    match = detect_topic(question, topics_for_country(sources, country))
    if match.method == "similarity" and use_llm:
        from .llm import classify_topic
        topic = classify_topic(question, all_topics(sources))
        if topic:
            return TopicMatch(topic, [], [t for t in match.alternatives if t != topic][:2], "claude", "medium")
    return match


def describe(match: TopicMatch) -> str:
    """One line saying how the topic was chosen, for the consultant."""
    if match.method == "keywords":
        text = "detected from " + ", ".join(f'"{m}"' for m in match.matched)
        if match.also_mentions:
            text += "; the question also mentions " + ", ".join(t.replace("_", " ") for t in match.also_mentions) + \
                    ", ask about that separately"
        return text
    if match.method == "similarity":
        return f"closest match to the wording (similarity {match.similarity:.2f})"
    if match.method == "claude":
        return "classified by Claude"
    return "chosen by the consultant"
