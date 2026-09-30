import sys
import types
from datetime import date

import pytest

from payroll_passport import KnowledgeBase, Question, ask
from payroll_passport.capture import capture
from payroll_passport.decision import Action
from payroll_passport.llm import DraftAnswer, draft_answer
from payroll_passport.models import Claim
from payroll_passport.ranking import personalized_pagerank, rank_evidence

MARCH_2026 = date(2026, 3, 15)


@pytest.fixture
def kb():
    return KnowledgeBase.load()


def q(topic, on=MARCH_2026, **kw):
    return Question(text=f"question about {topic}", country="NL", topic=topic, on_date=on, **kw)


def test_pagerank_sums_to_one_and_respects_prior():
    scores = personalized_pagerank(["a", "b"], [], {"a": 0.8, "b": 0.2})
    assert sum(scores.values()) == pytest.approx(1.0)
    assert scores["a"] > scores["b"]


def test_new_law_beats_heavily_cited_old_law(kb):
    result = rank_evidence(kb, q("min_wage_hourly", employee_ctx="adult"))
    assert result.ranked[0].source.id == "NL_LAW_MINWAGE_2026"
    assert "NL_LAW_MINWAGE_2025" in result.outdated
    # stale dependency: docs that only cite the old law are flagged too
    assert {"INT_GUIDE_2024", "EMAIL_A", "WIKI_MINWAGE"} <= set(result.outdated)


def test_historic_date_uses_law_in_force_then(kb):
    result = rank_evidence(kb, q("min_wage_hourly", on=date(2025, 6, 1), employee_ctx="adult"))
    assert result.ranked[0].source.id == "NL_LAW_MINWAGE_2025"


def test_staleness_does_not_cross_topics(kb):
    result = rank_evidence(kb, q("holiday_allowance", client="ClientX"))
    assert "INT_GUIDE_2024" in {r.source.id for r in result.ranked}
    assert "NL_LAW_MINWAGE_2025" not in result.outdated


def test_client_isolation(kb):
    result = rank_evidence(kb, q("holiday_allowance", client="ClientX"))
    ids = {r.source.id for r in result.ranked}
    assert "CLIENTX_CONTRACT" in ids
    assert "CLIENTY_CONTRACT" not in ids
    assert result.not_applicable["CLIENTY_CONTRACT"] == "belongs to another client"


def test_client_terms_surface_in_answer(kb):
    out = ask(kb, q("holiday_allowance", client="ClientX"), use_llm=False)
    assert out.decision.action == Action.ANSWER
    assert "CLIENTX_CONTRACT" in out.draft.cited_source_ids
    assert "8.5" in out.draft.answer


def test_answer_when_authoritative(kb):
    out = ask(kb, q("min_wage_hourly", employee_ctx="adult"), use_llm=False)
    assert out.decision.action == Action.ANSWER
    assert "14.0" in out.draft.answer


def test_conflict_escalates(kb):
    out = ask(kb, q("travel_allowance"), use_llm=False)
    assert out.decision.action == Action.ESCALATE
    assert out.decision.conflicts[0].rule == "travel_allowance_per_km"
    assert out.experts[0].id == "EXP_NL_1"
    assert "Conflict on travel_allowance_per_km" in out.expert_brief


def test_low_authority_escalates(kb):
    assert ask(kb, q("bonus_tax_timing"), use_llm=False).decision.action == Action.ESCALATE


def test_high_risk_topic_escalates_even_with_law(kb):
    out = ask(kb, q("termination"), use_llm=False)
    assert out.decision.action == Action.ESCALATE
    assert out.experts[0].id == "EXP_NL_2"


def test_no_source_escalates(kb):
    out = ask(kb, q("pension_contribution"), use_llm=False)
    assert out.decision.action == Action.ESCALATE
    assert out.decision.reasons == ["no source applies to this country, client, employee and date"]


def test_closed_loop_escalate_validate_answer(kb, tmp_path):
    question = q("travel_allowance")
    assert ask(kb, question, use_llm=False).decision.action == Action.ESCALATE

    item = capture(kb, question, "EXP_NL_1", "Use EUR 0.23/km; the ministry FAQ is outdated.",
                   [Claim("travel_allowance_per_km", 0.23, "EUR")],
                   evidence_ids=["NL_TAX_TRAVEL_2026"], supersedes=["NL_GUIDANCE_TRAVEL_2026"],
                   today=MARCH_2026)
    path = tmp_path / "kb.json"
    kb.save(path)
    reloaded = KnowledgeBase.load(path)

    out = ask(reloaded, question, use_llm=False)
    assert out.decision.action == Action.ANSWER
    assert "NL_GUIDANCE_TRAVEL_2026" in out.ranking.outdated
    assert item.id in {r.source.id for r in out.ranking.ranked}
    assert reloaded.sources[item.id].review_by == date(2027, 3, 15)


def test_capture_requires_evidence_and_known_expert(kb):
    with pytest.raises(ValueError):
        capture(kb, q("travel_allowance"), "EXP_NL_1", "x", [], evidence_ids=[])
    with pytest.raises(ValueError):
        capture(kb, q("travel_allowance"), "NOBODY", "x", [], evidence_ids=["NL_TAX_TRAVEL_2026"])


def test_expired_validation_loses_bonus(kb):
    after_review = date(2027, 2, 1)
    result = rank_evidence(kb, q("min_wage_hourly", on=after_review, employee_ctx="adult"))
    ki = next(r for r in result.ranked if r.source.id == "KI_NL_MINWAGE_2026")
    assert any("past its review date" in n for n in ki.notes)


def _fake_anthropic(monkeypatch, parsed):
    class FakeMessages:
        def parse(self, **kwargs):
            return types.SimpleNamespace(stop_reason="end_turn", parsed_output=parsed)

    class FakeClient:
        def __init__(self):
            self.beta = types.SimpleNamespace(messages=FakeMessages())

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=FakeClient))


def test_llm_citation_guard_drops_unknown_sources(kb, monkeypatch):
    _fake_anthropic(monkeypatch, DraftAnswer(
        answer="EUR 14.00", cited_source_ids=["NL_LAW_MINWAGE_2026", "MADE_UP"],
        unverified_points=[], missing_context_questions=[]))
    ranked = rank_evidence(kb, q("min_wage_hourly", employee_ctx="adult")).ranked
    draft, mode = draft_answer(q("min_wage_hourly"), ranked, Action.ANSWER)
    assert mode == "llm"
    assert draft.cited_source_ids == ["NL_LAW_MINWAGE_2026"]
    assert draft.unverified_points


def test_llm_failure_falls_back_to_template(kb, monkeypatch):
    class Boom:
        def __init__(self):
            raise RuntimeError("no credentials")

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Boom))
    ranked = rank_evidence(kb, q("min_wage_hourly", employee_ctx="adult")).ranked
    draft, mode = draft_answer(q("min_wage_hourly"), ranked, Action.ANSWER)
    assert mode == "template"
    assert "14.0" in draft.answer


def test_unspecified_employee_does_not_fall_back_to_outdated_guide(kb):
    out = ask(kb, q("min_wage_hourly"), use_llm=False)   # employee context not given
    assert out.ranking.ranked[0].source.id == "NL_LAW_MINWAGE_2026"
    assert "INT_GUIDE_2024" in out.ranking.outdated
    assert out.decision.action == Action.ANSWER_WITH_GAPS
    assert "14.0" in out.draft.answer
    assert out.draft.missing_context_questions == ["Confirm that the employee is adult."]


def test_outdated_and_not_applicable_are_disjoint(kb):
    result = rank_evidence(kb, q("min_wage_hourly", employee_ctx="adult"))
    other = rank_evidence(kb, Question("x", "DE", "min_wage_hourly", MARCH_2026))
    assert not set(result.outdated) & set(result.not_applicable)
    assert not set(other.outdated) & set(other.not_applicable)


def test_different_employee_groups_are_not_a_conflict(kb):
    out = ask(kb, Question("overtime?", "DE", "overtime_premium", MARCH_2026), use_llm=False)
    assert out.decision.conflicts == []
    assert out.decision.action == Action.ANSWER_WITH_GAPS
    assert "office_staff" in out.draft.answer          # other group's value is listed too


def test_sector_agreement_answers_for_its_sector(kb):
    out = ask(kb, Question("overtime?", "DE", "overtime_premium", MARCH_2026, employee_ctx="metal_sector"), use_llm=False)
    assert out.decision.action == Action.ANSWER
    assert "25.0" in out.draft.answer


def test_same_group_disagreement_still_escalates(kb):
    out = ask(kb, Question("waiting days?", "FR", "sick_pay", MARCH_2026), use_llm=False)
    assert out.decision.action == Action.ESCALATE
    assert out.experts[0].id == "EXP_FR_1"


def test_uk_rate_depends_on_pay_date(kb):
    before = ask(kb, Question("ssp?", "UK", "sick_pay", MARCH_2026), use_llm=False)
    after = ask(kb, Question("ssp?", "UK", "sick_pay", date(2026, 5, 1)), use_llm=False)
    assert "110.0" in before.draft.answer and "120.0" in after.draft.answer
    nmw = ask(kb, Question("nmw?", "UK", "min_wage_hourly", MARCH_2026, employee_ctx="21_plus"), use_llm=False)
    assert "11.5" in nmw.draft.answer


def test_every_country_has_an_expert(kb):
    countries = {s.country for s in kb.sources.values()}
    covered = {c for e in kb.experts for c in e.countries}
    assert countries <= covered


from payroll_passport.topics import detect_topic, resolve_topic, topics_for_country


@pytest.mark.parametrize("country,question,topic", [
    ("NL", "What is the adult minimum hourly wage?", "min_wage_hourly"),
    ("VN", "What is the minimum wage?", "min_wage_monthly"),        # same words, country decides
    ("BE", "What is the minimum wage?", "min_wage_monthly"),
    ("NL", "Can we reimburse commuting costs?", "travel_allowance"),
    ("FR", "What is the SMIC?", "min_wage_hourly"),
    ("BE", "Do we pay a year-end bonus?", "year_end_bonus"),          # "bonus" inside a longer phrase
    ("NL", "Is the year-end bonus taxed at the special rate?", "bonus_tax_timing"),
    ("NL", "What is the pension contribution?", "pension_contribution"),  # known topic, no NL sources
])
def test_topic_detected_from_question(kb, country, question, topic):
    assert detect_topic(question, topics_for_country(kb.sources.values(), country)).topic == topic


def test_mixed_topics_use_strongest_and_name_the_other(kb):
    m = detect_topic("sick leave and termination rules", topics_for_country(kb.sources.values(), "FR"))
    assert m.topic == "sick_pay" and m.confidence == "medium"
    assert m.also_mentions == ["termination"]


@pytest.mark.parametrize("country,question,topic", [
    ("NL", "minimun wage for adults", "min_wage_hourly"),              # typo
    ("DE", "Extra pay for weekend work?", "overtime_premium"),
    ("VN", "extra month salary before lunar new year", "thirteenth_month"),
    ("NL", "Can staff claim driving costs?", "travel_allowance"),
])
def test_no_keyword_falls_back_to_most_similar_topic(kb, country, question, topic):
    m = detect_topic(question, topics_for_country(kb.sources.values(), country))
    assert m.method == "similarity" and m.topic == topic and m.confidence in ("medium", "low")


def test_inferred_topic_never_gives_a_plain_answer(kb):
    m = resolve_topic("minimun wage for adults", kb.sources.values(), "NL")
    q = Question("minimun wage", "NL", m.topic, MARCH_2026, employee_ctx="adult",
                 topic_confidence=m.confidence, topic_note="similarity")
    out = ask(kb, q, use_llm=False)
    assert out.decision.action == Action.ANSWER_WITH_GAPS
    assert any("inferred" in r for r in out.decision.reasons)


def test_unrelated_question_is_escalated(kb):
    m = resolve_topic("hello", kb.sources.values(), "NL")
    assert m.topic and m.confidence == "very_low"
    q = Question("hello", "NL", m.topic, MARCH_2026, topic_confidence=m.confidence)
    assert ask(kb, q, use_llm=False).decision.action == Action.ESCALATE


def test_detected_topic_without_country_sources_escalates(kb):
    m = resolve_topic("What is the pension contribution?", kb.sources.values(), "NL")
    out = ask(kb, Question("pension?", "NL", m.topic, MARCH_2026), use_llm=False)
    assert out.decision.action == Action.ESCALATE


def test_generic_payroll_words_do_not_steer_similarity(kb):
    m = detect_topic("An employee is off for two weeks with the flu. What do we pay?", topics_for_country(kb.sources.values(), "UK"))
    assert m.topic == "sick_pay"


def test_inferred_topic_note_does_not_blame_strong_evidence(kb):
    q = Question("flu", "UK", "sick_pay", date(2026, 5, 1), topic_confidence="low", topic_note="similarity")
    out = ask(kb, q, use_llm=False)
    assert out.decision.action == Action.ANSWER_WITH_GAPS
    assert not any("authority tier" in u for u in out.draft.unverified_points)
    assert any("inferred" in u for u in out.draft.unverified_points)
