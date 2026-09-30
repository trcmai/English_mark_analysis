"""Web research tests with a simulated Claude session (no network, no API key)."""
from datetime import date

import pytest

from payroll_passport import KnowledgeBase, Question, ask
from payroll_passport.decision import Action
from payroll_passport.web_research import (Candidate, Retrieval, is_official, value_in_text, verify)

MARCH_2026 = date(2026, 3, 15)
OFFICIAL_URL = "https://www.rijksoverheid.nl/onderwerpen/pensioen"
BLOG_URL = "https://payroll-blog.example.com/nl-pension-2026"
PAGE_TEXT = ("Pensioen. From 1 January 2026 the employer pays at least 3.5% of the pensionable salary "
             "into the pension scheme. Employees pay the remainder.")


class Block:
    """Mimics an SDK content block: attribute access plus model_dump()."""
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self):
        def dump(v):
            if isinstance(v, Block):
                return v.model_dump()
            if isinstance(v, list):
                return [dump(x) for x in v]
            return v
        return {k: dump(v) for k, v in self.__dict__.items()}


def record(id, **inp):
    base = dict(url=OFFICIAL_URL, title="Pension (rijksoverheid)", source_type="official_guidance",
                quote="From 1 January 2026 the employer pays at least 3.5% of the pensionable salary",
                rule="pension_employer_min_pct", value=3.5, unit="%", effective_from="2026-01-01",
                effective_to="", employee_scope=["all"])
    base.update(inp)
    return Block(type="tool_use", id=id, name="record_source", input=base)


SEARCH = Block(type="web_search_tool_result", tool_use_id="s1", content=[
    Block(type="web_search_result", url=OFFICIAL_URL, title="Pensioen | Rijksoverheid", encrypted_content="xx", page_age=None),
    Block(type="web_search_result", url=BLOG_URL, title="NL pension rates explained for payroll teams", encrypted_content="yy", page_age=None),
])
FETCH = Block(type="web_fetch_tool_result", tool_use_id="f1", content=Block(
    type="web_fetch_result", url=OFFICIAL_URL, retrieved_at="2026-03-15",
    content=Block(type="document", source=Block(type="text", media_type="text/plain", data=PAGE_TEXT))))


class FakeClient:
    def __init__(self, turns):
        self.turns, self.calls = list(turns), []
        self.beta = Block(messages=Block(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.turns.pop(0)


def session(*records, stop="tool_use"):
    return FakeClient([
        Block(stop_reason=stop, content=[Block(type="server_tool_use", id="s1", name="web_search", input={}), SEARCH,
                                         Block(type="server_tool_use", id="f1", name="web_fetch", input={}), FETCH, *records]),
        Block(stop_reason="end_turn", content=[Block(type="text", text="done")]),
    ])


@pytest.fixture
def kb():
    return KnowledgeBase.load()


def pension_q():
    return Question("What is the minimum employer pension contribution?", "NL", "pension_contribution", MARCH_2026)


def test_no_source_triggers_search_and_verified_source_is_used(kb):
    client = session(
        record("t1"),                                                        # good
        record("t2", url="https://www.rijksoverheid.nl/made-up-page"),       # URL never returned
        record("t3", url=BLOG_URL, source_type="law", value=4.0,             # listed but never fetched
               quote="the employer pays 4% into the pension"),
        record("t4", quote="the employer pays at least 3% of salary", value=3.0),   # quote not on page
    )
    out = ask(kb, pension_q(), use_llm=False, use_web=True, web_client=client)

    verdicts = {v.candidate.url + v.candidate.quote[:10]: v for v in out.web.verdicts}
    accepted = [v for v in out.web.verdicts if v.accepted]
    assert len(accepted) == 1 and accepted[0].candidate.url == OFFICIAL_URL and accepted[0].official
    reasons = " ".join(c for v in out.web.verdicts if not v.accepted for c in v.checks)
    assert "never appeared in the search results" in reasons
    assert "page was not fetched" in reasons
    assert "quote not found on the fetched page" in reasons

    src = kb.sources[accepted[0].source_id]
    assert src.origin == "web" and src.type == "official_guidance" and src.url == OFFICIAL_URL
    assert out.ranking.ranked[0].source.id == src.id
    assert out.decision.action == Action.ANSWER_WITH_GAPS          # web sources never give a plain answer
    assert any("no expert has validated" in r for r in out.decision.reasons)
    assert "3.5" in out.draft.answer


def test_search_not_run_when_a_source_exists(kb):
    client = session(record("t1"))
    ask(kb, Question("min wage", "NL", "min_wage_hourly", MARCH_2026, employee_ctx="adult"),
        use_llm=False, use_web=True, web_client=client)
    assert client.calls == []


def test_unofficial_domain_is_capped_as_commentary(kb):
    fetch_blog = Block(type="web_fetch_tool_result", content=Block(type="web_fetch_result", url=BLOG_URL,
                       content=Block(type="document", source=Block(type="text", data="In 2026 the employer pays 3.5% into the pension."))))
    client = FakeClient([Block(stop_reason="tool_use", content=[SEARCH, fetch_blog,
                         record("t1", url=BLOG_URL, source_type="law", quote="In 2026 the employer pays 3.5% into the pension.")]),
                         Block(stop_reason="end_turn", content=[])])
    out = ask(kb, pension_q(), use_llm=False, use_web=True, web_client=client)
    v = out.web.accepted[0]
    assert not v.official and kb.sources[v.source_id].type == "web_commentary"
    assert out.decision.action == Action.ESCALATE          # tier 5 only: low-authority evidence


def test_disagreeing_official_sources_escalate(kb):
    page2 = "Tabel 2026: werkgeversbijdrage pensioen minimaal 4,0% van het loon."
    url2 = "https://www.belastingdienst.nl/pensioen-2026"
    search2 = Block(type="web_search_tool_result", content=[Block(type="web_search_result", url=url2, title="t")])
    fetch2 = Block(type="web_fetch_tool_result", content=Block(type="web_fetch_result", url=url2,
                   content=Block(type="document", source=Block(type="text", data=page2))))
    client = FakeClient([Block(stop_reason="tool_use", content=[SEARCH, FETCH, search2, fetch2, record("t1"),
                         record("t2", url=url2, source_type="tax_authority", value=4.0,
                                quote="werkgeversbijdrage pensioen minimaal 4,0% van het loon")]),
                         Block(stop_reason="end_turn", content=[])])
    out = ask(kb, pension_q(), use_llm=False, use_web=True, web_client=client)
    assert len(out.web.accepted) == 2
    assert out.decision.action == Action.ESCALATE and out.decision.conflicts


def test_pause_turn_is_resumed(kb):
    client = FakeClient([Block(stop_reason="pause_turn", content=[SEARCH]),
                         Block(stop_reason="tool_use", content=[FETCH, record("t1")]),
                         Block(stop_reason="end_turn", content=[])])
    out = ask(kb, pension_q(), use_llm=False, use_web=True, web_client=client)
    assert len(client.calls) == 3 and len(out.web.accepted) == 1


def test_search_failure_falls_back_to_expert(kb):
    class Broken:
        @property
        def beta(self):
            raise RuntimeError("no credentials")
    out = ask(kb, pension_q(), use_llm=False, use_web=True, web_client=Broken())
    assert out.web.error and out.decision.action == Action.ESCALATE


def test_undated_or_future_values_are_rejected():
    q = pension_q()
    retrieval = Retrieval({OFFICIAL_URL}, {OFFICIAL_URL: PAGE_TEXT})
    base = dict(url=OFFICIAL_URL, title="t", source_type="law", quote="the employer pays at least 3.5% of the pensionable salary",
                rule="r", value=3.5, unit="%", effective_to="", employee_scope=["all"])
    undated, future = verify([Candidate(effective_from="", **base), Candidate(effective_from="2027-01-01", **base)], retrieval, q)
    assert not undated.accepted and "no effective date" in " ".join(undated.checks)
    assert not future.accepted and "not in force" in " ".join(future.checks)


@pytest.mark.parametrize("value,text,expected", [
    (3.5, "at least 3,5% of salary", True),
    (5000000, "5.000.000 đồng/tháng", True),
    (5000000, "VND 5,000,000 per month", True),
    (14.0, "EUR 14.00 per hour", True),
    (14.0, "EUR 114.00 per hour", False),
    (3.5, "at least 3.55%", False),
])
def test_value_formats(value, text, expected):
    assert value_in_text(value, text) is expected


def test_official_domains():
    assert is_official("www.gov.uk".removeprefix("www."), "UK")
    assert is_official("legifrance.gouv.fr", "FR") and is_official("x.gouv.fr", "FR")
    assert not is_official("gov.uk.example.com", "UK")
    assert not is_official("rijksoverheid.nl", "DE")
