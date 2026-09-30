"""Web research when the knowledge base has no applicable source.

Claude searches and reads pages (web_search / web_fetch server tools) and records each rule it finds
with the record_source tool, including a verbatim quote. Nothing Claude records is trusted as is: every
candidate goes through verify(), which is plain code:

  1. retrieved   - the URL appeared in this session's search or fetch results (catches invented links)
  2. quoted      - the quote appears in the fetched page text (catches invented or altered quotes)
  3. value       - the recorded value appears in the quote (catches mis-extracted numbers)
  4. dated       - an effective date is stated, and the source is in force on the question's date
  5. domain      - official government domains keep their document type; any other site is capped at
                   "web_commentary" (tier 5)
  6. agreement   - how many independent domains state the same value (conflicts are caught later by the
                   normal conflict check)

Accepted sources are added to the knowledge base with origin="web". They can never produce a plain
"Answer" until an expert validates the answer (see decision.py).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlparse

from .knowledge_base import KnowledgeBase
from .models import ANY_EMPLOYEE, Claim, Question, Source

MODEL = "claude-opus-5-5"
MAX_TURNS = 8
MAX_SOURCES = 5

# Official domains: exact registrable domains per country, plus government suffixes.
OFFICIAL_DOMAINS = {
    "NL": ["belastingdienst.nl", "rijksoverheid.nl", "government.nl", "overheid.nl", "uwv.nl", "svb.nl"],
    "DE": ["bund.de", "bmas.de", "gesetze-im-internet.de", "deutsche-rentenversicherung.de", "zoll.de",
           "bundesfinanzministerium.de", "arbeitsagentur.de"],
    "BE": ["belgium.be", "fgov.be", "socialsecurity.be", "onss.be", "rsz.be", "emploi.belgique.be"],
    "FR": ["gouv.fr", "service-public.fr", "legifrance.gouv.fr", "urssaf.fr", "ameli.fr"],
    "UK": ["gov.uk", "legislation.gov.uk"],
    "VN": ["gov.vn", "chinhphu.vn", "vbpl.vn", "baohiemxahoi.gov.vn"],
}
OFFICIAL_TYPES = {"law", "tax_authority", "official_guidance", "collective_agreement"}

SYSTEM_PROMPT = """You research payroll rules on the web for a payroll consultant.
Find the rule values that answer the question, for the given country and pay date.

- Prefer primary official sources: legislation, tax and social security authorities, government sites.
- Before recording a source, fetch the page with web_fetch and read it.
- Call record_source once per rule value you found. The quote must be copied verbatim from the page and
  must contain the value. Give the effective date the page states for the value (YYYY-MM-DD); leave it
  empty if the page states none.
- Record only what you read on a page. Never fill in values from memory.
- Record at most 5 sources, then reply with the single word: done."""

RECORD_SOURCE_TOOL = {
    "name": "record_source",
    "description": "Record one rule value found on a web page you fetched, with a verbatim quote.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["url", "title", "source_type", "quote", "rule", "value", "unit",
                     "effective_from", "effective_to", "employee_scope"],
        "properties": {
            "url": {"type": "string", "description": "Exact URL of the page the quote comes from"},
            "title": {"type": "string"},
            "source_type": {"type": "string", "enum": ["law", "tax_authority", "official_guidance",
                                                       "collective_agreement", "commentary"]},
            "quote": {"type": "string", "description": "Verbatim sentence from the page containing the value"},
            "rule": {"type": "string", "description": "Rule id, reusing a suggested id when one fits"},
            "value": {"type": "number"},
            "unit": {"type": "string"},
            "effective_from": {"type": "string", "description": "YYYY-MM-DD, or empty if not stated"},
            "effective_to": {"type": "string", "description": "YYYY-MM-DD, or empty if open-ended"},
            "employee_scope": {"type": "array", "items": {"type": "string"},
                               "description": "Employee groups it applies to; [\"all\"] if not limited"},
        },
    },
}


@dataclass
class Candidate:
    url: str
    title: str
    source_type: str
    quote: str
    rule: str
    value: float
    unit: str
    effective_from: str
    effective_to: str
    employee_scope: list[str]


@dataclass
class Retrieval:
    """What the search session actually saw: URLs returned, and page text for fetched URLs."""
    urls: set[str] = field(default_factory=set)
    pages: dict[str, str] = field(default_factory=dict)


@dataclass
class Verdict:
    candidate: Candidate
    accepted: bool
    checks: list[str]        # human-readable check results, "ok: ..." / "flag: ..." / "fail: ..."
    domain: str = ""
    official: bool = False
    source_id: str = ""


@dataclass
class WebResearch:
    verdicts: list[Verdict] = field(default_factory=list)
    error: str = ""

    @property
    def accepted(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.accepted]


# ---------------------------------------------------------------- verification (pure, testable)

def _norm_url(u: str) -> str:
    p = urlparse(u.strip())
    host = (p.netloc or "").lower().removeprefix("www.")
    return f"{host}{p.path.rstrip('/')}"


def _domain(u: str) -> str:
    return (urlparse(u).netloc or "").lower().removeprefix("www.")


def _registrable(host: str) -> str:
    parts = host.split(".")
    two_level = {"gov.uk", "co.uk", "org.uk", "gov.vn", "com.vn", "gouv.fr", "fgov.be"}
    if len(parts) >= 3 and ".".join(parts[-2:]) in two_level:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def is_official(host: str, country: str) -> bool:
    return any(host == d or host.endswith("." + d) for d in OFFICIAL_DOMAINS.get(country, []))


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _value_forms(v: float) -> set[str]:
    """Ways a number may be written: 14, 14.0, 14.00, 14,00, 5000000, 5,000,000, 5.000.000, 5 000 000."""
    forms = set()
    whole = float(v).is_integer()
    for decimals in ([0, 1, 2] if whole else [1, 2, 3]):
        s = f"{v:.{decimals}f}"
        forms |= {s, s.replace(".", ",")}
        if whole and decimals == 0:
            n = int(v)
            forms |= {f"{n:,}", f"{n:,}".replace(",", "."), f"{n:,}".replace(",", " "), f"{n:,}".replace(",", " ")}
    return {f for f in forms if f}


def value_in_text(v: float, text: str) -> bool:
    return any(re.search(r"(?<![\d.,])" + re.escape(f) + r"(?![\d])", text) for f in _value_forms(v))


def verify(candidates: list[Candidate], retrieval: Retrieval, q: Question) -> list[Verdict]:
    seen = {_norm_url(u) for u in retrieval.urls}
    pages = {_norm_url(u): _squash(t) for u, t in retrieval.pages.items()}
    verdicts = []
    for c in candidates:
        checks, ok = [], True
        key, host = _norm_url(c.url), _domain(c.url)
        official = is_official(host, q.country)

        if key in seen or key in pages:
            checks.append("ok: URL was returned by the search")
        else:
            checks.append("fail: URL never appeared in the search results (possibly invented)"); ok = False

        page = pages.get(key)
        if page is None:
            checks.append("fail: page was not fetched, so the quote cannot be checked"); ok = False
        elif _squash(c.quote) and _squash(c.quote) in page:
            checks.append("ok: quote found word for word on the page")
        else:
            checks.append("fail: quote not found on the fetched page"); ok = False

        if value_in_text(c.value, c.quote):
            checks.append(f"ok: value {c.value:g} appears in the quote")
        else:
            checks.append(f"fail: value {c.value:g} does not appear in the quote"); ok = False

        frm = _parse_date(c.effective_from)
        to = _parse_date(c.effective_to)
        if frm is None:
            checks.append("fail: no effective date stated"); ok = False
        elif frm > q.on_date or (to and to < q.on_date):
            checks.append(f"fail: not in force on {q.on_date.isoformat()} ({c.effective_from} to {c.effective_to or 'open'})"); ok = False
        else:
            checks.append(f"ok: in force on {q.on_date.isoformat()} (from {c.effective_from})")

        if official:
            checks.append(f"ok: official domain for {q.country} ({host})")
        else:
            checks.append(f"flag: not an official {q.country} domain ({host}); ranked as web commentary")

        verdicts.append(Verdict(c, ok, checks, host, official))

    # Agreement across independent domains, among accepted candidates.
    for v in verdicts:
        if not v.accepted:
            continue
        agreeing = {_registrable(o.domain) for o in verdicts if o.accepted and o.candidate.rule == v.candidate.rule
                    and abs(o.candidate.value - v.candidate.value) < 1e-9}
        n = len(agreeing)
        v.checks.append(f"ok: stated by {n} independent sites" if n > 1 else "flag: only one site states this value")
    return verdicts


def _parse_date(s: str) -> date | None:
    try:
        return date.fromisoformat(s.strip()) if s and s.strip() else None
    except ValueError:
        return None


def add_to_kb(kb: KnowledgeBase, verdicts: list[Verdict], q: Question, today: date | None = None) -> None:
    today = today or date.today()
    n = sum(1 for s in kb.sources.values() if s.origin == "web")
    for v in verdicts:
        if not v.accepted:
            continue
        c = v.candidate
        n += 1
        stype = c.source_type if (v.official and c.source_type in OFFICIAL_TYPES) else "web_commentary"
        src = Source(
            id=f"WEB_{q.country}_{q.topic}_{n}",
            title=c.title or v.domain,
            type=stype,
            country=q.country,
            topics=[q.topic],
            effective_from=_parse_date(c.effective_from),
            effective_to=_parse_date(c.effective_to),
            employee_scope=c.employee_scope or [ANY_EMPLOYEE],
            claims=[Claim(c.rule, c.value, c.unit)],
            text=c.quote,
            url=c.url,
            origin="web",
            retrieved_at=today,
            checks=v.checks,
        )
        kb.add_source(src, cites=[])
        v.source_id = src.id


# ---------------------------------------------------------------- search session (Claude)

def _collect(obj, retrieval: Retrieval) -> None:
    """Walk a tool-result block (as a dict) and record every URL and any page text attached to it."""
    if isinstance(obj, dict):
        url = obj.get("url")
        if isinstance(url, str) and url.startswith("http"):
            retrieval.urls.add(url)
            if "fetch" in str(obj.get("type", "")):   # page text only from fetched pages, not search listings
                texts = []
                _strings(obj, texts, skip={"url", "encrypted_content", "type", "media_type", "retrieved_at"})
                text = " ".join(texts)
                if text:
                    retrieval.pages[url] = retrieval.pages.get(url, "") + " " + text
        for v in obj.values():
            _collect(v, retrieval)
    elif isinstance(obj, list):
        for v in obj:
            _collect(v, retrieval)
    elif isinstance(obj, str):
        for u in re.findall(r"https?://[^\s\"'<>)]+", obj):
            retrieval.urls.add(u.rstrip(".,;"))


def _strings(obj, out: list[str], skip: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in skip:
                _strings(v, out, skip)
    elif isinstance(obj, list):
        for v in obj:
            _strings(v, out, skip)
    elif isinstance(obj, str):
        out.append(obj)


def _as_dict(block) -> dict:
    return block.model_dump() if hasattr(block, "model_dump") else dict(block)


def known_rules(kb: KnowledgeBase, topic: str) -> list[str]:
    return sorted({c.rule for s in kb.sources.values() if topic in s.topics for c in s.claims})


def search(kb: KnowledgeBase, q: Question, client=None) -> tuple[list[Candidate], Retrieval]:
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    rules = known_rules(kb, q.topic) or [q.topic]
    prompt = (f"Question: {q.text}\nCountry: {q.country}\nPay date: {q.on_date.isoformat()}\n"
              f"Topic: {q.topic.replace('_', ' ')}\nEmployee group: {q.employee_ctx}\n"
              f"Suggested rule ids: {', '.join(rules)}")
    messages = [{"role": "user", "content": prompt}]
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 5},
             {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 8},
             RECORD_SOURCE_TOOL]
    candidates: list[Candidate] = []
    retrieval = Retrieval()
    for _ in range(MAX_TURNS):
        response = client.beta.messages.create(
            model=MODEL, max_tokens=16000, system=SYSTEM_PROMPT, tools=tools, messages=messages,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("the search request was declined")
        results = []
        for block in response.content:
            if block.type.endswith("_tool_result") and block.type != "tool_result":
                _collect(_as_dict(block), retrieval)
            elif block.type == "tool_use" and block.name == "record_source":
                try:
                    candidates.append(Candidate(**{k: block.input[k] for k in RECORD_SOURCE_TOOL["input_schema"]["required"]}))
                    content = "recorded"
                except (KeyError, TypeError) as exc:
                    content = f"not recorded: {exc}"
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "pause_turn":
            continue                      # server tools hit their loop limit: resend to resume
        if results and len(candidates) < MAX_SOURCES:
            messages.append({"role": "user", "content": results})
            continue
        break
    return candidates[:MAX_SOURCES], retrieval


def research(kb: KnowledgeBase, q: Question, client=None) -> WebResearch:
    """Search, verify, and add accepted sources to the knowledge base."""
    try:
        candidates, retrieval = search(kb, q, client)
    except Exception as exc:   # no credentials, network, API error
        return WebResearch(error=f"web search unavailable ({type(exc).__name__}: {str(exc)[:120]})")
    verdicts = verify(candidates, retrieval, q)
    add_to_kb(kb, verdicts, q)
    return WebResearch(verdicts)
