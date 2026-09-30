"""Local web interface (standard library only).

  python -m payroll_passport.web [--kb path] [--port 8000]

Serves static/index.html and a small JSON API:
  GET  /api/meta      - options for the form + knowledge base overview
  POST /api/ask       - run the pipeline for a question
  POST /api/validate  - store an expert-validated answer, then re-run the question
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .assistant import Outcome, ask
from .capture import capture
from .knowledge_base import DEFAULT_KB, KnowledgeBase
from .models import Claim, Question, Source
from .topics import TopicMatch, describe, resolve_topic

STATIC = Path(__file__).parent / "static"


def _source_dict(s: Source) -> dict:
    return {
        "id": s.id, "title": s.title, "type": s.type, "tier": s.tier, "country": s.country,
        "client": s.client, "topics": s.topics, "employee_scope": s.employee_scope,
        "effective_from": s.effective_from.isoformat(),
        "effective_to": s.effective_to.isoformat() if s.effective_to else None,
        "claims": [{"rule": c.rule, "value": c.value, "unit": c.unit} for c in s.claims],
        "text": s.text, "validated_by": s.validated_by, "url": s.url, "origin": s.origin,
        "checks": s.checks,
        "review_by": s.review_by.isoformat() if s.review_by else None,
    }


def outcome_dict(out: Outcome, kb: KnowledgeBase) -> dict:
    d = {
        "action": out.decision.action.value,
        "reasons": out.decision.reasons,
        "conflicts": [{"rule": c.rule, "values": c.values} for c in out.decision.conflicts],
        "ranked": [{"score": r.score, "notes": r.notes, "source": _source_dict(r.source)}
                   for r in out.ranking.ranked],
        "outdated": [{"id": k, "reason": v, "title": kb.sources[k].title}
                     for k, v in out.ranking.outdated.items()],
        "not_applicable": [{"id": k, "reason": v, "title": kb.sources[k].title}
                           for k, v in out.ranking.not_applicable.items()],
        "draft": None,
        "draft_mode": out.draft_mode,
        "experts": [{"id": e.id, "name": e.name, "topics": e.topics, "open_cases": e.open_cases}
                    for e in out.experts or []],
        "expert_brief": out.expert_brief,
        "web": None if out.web is None else {
            "error": out.web.error,
            "results": [{"url": v.candidate.url, "title": v.candidate.title, "domain": v.domain,
                         "official": v.official, "accepted": v.accepted, "source_id": v.source_id,
                         "quote": v.candidate.quote, "claim": f"{v.candidate.rule} = {v.candidate.value:g} {v.candidate.unit}".strip(),
                         "checks": v.checks} for v in out.web.verdicts],
        },
    }
    if out.draft:
        d["draft"] = out.draft.model_dump()
    return d


def _question(p: dict) -> Question:
    for key in ("question", "country", "topic", "date"):
        if not str(p.get(key, "")).strip():
            raise ValueError(f"missing field: {key}")
    return Question(text=p["question"].strip(), country=p["country"], topic=p["topic"],
                    on_date=date.fromisoformat(p["date"]),
                    client=p.get("client") or "GENERIC", employee_ctx=p.get("employee") or "all")


class App:
    def __init__(self, kb_path: Path):
        self.kb_path = kb_path
        self.kb = KnowledgeBase.load(kb_path)

    def meta(self) -> dict:
        srcs = self.kb.sources.values()
        return {
            "kb_path": str(self.kb_path),
            "countries": sorted({s.country for s in srcs} | {c for e in self.kb.experts for c in e.countries}),
            "topics": sorted({t for s in srcs for t in s.topics}),
            "clients": sorted({s.client for s in srcs} - {"GENERIC"}),
            "employee_scopes": sorted({e for s in srcs for e in s.employee_scope} - {"all"}),
            "experts": [{"id": e.id, "name": e.name, "countries": e.countries, "topics": e.topics}
                        for e in self.kb.experts],
            "sources": [_source_dict(s) for s in srcs],
        }

    def ask(self, p: dict) -> dict:
        """The topic is detected from the question text; the result says which words decided it."""
        if not str(p.get("question", "")).strip():
            raise ValueError("missing field: question")
        use_llm = bool(p.get("use_llm"))
        if p.get("topic"):   # consultant picked one of the suggested topics
            match = TopicMatch(p["topic"], method="chosen", confidence="high")
        else:
            match = resolve_topic(p["question"], self.kb.sources.values(), p.get("country", ""), use_llm)
        q = _question({**p, "topic": match.topic})
        q.topic_confidence, q.topic_note = match.confidence, describe(match)
        out = ask(self.kb, q, use_llm=use_llm, use_web=bool(p.get("use_web")))
        if out.web and out.web.accepted:
            self.kb.save(self.kb_path)      # keep verified web sources (flagged origin=web) for next time
        result = outcome_dict(out, self.kb)
        result["topic"] = {"topic": match.topic, "method": match.method, "confidence": match.confidence,
                           "description": describe(match), "alternatives": match.alternatives}
        return result

    def validate(self, p: dict) -> dict:
        q = _question(p)
        claims = []
        for c in p.get("claims", []):
            if not c.get("rule"):
                continue
            try:
                value = float(c["value"])
            except (TypeError, ValueError):
                value = c.get("value", "")
            claims.append(Claim(c["rule"], value, c.get("unit", "")))
        item = capture(self.kb, q, p.get("expert", ""), p.get("answer", "").strip(), claims,
                       p.get("evidence", []), p.get("supersedes", []))
        self.kb.save(self.kb_path)
        outcome = outcome_dict(ask(self.kb, q, use_llm=False), self.kb)
        outcome["topic"] = {"topic": q.topic, "method": "previous", "confidence": "high",
                            "description": "same question as before", "alternatives": []}
        return {"stored": _source_dict(item), "outcome": outcome}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body: bytes, ctype: str):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, obj):
            self._send(status, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/meta":
                self._json(HTTPStatus.OK, app.meta())
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self):
            routes = {"/api/ask": app.ask, "/api/validate": app.validate}
            if self.path not in routes:
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                self._json(HTTPStatus.OK, routes[self.path](payload))
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="payroll_passport.web")
    p.add_argument("--kb", default="payroll_kb.json",
                   help="working knowledge base file (created from the bundled sample if missing)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)

    kb_path = Path(a.kb)
    if kb_path.resolve() == DEFAULT_KB.resolve():
        raise SystemExit("use a working copy, not the bundled sample")
    if not kb_path.exists():
        shutil.copy(DEFAULT_KB, kb_path)
        print(f"Created {kb_path} from the bundled sample")
    server = ThreadingHTTPServer((a.host, a.port), make_handler(App(kb_path)))
    print(f"Payroll Passport running at http://{a.host}:{a.port}  (knowledge base: {kb_path})")
    server.serve_forever()


if __name__ == "__main__":
    main()
