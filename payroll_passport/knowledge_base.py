"""Loading and saving the knowledge base (sources, relations, experts) as JSON."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from .models import Claim, Edge, Expert, Source

DEFAULT_KB = Path(__file__).parent / "data" / "sample_kb.json"


def _d(value):
    return date.fromisoformat(value) if value else None


class KnowledgeBase:
    def __init__(self, sources: dict[str, Source], edges: list[Edge], experts: list[Expert]):
        self.sources = sources
        self.edges = edges
        self.experts = experts

    @classmethod
    def load(cls, path: Path | str = DEFAULT_KB) -> "KnowledgeBase":
        raw = json.loads(Path(path).read_text())
        sources = {}
        for s in raw["sources"]:
            sources[s["id"]] = Source(
                id=s["id"], title=s["title"], type=s["type"], country=s["country"],
                topics=s["topics"],
                effective_from=_d(s["effective_from"]),
                effective_to=_d(s.get("effective_to")),
                client=s.get("client", "GENERIC"),
                employee_scope=s.get("employee_scope", ["all"]),
                claims=[Claim(**c) for c in s.get("claims", [])],
                text=s.get("text", ""), url=s.get("url", ""),
                validated_by=s.get("validated_by"),
                review_by=_d(s.get("review_by")),
            )
        edges = [Edge(**e) for e in raw.get("edges", [])]
        experts = [Expert(**e) for e in raw.get("experts", [])]
        return cls(sources, edges, experts)

    def save(self, path: Path | str) -> None:
        def ser(obj):
            return obj.isoformat() if isinstance(obj, date) else obj

        payload = {
            "sources": [{k: ser(v) for k, v in asdict(s).items()} for s in self.sources.values()],
            "edges": [asdict(e) for e in self.edges],
            "experts": [asdict(e) for e in self.experts],
        }
        Path(path).write_text(json.dumps(payload, indent=2, default=ser) + "\n")

    def add_source(self, source: Source, cites: list[str], supersedes: list[str] = ()) -> None:
        if source.id in self.sources:
            raise ValueError(f"source id already exists: {source.id}")
        self.sources[source.id] = source
        self.edges += [Edge(source.id, c, "cites") for c in cites]
        self.edges += [Edge(source.id, s, "supersedes") for s in supersedes]
