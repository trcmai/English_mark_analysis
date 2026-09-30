"""Core data model for Payroll Passport."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

GENERIC_CLIENT = "GENERIC"
ANY_EMPLOYEE = "all"

# Lower number = higher authority. Used for the trust prior and the decision gate.
AUTHORITY_TIERS = {
    "law": 1,
    "tax_authority": 1,
    "official_guidance": 2,
    "collective_agreement": 2,   # binding for the sector it covers
    "knowledge": 3,              # expert-validated knowledge item
    "client_contract": 3,
    "internal_guide": 4,
    "wiki": 5,
    "email": 6,
    "note": 6,
}


@dataclass
class Claim:
    """A structured fact extracted from a source, e.g. min_wage_hourly = 14.00 EUR."""
    rule: str
    value: float | str
    unit: str = ""


@dataclass
class Source:
    id: str
    title: str
    type: str
    country: str
    topics: list[str]
    effective_from: date
    effective_to: Optional[date] = None
    client: str = GENERIC_CLIENT
    employee_scope: list[str] = field(default_factory=lambda: [ANY_EMPLOYEE])
    claims: list[Claim] = field(default_factory=list)
    text: str = ""
    url: str = ""
    validated_by: Optional[str] = None
    review_by: Optional[date] = None

    @property
    def tier(self) -> int:
        return AUTHORITY_TIERS.get(self.type, 6)

    @property
    def validated(self) -> bool:
        return self.validated_by is not None


@dataclass
class Edge:
    """Directed relation between sources: 'cites' (endorsement) or 'supersedes' (replacement)."""
    src: str
    dst: str
    relation: str


@dataclass
class Question:
    text: str
    country: str
    topic: str
    on_date: date
    client: str = GENERIC_CLIENT
    employee_ctx: str = ANY_EMPLOYEE
    topic_confidence: str = "high"   # how sure the topic detection was: high | medium | low | very_low
    topic_note: str = ""             # how the topic was chosen, shown when confidence is not high


@dataclass
class Expert:
    id: str
    name: str
    countries: list[str]
    topics: list[str]
    open_cases: int = 0
