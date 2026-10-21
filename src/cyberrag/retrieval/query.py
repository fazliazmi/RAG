"""Pull IDs and simple filters out of a question.

Embeddings are bad at exact identifiers, so CVE/CWE IDs are looked up
directly. Filters are only inferred from clear phrases ("critical
vulnerabilities", "since 2023", "actively exploited") to avoid surprises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_DASH = r"[-‐-―_ ]?"
CVE_RE = re.compile(rf"\bCVE{_DASH}(\d{{4}}){_DASH}(\d{{4,7}})\b", re.I)
CWE_RE = re.compile(rf"\bCWE{_DASH}(\d{{1,4}})\b", re.I)
_SEVERITY_RE = re.compile(
    r"\b(critical|high|medium|low)(?:[- ]severity|[- ]risk)?\s+"
    r"(?:cves?|vulnerabilit(?:y|ies)|vulns?|bugs?|flaws?|issues?)\b"
    r"|\b(critical|high|medium|low)[- ]severity\b"
    r"|\bseverity\s+(?:of\s+|is\s+|=\s*)?(critical|high|medium|low)\b",
    re.I,
)
_YEAR_RE = re.compile(r"\b(in|from|during|since|after)\s+((?:19|20)\d{2})\b", re.I)
_KEV_RE = re.compile(
    r"\b(actively exploited|known[- ]exploited|exploited in the wild|cisa kev|in the kev)\b", re.I
)


@dataclass
class ParsedQuery:
    text: str
    cve_ids: list[str] = field(default_factory=list)
    cwe_ids: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)


def extract_cve_ids(text: str) -> list[str]:
    return list(dict.fromkeys(f"CVE-{y}-{n}" for y, n in CVE_RE.findall(text)))


def extract_cwe_ids(text: str) -> list[str]:
    return list(dict.fromkeys(f"CWE-{int(n)}" for n in CWE_RE.findall(text)))


def parse_query(text: str) -> ParsedQuery:
    filters: dict[str, Any] = {}

    severities = {
        next(g for g in m.groups() if g).upper() for m in _SEVERITY_RE.finditer(text)
    }
    if severities:
        filters["severity"] = sorted(severities)

    year_match = _YEAR_RE.search(CVE_RE.sub(" ", text))
    if year_match:
        word, year = year_match.group(1).lower(), int(year_match.group(2))
        if word in ("since", "after"):
            filters["published"] = {"gte": f"{year}-01-01T00:00:00Z"}
        else:
            filters["published"] = {
                "gte": f"{year}-01-01T00:00:00Z",
                "lt": f"{year + 1}-01-01T00:00:00Z",
            }

    if _KEV_RE.search(text):
        filters["known_exploited"] = True

    return ParsedQuery(
        text=text.strip(),
        cve_ids=extract_cve_ids(text),
        cwe_ids=extract_cwe_ids(text),
        filters=filters,
    )
