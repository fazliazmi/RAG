"""Raw MITRE CWE weakness JSON -> section Documents.

A CWE entry is long, so it is split by topic (overview, consequences,
mitigations, detection, examples, observed CVEs). Each chunk repeats a short
header, so it still makes sense when retrieved alone.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any

from cyberrag.schemas import Document

MAX_CHUNK_CHARS = 1800  # stays under the embedder's max_chars, so each chunk is embedded whole
MAX_CODE_CHARS = 600
_TITLES = {
    "overview": "Overview",
    "consequences": "Common consequences",
    "mitigations": "Potential mitigations",
    "detection": "Detection methods",
    "examples": "Demonstrative examples",
    "observed": "Observed examples (real CVEs)",
}


def clean(text: Any) -> str:
    if not text:
        return ""
    text = str(text).replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _code(code: str) -> str:
    code = re.sub(r"^```\w*\n?|```$", "", str(code).strip(), flags=re.M)
    code = textwrap.dedent(code.replace("\t", "    ")).strip()
    return code if len(code) <= MAX_CODE_CHARS else code[:MAX_CODE_CHARS].rstrip() + "\n..."


def _overview(w: dict[str, Any]) -> list[str]:
    blocks = [clean(w.get("Description")), clean(w.get("ExtendedDescription"))]
    terms = [
        f"{t['Term']}" + (f" ({clean(t['Description'])})" if t.get("Description") else "")
        for t in _as_list(w.get("AlternateTerms")) if t.get("Term")
    ]
    if terms:
        blocks.append("Also known as: " + "; ".join(terms))
    platforms = []
    for p in _as_list(w.get("ApplicablePlatforms")):
        label = p.get("Name") or p.get("Class")
        if label:
            prevalence = p.get("Prevalence")
            platforms.append(f"{p.get('Type', '')}: {label}" + (f" ({prevalence})" if prevalence else ""))
    if platforms:
        blocks.append("Applies to: " + "; ".join(platforms))
    phases = sorted({m.get("Phase") for m in _as_list(w.get("ModesOfIntroduction")) if m.get("Phase")})
    if phases:
        blocks.append("Introduced during: " + ", ".join(phases))
    related = []
    for r in _as_list(w.get("RelatedWeaknesses")):
        item = f"{r.get('Nature')} CWE-{r.get('CweID')}"
        if r.get("CweID") and item not in related:
            related.append(item)
    if related:
        blocks.append("Related weaknesses: " + "; ".join(related))
    return [b for b in blocks if b]


def _consequences(w: dict[str, Any]) -> list[str]:
    out = []
    for c in _as_list(w.get("CommonConsequences")):
        parts = [
            "Scope: " + ", ".join(_as_list(c.get("Scope"))) if c.get("Scope") else "",
            "Impact: " + ", ".join(_as_list(c.get("Impact"))) if c.get("Impact") else "",
        ]
        line = " | ".join(p for p in parts if p)
        note = clean(c.get("Note"))
        out.append(f"- {line}" + (f". {note}" if note else ""))
    return out


def _mitigations(w: dict[str, Any]) -> list[str]:
    out = []
    for m in _as_list(w.get("PotentialMitigations")):
        phase = ", ".join(_as_list(m.get("Phase")))
        label = " ".join(p for p in (f"[{phase}]" if phase else "", m.get("Strategy") or "") if p)
        text = clean(m.get("Description"))
        effectiveness = m.get("Effectiveness")
        line = f"- {label}: {text}" if label else f"- {text}"
        if effectiveness:
            line += f" (Effectiveness: {effectiveness})"
        out.append(line)
    return out


def _detection(w: dict[str, Any]) -> list[str]:
    out = []
    for d in _as_list(w.get("DetectionMethods")):
        line = f"- {d.get('Method', 'Method')}: {clean(d.get('Description'))}"
        if d.get("Effectiveness"):
            line += f" (Effectiveness: {d['Effectiveness']})"
        out.append(line)
    return out


def _examples(w: dict[str, Any]) -> list[str]:
    out = []
    for n, example in enumerate(_as_list(w.get("DemonstrativeExamples")), 1):
        parts = [f"Example {n}:"]
        for entry in _as_list(example.get("Entries")):
            if entry.get("IntroText"):
                parts.append(clean(entry["IntroText"]))
            if entry.get("BodyText"):
                parts.append(clean(entry["BodyText"]))
            if entry.get("ExampleCode"):
                label = ", ".join(x for x in (entry.get("Nature"), entry.get("Language")) if x)
                parts.append(f"Code ({label}):\n{_code(entry['ExampleCode'])}" if label
                             else f"Code:\n{_code(entry['ExampleCode'])}")
        out.append("\n".join(parts))
    return out


def _observed(w: dict[str, Any]) -> list[str]:
    return [
        f"- {o['Reference']}: {clean(o.get('Description'))}"
        for o in _as_list(w.get("ObservedExamples")) if o.get("Reference")
    ]


_SECTIONS = {
    "overview": _overview,
    "consequences": _consequences,
    "mitigations": _mitigations,
    "detection": _detection,
    "examples": _examples,
    "observed": _observed,
}


def _split(blocks: list[str], limit: int) -> list[str]:
    """Group blocks into chunks of at most `limit` characters, splitting only between blocks
    (a single oversized block is cut at a line or word boundary)."""
    pieces: list[str] = []
    for block in blocks:
        while len(block) > limit:
            cut = block.rfind("\n", 0, limit)
            if cut < limit // 2:
                cut = block.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            pieces.append(block[:cut].rstrip())
            block = block[cut:].lstrip()
        if block:
            pieces.append(block)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n{piece}" if current else piece
        if len(candidate) > limit and current:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def normalize_cwe(w: dict[str, Any], *, max_chars: int = MAX_CHUNK_CHARS) -> list[Document]:
    status = w.get("Status", "")
    if not w.get("ID") or status.lower() == "deprecated":
        return []
    cwe_id = f"CWE-{w['ID']}"
    name = clean(w.get("Name"))
    header_bits = [f"Abstraction: {w.get('Abstraction', '?')}", f"Status: {status}"]
    if w.get("LikelihoodOfExploit"):
        header_bits.append(f"Likelihood of exploit: {w['LikelihoodOfExploit']}")
    header = f"{cwe_id}: {name}\n" + " | ".join(header_bits)

    related = sorted(
        {f"CWE-{r['CweID']}" for r in _as_list(w.get("RelatedWeaknesses")) if r.get("CweID")},
        key=lambda s: int(s[4:]),
    )
    observed = [o["Reference"] for o in _as_list(w.get("ObservedExamples")) if o.get("Reference")]
    base_payload = {
        "cwe_id": cwe_id,
        "title": name,
        "name": name,
        "abstraction": w.get("Abstraction"),
        "status": status,
        "likelihood": w.get("LikelihoodOfExploit"),
        "related_cwes": related,
        "observed_cves": [o for o in observed if o.upper().startswith("CVE-")][:50],
        "url": f"https://cwe.mitre.org/data/definitions/{w['ID']}.html",
    }

    docs: list[Document] = []
    budget = max(400, max_chars - len(header) - 40)
    for section, builder in _SECTIONS.items():
        blocks = builder(w)
        if not blocks:
            continue
        for part, body in enumerate(_split(blocks, budget), 1):
            doc_id = f"{cwe_id}#{section}" if part == 1 else f"{cwe_id}#{section}-{part}"
            title = _TITLES[section] + (f" (part {part})" if part > 1 else "")
            docs.append(
                Document(
                    id=doc_id,
                    source="cwe",
                    text=f"{header}\n{title}:\n{body}",
                    payload={**base_payload, "section": section, "part": part},
                )
            )
    return docs
