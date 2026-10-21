"""Raw NVD CVE JSON -> one Document per CVE.

CVE descriptions are short, so each CVE is a single document. Its text puts
the most useful facts (score, CWE, affected products) next to the
description, so both the embedder and the LLM see them.
"""

from __future__ import annotations

import re
from typing import Any

from cyberrag.schemas import Document

_CWE_ID = re.compile(r"^CWE-(\d+)$")
_METRIC_KEYS = ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2")
_MAX_AFFECTED_LINES = 8
_MAX_VERSIONS_PER_PRODUCT = 4
_MAX_PAYLOAD_PRODUCTS = 60


def english(items: list[dict[str, Any]] | None) -> str:
    items = items or []
    for item in items:
        if item.get("lang") == "en" and item.get("value"):
            return str(item["value"]).strip()
    return str(items[0].get("value", "")).strip() if items else ""


def pick_cvss(metrics: dict[str, Any]) -> dict[str, Any] | None:
    """Newest CVSS version available; within it, NVD's own ("Primary") score first."""
    for key in _METRIC_KEYS:
        entries = metrics.get(key) or []
        if not entries:
            continue
        entry = next((e for e in entries if e.get("type") == "Primary"), entries[0])
        data = entry.get("cvssData") or {}
        score = data.get("baseScore")
        severity = data.get("baseSeverity") or entry.get("baseSeverity")
        return {
            "version": str(data.get("version") or key.removeprefix("cvssMetricV")),
            "score": float(score) if score is not None else None,
            "severity": str(severity).upper() if severity else None,
            "vector": data.get("vectorString"),
        }
    return None


def cwe_ids_of(cve: dict[str, Any]) -> list[str]:
    ids: set[int] = set()
    for weakness in cve.get("weaknesses") or []:
        for desc in weakness.get("description") or []:
            m = _CWE_ID.match(str(desc.get("value", "")).strip())
            if m:
                ids.add(int(m.group(1)))
    return [f"CWE-{i}" for i in sorted(ids)]


def parse_cpe(criteria: str) -> tuple[str, str, str] | None:
    """'cpe:2.3:a:apache:log4j:2.0:*:...' -> ('apache', 'log4j', '2.0')."""
    parts = criteria.split(":")
    if len(parts) < 6 or parts[0] != "cpe":
        return None
    return parts[3], parts[4], parts[5]


def _version_range(match: dict[str, Any], version: str) -> str:
    lo = (
        f">={match['versionStartIncluding']}" if match.get("versionStartIncluding")
        else f">{match['versionStartExcluding']}" if match.get("versionStartExcluding")
        else ""
    )
    hi = (
        f"<={match['versionEndIncluding']}" if match.get("versionEndIncluding")
        else f"<{match['versionEndExcluding']}" if match.get("versionEndExcluding")
        else ""
    )
    if lo or hi:
        return " ".join(p for p in (lo, hi) if p)
    return version if version not in ("*", "-", "") else ""


def _is_blank(value: Any) -> bool:
    return not value or str(value).strip().lower() in ("n/a", "unknown", "-", "*")


def _display_name(vendor: str, product: str) -> str:
    vendor, product = vendor.replace("_", " "), product.replace("_", " ")
    if not vendor or product.startswith(vendor):
        return product
    return f"{vendor} {product}"


def affected_summary(
    cve: dict[str, Any], description: str = ""
) -> tuple[list[str], list[str], list[str]]:
    """Returns (vendors, products, display lines like 'apache log4j (>=2.0 <2.15.0)').

    Products named in the description come first, then the CNA's own list, then
    NVD's CPE list, so the main product is not buried under downstream vendors.
    """
    versions: dict[tuple[str, str], list[str]] = {}

    def add(vendor: str, product: str, version: str) -> None:
        key = (vendor.lower().strip(), product.lower().strip())
        bucket = versions.setdefault(key, [])
        if version and version not in bucket:
            bucket.append(version)

    # 1. The CNA's "affected" data (the reporter's own statement, often present before
    #    NVD analysis)
    for block in cve.get("affected") or []:
        for item in block.get("affectedData") or []:
            vendor = item.get("vendor")
            product = item.get("product") or item.get("packageName")
            if _is_blank(product):
                continue
            vendor = "" if _is_blank(vendor) else str(vendor)
            for v in item.get("versions") or []:
                if v.get("status") != "affected":
                    continue
                version = "" if _is_blank(v.get("version")) else str(v["version"])
                upper = v.get("lessThan") or v.get("lessThanOrEqual")
                if upper and not _is_blank(upper) and "*" not in str(upper):
                    op = "<" if v.get("lessThan") else "<="
                    version = f"{version} to {op}{upper}".strip() if version else f"{op}{upper}"
                add(vendor, str(product), version)
            if not item.get("versions"):
                add(vendor, str(product), "")

    # 2. NVD's CPE configurations (structured, present for analysed CVEs)
    for config in cve.get("configurations") or []:
        for node in config.get("nodes") or []:
            for match in node.get("cpeMatch") or []:
                if not match.get("vulnerable"):
                    continue
                parsed = parse_cpe(str(match.get("criteria", "")))
                if parsed:
                    vendor, product, version = parsed
                    add(vendor, product, _version_range(match, version))

    desc = description.lower()

    def mentioned(key: tuple[str, str]) -> bool:
        product = key[1].replace("_", " ")
        return bool(product) and product in desc

    ordered = sorted(versions.items(), key=lambda kv: not mentioned(kv[0]))  # stable sort
    vendors = list(dict.fromkeys(v for (v, _), _ in ordered if v))
    products = list(dict.fromkeys(p for (_, p), _ in ordered if p))
    lines = []
    for (vendor, product), vers in ordered:
        name = _display_name(vendor, product)
        shown = ", ".join(vers[:_MAX_VERSIONS_PER_PRODUCT])
        if len(vers) > _MAX_VERSIONS_PER_PRODUCT:
            shown += ", ..."
        lines.append(f"{name} ({shown})" if shown else name)
    return vendors[:_MAX_PAYLOAD_PRODUCTS], products[:_MAX_PAYLOAD_PRODUCTS], lines


def _utc(ts: str | None) -> str | None:
    if not ts:
        return None
    return ts if ts.endswith("Z") or "+" in ts[10:] else ts + "Z"


def is_rejected(cve: dict[str, Any]) -> bool:
    return str(cve.get("vulnStatus", "")).lower() == "rejected"


def normalize_cve(cve: dict[str, Any], *, skip_rejected: bool = True) -> Document | None:
    """Returns None for records that should not be indexed (rejected or empty)."""
    cve_id = cve.get("id")
    if not cve_id or (skip_rejected and is_rejected(cve)):
        return None
    description = english(cve.get("descriptions"))
    if not description:
        return None

    status = cve.get("vulnStatus", "")
    published = cve.get("published", "")
    cvss = pick_cvss(cve.get("metrics") or {})
    cwe_ids = cwe_ids_of(cve)
    vendors, products, affected_lines = affected_summary(cve, description)
    kev_added = cve.get("cisaExploitAdd")
    kev_name = cve.get("cisaVulnerabilityName")

    title_line = f"{cve_id}: {kev_name}" if kev_name else cve_id
    facts = [f"Published: {published[:10]}", f"Status: {status}"]
    if cvss and cvss["score"] is not None:
        facts.append(f"CVSS {cvss['version']}: {cvss['score']:.1f} {cvss['severity'] or ''}".strip())
    lines = [title_line, " | ".join(facts)]
    if kev_added:
        lines.append(f"Known exploited (CISA KEV): yes, added {kev_added}")
        if cve.get("cisaRequiredAction"):
            lines.append(f"CISA required action: {cve['cisaRequiredAction']}")
    if cwe_ids:
        lines.append("Weaknesses: " + ", ".join(cwe_ids))
    if affected_lines:
        more = len(affected_lines) - _MAX_AFFECTED_LINES
        shown = "; ".join(affected_lines[:_MAX_AFFECTED_LINES]) + (f"; +{more} more" if more > 0 else "")
        lines.append(f"Affected: {shown}")
    lines.append(f"Description: {description}")

    payload: dict[str, Any] = {
        "cve_id": cve_id,
        "title": kev_name or (description[:100] + ("..." if len(description) > 100 else "")),
        "published": _utc(published),
        "year": int(published[:4]) if published[:4].isdigit() else None,
        "last_modified": _utc(cve.get("lastModified")),
        "vuln_status": status,
        "cvss_version": cvss["version"] if cvss else None,
        "base_score": cvss["score"] if cvss else None,
        "severity": cvss["severity"] if cvss else None,
        "vector": cvss["vector"] if cvss else None,
        "cwe_ids": cwe_ids,
        "vendors": vendors,
        "products": products,
        "known_exploited": bool(kev_added),
        "kev_date_added": kev_added,
        "references": [r["url"] for r in (cve.get("references") or [])[:5] if r.get("url")],
        "url": f"https://nvd.nist.gov/vuln/detail/{cve_id}",
    }
    return Document(id=cve_id, source="cve", text="\n".join(lines), payload=payload)
