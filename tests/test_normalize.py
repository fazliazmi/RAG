from cyberrag.ingest.normalize_cve import normalize_cve, parse_cpe, pick_cvss
from cyberrag.ingest.normalize_cwe import MAX_CHUNK_CHARS, _split, normalize_cwe
from tests.conftest import REJECTED_CVE, cve_fixture, load_fixture


def test_xz_backdoor_cve():
    doc = normalize_cve(cve_fixture("cve-2024-3094.json"))
    assert doc is not None
    assert doc.id == "CVE-2024-3094" and doc.source == "cve"
    p = doc.payload
    assert p["severity"] == "CRITICAL" and p["base_score"] == 10.0
    assert p["cwe_ids"] == ["CWE-506"]
    assert "xz" in p["products"]
    assert p["known_exploited"] is False
    assert p["published"].startswith("2024-03-29") and p["published"].endswith("Z")
    assert p["year"] == 2024
    assert "Weaknesses: CWE-506" in doc.text
    assert "red hat red hat" not in doc.text  # vendor not repeated


def test_log4shell_cve():
    doc = normalize_cve(cve_fixture("cve-2021-44228.json"))
    assert doc is not None
    p = doc.payload
    assert p["known_exploited"] is True
    assert p["title"] == "Apache Log4j2 Remote Code Execution Vulnerability"
    assert p["cwe_ids"] == ["CWE-20", "CWE-400", "CWE-502", "CWE-917"]
    assert "Known exploited (CISA KEV): yes" in doc.text
    # the product named in the description is listed before downstream vendors
    affected = next(line for line in doc.text.splitlines() if line.startswith("Affected:"))
    assert "log4j" in affected.split(";")[0].lower()


def test_rejected_cve_is_skipped_unless_asked():
    assert normalize_cve(REJECTED_CVE) is None
    doc = normalize_cve(REJECTED_CVE, skip_rejected=False)
    assert doc is not None and doc.payload["vuln_status"] == "Rejected"


def test_pick_cvss_prefers_newest_version_and_primary():
    metrics = {
        "cvssMetricV2": [{"type": "Primary", "baseSeverity": "HIGH",
                          "cvssData": {"version": "2.0", "baseScore": 7.5}}],
        "cvssMetricV31": [
            {"type": "Secondary", "cvssData": {"version": "3.1", "baseScore": 5.0,
                                               "baseSeverity": "MEDIUM"}},
            {"type": "Primary", "cvssData": {"version": "3.1", "baseScore": 9.1,
                                             "baseSeverity": "CRITICAL"}},
        ],
    }
    assert pick_cvss(metrics)["score"] == 9.1
    only_v2 = pick_cvss({"cvssMetricV2": metrics["cvssMetricV2"]})
    assert only_v2["severity"] == "HIGH" and only_v2["version"] == "2.0"
    assert pick_cvss({}) is None


def test_parse_cpe():
    assert parse_cpe("cpe:2.3:a:apache:log4j:2.0:*:*:*:*:*:*:*") == ("apache", "log4j", "2.0")
    assert parse_cpe("garbage") is None


def test_cwe_sections():
    sqli = next(w for w in load_fixture("cwe-79-89.json")["Weaknesses"] if w["ID"] == "89")
    docs = normalize_cwe(sqli)
    ids = [d.id for d in docs]
    assert ids[0] == "CWE-89#overview"
    assert "CWE-89#mitigations" in ids and "CWE-89#observed" in ids
    assert len(ids) == len(set(ids))
    for d in docs:
        assert d.text.startswith("CWE-89: Improper Neutralization")
        assert len(d.text) <= MAX_CHUNK_CHARS
        assert d.payload["cwe_id"] == "CWE-89"
    overview = docs[0]
    assert "SQLi" in overview.text
    assert overview.payload["observed_cves"][0].startswith("CVE-")
    assert "CWE-943" in overview.payload["related_cwes"]


def test_deprecated_cwe_is_skipped():
    assert normalize_cwe({"ID": "1", "Name": "DEPRECATED: x", "Status": "Deprecated"}) == []


def test_split_respects_limit():
    blocks = ["a" * 300, "b" * 300, "c" * 900, "d " * 400]
    chunks = _split(blocks, 500)
    assert all(len(c) <= 500 for c in chunks)
    assert "".join(chunks).replace("\n", "").replace(" ", "") == "".join(blocks).replace(" ", "")
