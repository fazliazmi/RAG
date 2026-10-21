from cyberrag.retrieval.context import build_context, estimate_tokens
from cyberrag.retrieval.query import extract_cve_ids, extract_cwe_ids, parse_query
from cyberrag.retrieval.sparse import SparseEncoder, tokenize
from cyberrag.schemas import Hit


def test_extract_ids_in_many_spellings():
    text = "cve-2021-44228, CVE 2024 3094, CVE‑2023‑12345 and CVE-2021-44228 again"
    assert extract_cve_ids(text) == ["CVE-2021-44228", "CVE-2024-3094", "CVE-2023-12345"]
    assert extract_cwe_ids("cwe79 or CWE-079 or CWE 89") == ["CWE-79", "CWE-89"]


def test_filters_only_from_clear_phrases():
    assert parse_query("critical vulnerabilities in nginx").filters == {"severity": ["CRITICAL"]}
    assert parse_query("high severity bugs").filters["severity"] == ["HIGH"]
    assert parse_query("does this have high impact on availability?").filters == {}
    assert parse_query("actively exploited flaws in Exchange").filters["known_exploited"] is True


def test_year_filters():
    f = parse_query("SQL injection CVEs in 2023").filters["published"]
    assert f == {"gte": "2023-01-01T00:00:00Z", "lt": "2024-01-01T00:00:00Z"}
    assert parse_query("since 2022").filters["published"] == {"gte": "2022-01-01T00:00:00Z"}
    # a year inside a CVE ID is not a date filter
    assert "published" not in parse_query("what is in CVE-2023-4966").filters


def test_tokenize_keeps_versions_and_ids():
    tokens = tokenize("Log4j 2.14.1 is affected by CVE-2021-44228 in the logger")
    for expected in ("log4j", "2.14.1", "cve-2021-44228", "44228", "logger"):
        assert expected in tokens
    assert "the" not in tokens


def test_sparse_vectors_are_deterministic_and_valid():
    enc = SparseEncoder()
    a = enc.encode_document("xz backdoor xz liblzma")
    b = enc.encode_document("xz backdoor xz liblzma")
    assert a == b
    assert a.indices == sorted(set(a.indices))
    assert len(a.indices) == len(a.values) == 3
    assert enc.encode_query("").indices == []


def _hit(ref: str, text: str) -> Hit:
    return Hit(id=ref, source="cve", text=text)


def test_context_fits_budget_and_skips_oversized():
    hits = [_hit("CVE-1-1", "a" * 400), _hit("CVE-1-2", "b" * 4000), _hit("CVE-1-3", "c" * 400)]
    context, used = build_context(hits, budget_tokens=300, chars_per_token=4)
    assert [h.id for h in used] == ["CVE-1-1", "CVE-1-3"]
    assert estimate_tokens(context, 4) <= 300
    assert "[CVE-1-1]" in context


def test_context_truncates_first_hit_if_needed():
    context, used = build_context([_hit("CVE-1-1", "x" * 10_000)], 200, 4)
    assert len(used) == 1
    assert estimate_tokens(context, 4) <= 200
