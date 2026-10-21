"""Small evaluation harness, mainly for comparing LLM profiles and embedders.

Metrics per question set:
  hit rate   - an expected ID is in the retrieved context
  MRR        - 1 / rank of the first expected ID (higher is better)
  citation   - the answer mentions an expected ID (skipped with --retrieval-only)
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from cyberrag.config import Settings
from cyberrag.errors import ConfigError
from cyberrag.generation.rag import RAGPipeline


@dataclass
class EvalCase:
    question: str
    expected: list[str]
    requires: str | None = None  # "cve" or "cwe": skipped if that index is missing


@dataclass
class EvalRow:
    question: str
    expected: list[str]
    retrieved: list[str] = field(default_factory=list)
    rank: int | None = None
    cited: bool | None = None
    answer: str | None = None
    seconds: float = 0.0
    skipped: str | None = None


def load_cases(path: Path) -> list[EvalCase]:
    if not path.is_file():
        raise ConfigError(f"Question file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [
        EvalCase(
            question=item["question"],
            expected=[e.upper() for e in item["expected"]],
            requires=item.get("requires"),
        )
        for item in data
    ]


def run_eval(
    pipeline: RAGPipeline,
    cases: list[EvalCase],
    retrieval_only: bool = False,
    on_row: Callable[[EvalRow], None] | None = None,
) -> list[EvalRow]:
    retriever = pipeline.retriever
    available = {
        "cve": retriever.store.exists(retriever.cve_collection),
        "cwe": retriever.store.exists(retriever.cwe_collection),
    }
    rows = []
    for case in cases:
        row = EvalRow(question=case.question, expected=case.expected)
        if case.requires and not available.get(case.requires, False):
            row.skipped = f"no {case.requires} index"
        else:
            started = time.perf_counter()
            prep = pipeline.prepare(case.question)
            row.retrieved = list(dict.fromkeys(h.ref for h in prep.hits))
            row.rank = next(
                (i for i, ref in enumerate(row.retrieved, 1) if ref in case.expected), None
            )
            if not retrieval_only:
                row.answer = pipeline.llm.chat(prep.messages)
                upper = row.answer.upper()
                row.cited = any(e in upper for e in case.expected)
            row.seconds = time.perf_counter() - started
        rows.append(row)
        if on_row:
            on_row(row)
    return rows


def summarize(rows: list[EvalRow]) -> dict[str, float | int | None]:
    scored = [r for r in rows if r.skipped is None]
    n = len(scored)
    cited = [r for r in scored if r.cited is not None]
    return {
        "questions": n,
        "skipped": len(rows) - n,
        "hit_rate": sum(r.rank is not None for r in scored) / n if n else 0.0,
        "mrr": sum(1 / r.rank for r in scored if r.rank) / n if n else 0.0,
        "citation_rate": sum(bool(r.cited) for r in cited) / len(cited) if cited else None,
        "avg_seconds": sum(r.seconds for r in scored) / n if n else 0.0,
    }


def save_results(
    settings: Settings, pipeline: RAGPipeline, rows: list[EvalRow], summary: dict
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    profile = pipeline.llm.cfg.profile
    path = settings.data_path / "eval" / f"{stamp}-{profile}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "llm": pipeline.llm.name,
        "embedder": pipeline.retriever.embedder.name,
        "summary": summary,
        "rows": [asdict(r) for r in rows],
    }, indent=2), encoding="utf-8")
    return path
