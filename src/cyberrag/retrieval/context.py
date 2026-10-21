"""Fit retrieved documents into the LLM's context window.

Token counts are estimated from character length, which works for any model
without loading its tokenizer. The budget comes from the LLM profile's
num_ctx, so a bigger model automatically gets more context.
"""

from __future__ import annotations

import math
import re

from cyberrag.schemas import Hit

SEPARATOR = "\n\n---\n\n"
# MITRE's internal bibliography tags ("[REF-1482]") mean nothing to the reader,
# and small models copy them into answers as if they were citations.
_MITRE_REF = re.compile(r"\s*\[REF-\d+\]")


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return math.ceil(len(text) / max(chars_per_token, 1.0))


def format_hit(hit: Hit) -> str:
    if hit.source == "cwe":
        label = f"[{hit.ref}] (CWE entry, section: {hit.payload.get('section', 'overview')})"
    else:
        label = f"[{hit.ref}] (CVE record)"
    return f"{label}\n{_MITRE_REF.sub('', hit.text)}"


def build_context(
    hits: list[Hit], budget_tokens: int, chars_per_token: float
) -> tuple[str, list[Hit]]:
    """Add hits in priority order until the budget is used. A hit that does not fit is
    skipped (a later, shorter one may still fit); the first hit is cut down if needed."""
    blocks: list[str] = []
    used: list[Hit] = []
    spent = 0
    sep_cost = estimate_tokens(SEPARATOR, chars_per_token)
    for hit in hits:
        block = format_hit(hit)
        cost = estimate_tokens(block, chars_per_token) + sep_cost
        if spent + cost > budget_tokens:
            remaining = budget_tokens - spent - sep_cost
            if used or remaining < 64:
                continue
            block = block[: int(remaining * chars_per_token)].rstrip() + " ..."
            cost = budget_tokens - spent
        blocks.append(block)
        used.append(hit)
        spent += cost
    return SEPARATOR.join(blocks), used
