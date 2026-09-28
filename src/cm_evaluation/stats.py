"""Per-construct (and FoR) summaries. No LLM involved."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _key(record: dict[str, Any], *fields: str) -> str:
    parts = [str(record.get(field) or "unknown") for field in fields]
    return " × ".join(parts)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """``records`` are prediction dicts with item metadata + a ``score`` block."""
    mcq = [r for r in records if (r.get("score") or {}).get("kind") == "mcq"]
    nav = [r for r in records if (r.get("score") or {}).get("kind") == "nav_sequence"]
    errors = [r for r in records if r.get("error")]

    def _group(rows: list[dict[str, Any]], *fields: str) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[_key(row, *fields)].append(row)
        out: dict[str, Any] = {}
        for name, group in sorted(buckets.items()):
            scored = [g for g in group if g.get("score", {}).get("correct") is not None]
            n_correct = sum(1 for g in scored if g["score"]["correct"])
            n_parse_fail = sum(1 for g in group if g.get("score") and not g["score"].get("parse_ok"))
            out[name] = {
                "n": len(group),
                "n_scored": len(scored),
                "n_correct": n_correct,
                "accuracy": (n_correct / len(scored)) if scored else None,
                "n_parse_fail": n_parse_fail,
            }
        return out

    n_correct = sum(1 for r in mcq if r.get("score", {}).get("correct"))
    return {
        "n_records": len(records),
        "n_errors": len(errors),
        "n_mcq": len(mcq),
        "n_nav": len(nav),
        "n_mcq_correct": n_correct,
        "overall_mcq_accuracy": (n_correct / len(mcq)) if mcq else None,
        "by_construct": _group(mcq, "construct"),
        "by_construct_x_frame_of_reference": _group(mcq, "construct", "frame_of_reference"),
        "by_class": _group(mcq, "class"),
        "by_question_style": _group(mcq, "question_style"),
        "nav_by_construct": _group(nav, "construct"),
    }
