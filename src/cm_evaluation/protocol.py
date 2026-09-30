"""Eval format contract: re-export from cm-benchmark.

``SYSTEM_INSTRUCTION`` and ``wrap_item_for_eval`` live in
``cm_benchmark.evaluation.protocol``. This package does not copy them.

Install cm-benchmark in the same environment. For protocol edits in a local
clone to apply on the next run::

    pip install -e /path/to/cm-benchmark
"""

from __future__ import annotations

from typing import Any

from cm_benchmark.evaluation.protocol import (
    ACTION_VOCAB,
    SYSTEM_INSTRUCTION,
    wrap_item_for_eval,
)

MCQ_CONSTRUCTS = {
    "egocentric_encoding",
    "allocentric_encoding",
    "spatial_working_memory",
    "invisible_displacement",
    "spatial_updating",
    "perspective_taking",
}

NAV_CONSTRUCTS = {
    "route_knowledge",
    "survey_based_route_planning",
}

CLASS_OF = {
    "egocentric_encoding": 1,
    "allocentric_encoding": 1,
    "spatial_working_memory": 2,
    "invisible_displacement": 2,
    "spatial_updating": 3,
    "perspective_taking": 3,
    "route_knowledge": 4,
    "survey_based_route_planning": 4,
}


def is_mcq(item: dict[str, Any]) -> bool:
    construct = item.get("construct")
    options = item.get("options") or {}
    if construct in NAV_CONSTRUCTS:
        return False
    return bool(options) or construct in MCQ_CONSTRUCTS


def format_user_text(item: dict[str, Any]) -> str:
    """Question plus option list when the item text does not already list them."""
    bundle = wrap_item_for_eval(item)
    question = bundle["question"].rstrip()
    options = item.get("options") or {}
    if not options:
        return question
    already = all(
        f"{letter}." in question or f"{letter})" in question for letter in options
    )
    if already:
        return question
    lines = [question, ""]
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    return "\n".join(lines)
