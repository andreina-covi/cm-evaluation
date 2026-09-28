"""Eval format contract. Keep in sync with cm-benchmark ``evaluation.protocol``.

The Model Runner prepends this once per call. Item ``question`` text stays
construct-specific and must not repeat the contract.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from cm_benchmark.evaluation.protocol import (  # type: ignore
        ACTION_VOCAB,
        SYSTEM_INSTRUCTION,
        wrap_item_for_eval,
    )
except ImportError:  # cm-benchmark is optional for MCQ-only runs
    ACTION_VOCAB = (
        "move_ahead",
        "rotate_left",
        "rotate_right",
        "move_back",
    )
    _ACTION_LIST = ", ".join(ACTION_VOCAB)
    SYSTEM_INSTRUCTION = (
        "You are answering spatial-cognition questions from first-person views.\n"
        "\n"
        "Multiple-choice items: reply with exactly one option letter (A, B, C, or D).\n"
        "\n"
        "Navigation-action items (route knowledge and survey-based route planning): "
        "reply with an ordered sequence using only these action names: "
        f"{_ACTION_LIST}. Separate actions with commas or arrows. "
        "Do not add explanation."
    )

    def wrap_item_for_eval(
        item: Optional[dict[str, Any]],
        *,
        system: str = SYSTEM_INSTRUCTION,
    ) -> dict[str, str]:
        rec = item or {}
        return {
            "system": system,
            "question": rec.get("question") or "",
        }


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
