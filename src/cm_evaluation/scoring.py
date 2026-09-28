"""[CODE] deterministic scorers. An LLM must never judge correctness."""

from __future__ import annotations

import re
from typing import Any, Optional

_LETTER = re.compile(r"\b([A-D])\b", re.IGNORECASE)
_LETTER_LINE = re.compile(r"^\s*([A-D])(?:\s*[.)]|:|\s|$)", re.IGNORECASE | re.MULTILINE)
NAV_ACTIONS = ("move_ahead", "rotate_left", "rotate_right", "move_back")
_ACTION_RE = re.compile(
    r"(move_ahead|rotate_left|rotate_right|move_back|moveahead|rotateleft|rotateright|moveback)",
    re.IGNORECASE,
)


def parse_mcq_letter(raw: str) -> Optional[str]:
    """Extract a single A–D choice from a model reply."""
    if not raw or not str(raw).strip():
        return None
    text = str(raw).strip()
    first_line = text.splitlines()[0].strip()
    match = _LETTER_LINE.search(first_line) or _LETTER_LINE.search(text)
    if match:
        return match.group(1).upper()
    letters = _LETTER.findall(text)
    if len(set(letter.upper() for letter in letters)) == 1:
        return letters[0].upper()
    if letters:
        return letters[0].upper()
    return None


def parse_nav_actions(raw: str) -> list[str]:
    found: list[str] = []
    for match in _ACTION_RE.findall(raw or ""):
        key = match.lower().replace(" ", "")
        mapping = {
            "moveahead": "move_ahead",
            "rotateleft": "rotate_left",
            "rotateright": "rotate_right",
            "moveback": "move_back",
        }
        found.append(mapping.get(key, key if key in NAV_ACTIONS else key))
    return found


def score_mcq(item: dict[str, Any], raw: str) -> dict[str, Any]:
    gold = str(item.get("answer") or "").strip().upper()
    predicted = parse_mcq_letter(raw)
    parse_ok = predicted is not None
    correct = bool(parse_ok and gold and predicted == gold)
    return {
        "kind": "mcq",
        "gold": gold or None,
        "predicted": predicted,
        "correct": correct,
        "parse_ok": parse_ok,
        "raw": raw,
    }


def score_item(item: dict[str, Any], raw: str) -> dict[str, Any]:
    from cm_evaluation.protocol import is_mcq

    if is_mcq(item):
        return score_mcq(item, raw)
    actions = parse_nav_actions(raw)
    return {
        "kind": "nav_sequence",
        "gold": item.get("answer"),
        "predicted": actions,
        "correct": None,
        "parse_ok": bool(actions),
        "raw": raw,
        "note": "Class-4 metric scoring (success/validity/SPL) needs the nav graph; stored raw + parsed actions only.",
    }
