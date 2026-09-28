from cm_evaluation.scoring import parse_mcq_letter, parse_nav_actions, score_item
from cm_evaluation.stats import summarize


def test_parse_mcq_variants() -> None:
    assert parse_mcq_letter("A") == "A"
    assert parse_mcq_letter("Answer: B.") == "B"
    assert parse_mcq_letter("C. left") == "C"
    assert parse_mcq_letter("the window is on the left") is None


def test_score_mcq_match() -> None:
    item = {"construct": "egocentric_encoding", "options": {"A": "left"}, "answer": "A"}
    assert score_item(item, "A")["correct"] is True
    assert score_item(item, "B")["correct"] is False
    assert score_item(item, "")["parse_ok"] is False


def test_parse_nav_actions() -> None:
    raw = "move_ahead -> rotate_left, MoveAhead"
    assert parse_nav_actions(raw) == ["move_ahead", "rotate_left", "move_ahead"]


def test_class4_not_letter_scored() -> None:
    item = {"construct": "route_knowledge", "options": {}, "answer": "move_ahead"}
    score = score_item(item, "move_ahead, rotate_right")
    assert score["kind"] == "nav_sequence"
    assert score["correct"] is None


def test_summarize_per_construct() -> None:
    records = [
        {
            "construct": "egocentric_encoding",
            "frame_of_reference": "egocentric",
            "class": 1,
            "question_style": "concise",
            "score": {"kind": "mcq", "correct": True, "parse_ok": True},
        },
        {
            "construct": "egocentric_encoding",
            "frame_of_reference": "egocentric",
            "class": 1,
            "question_style": "verbose",
            "score": {"kind": "mcq", "correct": False, "parse_ok": True},
        },
        {
            "construct": "spatial_working_memory",
            "frame_of_reference": "egocentric",
            "class": 2,
            "question_style": "concise",
            "score": {"kind": "mcq", "correct": True, "parse_ok": True},
        },
    ]
    summary = summarize(records)
    assert summary["overall_mcq_accuracy"] == 2 / 3
    assert summary["by_construct"]["egocentric_encoding"]["accuracy"] == 0.5
    assert summary["by_construct"]["spatial_working_memory"]["n_correct"] == 1
