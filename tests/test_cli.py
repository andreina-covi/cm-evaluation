from cm_evaluation.cli import build_parser
import pytest


def test_data_root_required() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["print-paths"])
    assert exc.value.code == 2


def test_evaluate_requires_items() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["evaluate", "--data-root", "/tmp/eval"])
    assert exc.value.code == 2


def test_evaluate_parses_items_path(tmp_path) -> None:
    parser = build_parser()
    items = tmp_path / "items"
    ns = parser.parse_args(
        [
            "evaluate",
            "--data-root",
            str(tmp_path),
            "--items",
            str(items),
            "--frames-root",
            str(tmp_path / "nav"),
        ]
    )
    assert ns.items == [items]


def test_evaluate_items_missing_value() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            ["evaluate", "--data-root", "/tmp/eval", "--items", "--frames-root", "/tmp/nav"]
        )
    assert exc.value.code == 2


def test_print_paths_parses(tmp_path) -> None:
    parser = build_parser()
    ns = parser.parse_args(
        ["print-paths", "--data-root", str(tmp_path), "--frames-root", str(tmp_path / "nav")]
    )
    assert ns.data_root == tmp_path
    assert ns.frames_root == tmp_path / "nav"
