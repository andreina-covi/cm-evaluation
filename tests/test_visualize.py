import json
from pathlib import Path

from cm_evaluation.cli import build_parser
from cm_evaluation.visualize import discover_run_dirs, load_run, visualize, write_figures


def _write_run(folder: Path) -> None:
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(
        json.dumps(
            {
                "n_mcq": 4,
                "overall_mcq_accuracy": 0.5,
                "by_construct": {
                    "egocentric_encoding": {
                        "n": 4,
                        "n_scored": 4,
                        "n_correct": 2,
                        "accuracy": 0.5,
                        "n_parse_fail": 0,
                    }
                },
                "by_class": {
                    "1": {
                        "n": 4,
                        "n_scored": 4,
                        "n_correct": 2,
                        "accuracy": 0.5,
                        "n_parse_fail": 0,
                    }
                },
                "by_construct_x_frame_of_reference": {},
                "by_question_style": {},
            }
        )
    )
    (folder / "predictions.jsonl").write_text(
        json.dumps(
            {
                "construct": "egocentric_encoding",
                "class": 1,
                "score": {
                    "kind": "mcq",
                    "gold": "A",
                    "predicted": "A",
                    "correct": True,
                    "parse_ok": True,
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "construct": "egocentric_encoding",
                "class": 1,
                "score": {
                    "kind": "mcq",
                    "gold": "B",
                    "predicted": "C",
                    "correct": False,
                    "parse_ok": True,
                },
            }
        )
        + "\n"
    )


def test_discover_flat_run(tmp_path: Path) -> None:
    run = tmp_path / "results"
    _write_run(run)
    assert discover_run_dirs([run]) == [run.resolve()]
    loaded = load_run(run)
    assert loaded["summary"]["by_construct"]["egocentric_encoding"]["accuracy"] == 0.5


def test_discover_nested_runs(tmp_path: Path) -> None:
    a = tmp_path / "model" / "run_a"
    b = tmp_path / "model" / "run_b"
    _write_run(a)
    _write_run(b)
    found = discover_run_dirs([tmp_path])
    assert set(found) == {a.resolve(), b.resolve()}


def test_write_figures(tmp_path: Path) -> None:
    run = tmp_path / "results"
    _write_run(run)
    out = tmp_path / "figs"
    paths = write_figures([load_run(run)], out)
    names = {p.name for p in paths}
    assert "accuracy_by_construct.png" in names
    assert "confusion_gold_predicted.png" in names
    assert (out / "accuracy_by_construct.png").is_file()


def test_visualize_cli_parses(tmp_path: Path) -> None:
    parser = build_parser()
    ns = parser.parse_args(
        ["visualize", "--results", str(tmp_path), "--output", str(tmp_path / "out")]
    )
    assert ns.results == [tmp_path]
    assert ns.output == tmp_path / "out"


def test_visualize_end_to_end(tmp_path: Path) -> None:
    run = tmp_path / "results"
    _write_run(run)
    out = tmp_path / "figs"
    written = visualize([run], out)
    assert written
    assert all(p.is_file() for p in written)
