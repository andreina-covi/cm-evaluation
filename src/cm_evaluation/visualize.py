"""Figures from ``evaluate`` run folders (``summary.json`` / ``predictions.jsonl``).

Works on a single run directory (the four files from one evaluate) or a tree
of later runs (``results/<model>/run_*/``).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from cm_evaluation.run import read_jsonl
from cm_evaluation.stats import summarize

CHANCE_MCQ = 0.25
LETTERS = ("A", "B", "C", "D")


def is_run_dir(path: Path) -> bool:
    return (path / "summary.json").is_file() or (path / "predictions.jsonl").is_file()


def discover_run_dirs(roots: Sequence[str | Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in roots:
        root = Path(raw).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        candidates: list[Path] = []
        if is_run_dir(root):
            candidates.append(root)
        else:
            candidates.extend(sorted({p.parent.resolve() for p in root.rglob("summary.json")}))
            candidates.extend(sorted({p.parent.resolve() for p in root.rglob("predictions.jsonl")}))
        for path in candidates:
            if path not in seen:
                seen.add(path)
                found.append(path)
    if not found:
        raise FileNotFoundError(
            f"No summary.json or predictions.jsonl under {list(roots)}"
        )
    return found


def load_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config: dict[str, Any] = {}
    config_path = run_dir / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
    records = read_jsonl(run_dir / "predictions.jsonl")
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
    elif records:
        summary = summarize(records)
    else:
        raise FileNotFoundError(f"No summary.json or predictions.jsonl in {run_dir}")
    return {
        "dir": run_dir,
        "label": run_dir.name,
        "config": config,
        "summary": summary,
        "records": records,
    }


def _acc_pairs(block: Optional[dict[str, Any]]) -> list[tuple[str, float, int]]:
    out: list[tuple[str, float, int]] = []
    for name, row in sorted((block or {}).items()):
        acc = row.get("accuracy")
        if acc is None:
            continue
        out.append((name, float(acc), int(row.get("n_scored") or row.get("n") or 0)))
    return out


def _confusion(records: list[dict[str, Any]]) -> list[list[int]]:
    grid = [[0] * len(LETTERS) for _ in LETTERS]
    index = {letter: i for i, letter in enumerate(LETTERS)}
    for rec in records:
        score = rec.get("score") or {}
        if score.get("kind") != "mcq":
            continue
        gold = score.get("gold")
        pred = score.get("predicted")
        if gold in index and pred in index:
            grid[index[gold]][index[pred]] += 1
    return grid


def write_figures(runs: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    primary = runs[0]
    summary = primary["summary"]

    def _bar(path: Path, title: str, pairs: list[tuple[str, float, int]], xlabel: str) -> None:
        if not pairs:
            return
        labels = [p[0] for p in pairs]
        values = [p[1] * 100 for p in pairs]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.bar(range(len(labels)), values, color="#4a6fa5")
        ax.axhline(CHANCE_MCQ * 100, color="#888888", linestyle="--", linewidth=1, label="chance (25%)")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 100)
        ax.set_title(title)
        ax.legend(frameon=False)
        for i, (_, _, n) in enumerate(pairs):
            ax.text(i, values[i] + 1.5, f"n={n}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    _bar(
        output_dir / "accuracy_by_construct.png",
        "MCQ accuracy by construct",
        _acc_pairs(summary.get("by_construct")),
        "Construct",
    )
    _bar(
        output_dir / "accuracy_by_class.png",
        "MCQ accuracy by class",
        _acc_pairs(summary.get("by_class")),
        "Class",
    )
    _bar(
        output_dir / "accuracy_by_frame_of_reference.png",
        "MCQ accuracy by construct × frame of reference",
        _acc_pairs(summary.get("by_construct_x_frame_of_reference")),
        "Construct × FoR",
    )
    _bar(
        output_dir / "accuracy_by_question_style.png",
        "MCQ accuracy by question style",
        _acc_pairs(summary.get("by_question_style")),
        "Style",
    )

    records = primary.get("records") or []
    if records:
        grid = _confusion(records)
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        im = ax.imshow(grid, cmap="Blues")
        ax.set_xticks(range(len(LETTERS)), LETTERS)
        ax.set_yticks(range(len(LETTERS)), LETTERS)
        ax.set_xlabel("Predicted letter")
        ax.set_ylabel("Gold letter")
        ax.set_title("MCQ confusion (gold × predicted)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        vmax = max((v for row in grid for v in row), default=0)
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                color = "white" if vmax and val > vmax / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)
        fig.tight_layout()
        path = output_dir / "confusion_gold_predicted.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

        gold_c = Counter()
        pred_c = Counter()
        for rec in records:
            score = rec.get("score") or {}
            if score.get("kind") != "mcq":
                continue
            if score.get("gold"):
                gold_c[score["gold"]] += 1
            if score.get("predicted"):
                pred_c[score["predicted"]] += 1
        fig, ax = plt.subplots(figsize=(6, 3.8))
        x = range(len(LETTERS))
        w = 0.35
        ax.bar([i - w / 2 for i in x], [gold_c[L] for L in LETTERS], w, label="gold", color="#4a6fa5")
        ax.bar([i + w / 2 for i in x], [pred_c[L] for L in LETTERS], w, label="predicted", color="#c47b3a")
        ax.set_xticks(list(x), LETTERS)
        ax.set_ylabel("Count")
        ax.set_xlabel("Option letter")
        ax.set_title("Gold vs predicted letter counts")
        ax.legend(frameon=False)
        fig.tight_layout()
        path = output_dir / "letter_counts.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    if len(runs) > 1:
        constructs = sorted(
            {name for run in runs for name in (run["summary"].get("by_construct") or {})}
        )
        if constructs:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            n_run = len(runs)
            width = 0.8 / n_run
            for i, run in enumerate(runs):
                block = run["summary"].get("by_construct") or {}
                vals = [
                    (block.get(c) or {}).get("accuracy") or 0.0
                    for c in constructs
                ]
                xs = [j + (i - (n_run - 1) / 2) * width for j in range(len(constructs))]
                ax.bar(xs, [v * 100 for v in vals], width, label=run["label"])
            ax.axhline(CHANCE_MCQ * 100, color="#888888", linestyle="--", linewidth=1, label="chance (25%)")
            ax.set_xticks(range(len(constructs)))
            ax.set_xticklabels(constructs, rotation=25, ha="right")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xlabel("Construct")
            ax.set_ylim(0, 100)
            ax.set_title("MCQ accuracy by construct across runs")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            path = output_dir / "accuracy_by_construct_across_runs.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            written.append(path)

    return written


def visualize(results: Sequence[str | Path], output: str | Path) -> list[Path]:
    runs = [load_run(path) for path in discover_run_dirs(results)]
    return write_figures(runs, Path(output))
