"""Resume-safe evaluation loop: items → sequential images → predictions JSONL."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from cm_evaluation.items import resolve_image_path
from cm_evaluation.paths import EvalPaths
from cm_evaluation.protocol import CLASS_OF, format_user_text, is_mcq, wrap_item_for_eval
from cm_evaluation.scoring import score_item
from cm_evaluation.stats import summarize


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def new_run_dir(results_root: Path, model_slug: str) -> Path:
    path = results_root / model_slug / f"run_{_now_stamp()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_done_ids(predictions_path: Path) -> set[str]:
    done: set[str] = set()
    if not predictions_path.is_file():
        return done
    with predictions_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            item_id = rec.get("item_id")
            if item_id:
                done.add(item_id)
    return done


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_summary(run_dir: Path, records: list[dict[str, Any]], extra: Optional[dict] = None) -> dict[str, Any]:
    summary = summarize(records)
    if extra:
        summary = {**extra, **summary}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _write_construct_table(run_dir / "by_construct.csv", summary.get("by_construct") or {})
    return summary


def _write_construct_table(path: Path, by_construct: dict[str, Any]) -> None:
    lines = ["construct,n,n_scored,n_correct,accuracy,n_parse_fail"]
    for name, row in by_construct.items():
        acc = row.get("accuracy")
        acc_s = "" if acc is None else f"{acc:.4f}"
        lines.append(
            f"{name},{row.get('n', 0)},{row.get('n_scored', 0)},"
            f"{row.get('n_correct', 0)},{acc_s},{row.get('n_parse_fail', 0)}"
        )
    path.write_text("\n".join(lines) + "\n")


def resolve_item_images(item: dict[str, Any], paths: EvalPaths) -> list[Path]:
    resolved = []
    for raw in item.get("image_paths") or []:
        resolved.append(resolve_image_path(raw, frames_root=paths.frames_root))
    if not resolved:
        raise FileNotFoundError(f"item {item.get('item_id')} has no image_paths")
    return resolved


def evaluate_items(
    runner,
    items: Iterable[dict[str, Any]],
    *,
    paths: EvalPaths,
    run_dir: Path,
    max_new_tokens_mcq: int,
    max_new_tokens_nav: int,
    do_sample: bool = False,
    resume: bool = True,
) -> list[dict[str, Any]]:
    predictions_path = run_dir / "predictions.jsonl"
    done = load_done_ids(predictions_path) if resume else set()
    for item in items:
        item_id = item.get("item_id") or ""
        if item_id and item_id in done:
            continue
        construct = item.get("construct")
        record: dict[str, Any] = {
            "item_id": item_id,
            "construct": construct,
            "class": item.get("class") or CLASS_OF.get(construct or "", None),
            "frame_of_reference": item.get("frame_of_reference"),
            "scene_id": item.get("scene_id"),
            "question_style": item.get("question_style"),
            "gold": item.get("answer"),
        }
        try:
            images = resolve_item_images(item, paths)
            bundle = wrap_item_for_eval(item)
            question = format_user_text(item)
            max_tokens = max_new_tokens_mcq if is_mcq(item) else max_new_tokens_nav
            result = runner.generate(
                images,
                system=bundle["system"],
                question=question,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
            )
            score = score_item(item, result.text)
            record.update(
                {
                    "n_images": result.n_images,
                    "image_paths": [str(p) for p in images],
                    "score": score,
                }
            )
        except Exception as exc:  # keep going; one bad frame must not kill the run
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["score"] = None
        append_jsonl(predictions_path, record)
        if item_id:
            done.add(item_id)
    records = read_jsonl(predictions_path)
    write_summary(
        run_dir,
        records,
        extra={"run_dir": str(run_dir), "predictions": str(predictions_path)},
    )
    return records
