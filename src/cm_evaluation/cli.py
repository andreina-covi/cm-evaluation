"""CLI: print-paths, download-model, smoke, evaluate.

Paths are always passed on the command (cm-benchmark style). There are no
machine-specific defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

from cm_evaluation.paths import (
    ENV_KEEP_HF_ENV,
    QWEN3_VL_8B_DISK_GB,
    EvalPaths,
    load_eval_config,
    pin_hf_env,
    resolve_paths,
)


def _add_data_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory for model weights (HF cache) and, unless overridden, results.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Where run JSONL/stats are written (default: <data-root>/results).",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help="Hugging Face home (default: <data-root>/models/hf).",
    )


def _add_frames_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=None,
        help="Root of episode image folders. Relative item image_paths are resolved here.",
    )


def _paths_from_ns(ns: argparse.Namespace) -> EvalPaths:
    return resolve_paths(
        data_root=ns.data_root,
        frames_root=getattr(ns, "frames_root", None),
        results_root=getattr(ns, "results_root", None),
        hf_home=getattr(ns, "hf_home", None),
    )


def cmd_print_paths(ns: argparse.Namespace) -> int:
    paths = _paths_from_ns(ns)
    cfg = load_eval_config()
    payload = {
        "data_root": str(paths.data_root),
        "hf_home": str(paths.hf_home),
        "hf_hub_cache": str(paths.hf_hub_cache),
        "models_root": str(paths.models_root),
        "results_root": str(paths.results_root),
        "frames_root": str(paths.frames_root) if paths.frames_root else None,
        "keep_hf_env": os.environ.get(ENV_KEEP_HF_ENV) == "1",
        "model_id": cfg.get("model_id"),
        "approx_model_disk_gb": QWEN3_VL_8B_DISK_GB,
        "min_free_gb": cfg.get("min_free_gb"),
        "disk": paths.disk_usage(),
        "layout": {
            "weights": f"{paths.hf_hub_cache}/models--Qwen--Qwen3-VL-8B-Instruct/",
            "frames": "<frames-root>/<episode>/images/img_<t>.png",
            "runs": f"{paths.results_root}/qwen3-vl-8b-instruct/run_<stamp>/",
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_download_model(ns: argparse.Namespace) -> int:
    paths = _paths_from_ns(ns)
    paths.ensure_layout()
    applied = pin_hf_env(paths)
    cfg = load_eval_config()
    model_id = ns.model_id or cfg.get("model_id") or "Qwen/Qwen3-VL-8B-Instruct"
    min_free = float(ns.min_free_gb if ns.min_free_gb is not None else cfg.get("min_free_gb") or 25)
    usage = shutil.disk_usage(paths.data_root)
    free_gb = usage.free / (1024**3)
    print(
        f"data_root={paths.data_root}\n"
        f"HF_HOME={applied.get('HF_HOME', '(kept)')}\n"
        f"free_gb={free_gb:.1f} required_gb={min_free:.1f} model={model_id}"
    )
    if free_gb < min_free and not ns.force:
        print(
            f"Refusing download: only {free_gb:.1f} GB free on {paths.data_root}. "
            f"Need ~{QWEN3_VL_8B_DISK_GB:.0f} GB for Qwen3-VL-8B-Instruct plus slack "
            f"(threshold {min_free:.0f} GB). Use --force to override.",
            file=sys.stderr,
        )
        return 2

    from huggingface_hub import snapshot_download

    local_dir = paths.snapshot_dir(model_id) if ns.snapshot else None
    if local_dir is not None:
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        print(
            "WARNING: --snapshot writes a second copy besides HF_HUB_CACHE. "
            "Skip it unless you need a portable folder.",
            file=sys.stderr,
        )
    dest = snapshot_download(
        repo_id=model_id,
        cache_dir=str(paths.hf_hub_cache),
        local_dir=str(local_dir) if local_dir else None,
        local_files_only=ns.offline,
    )
    print(f"model ready at {dest}")
    return 0


def cmd_smoke(ns: argparse.Namespace) -> int:
    paths = _paths_from_ns(ns)
    paths.ensure_layout()
    pin_hf_env(paths)
    cfg = load_eval_config()
    images = [Path(p) for p in ns.images]
    missing = [p for p in images if not p.is_file()]
    if missing:
        print(f"Missing images: {missing}", file=sys.stderr)
        return 2
    from cm_evaluation.models.qwen3_vl import Qwen3VLRunner
    from cm_evaluation.protocol import SYSTEM_INSTRUCTION

    runner = Qwen3VLRunner(
        ns.model_id or cfg.get("model_id") or "Qwen/Qwen3-VL-8B-Instruct",
        dtype=cfg.get("dtype") or "auto",
        device_map=cfg.get("device_map") or "auto",
        attn_implementation=cfg.get("attn_implementation"),
        local_files_only=ns.offline,
    )
    result = runner.generate(
        images,
        system=SYSTEM_INSTRUCTION,
        question=ns.question,
        max_new_tokens=int(cfg.get("max_new_tokens_smoke") or 128),
        do_sample=False,
    )
    print(result.text)
    return 0


def cmd_evaluate(ns: argparse.Namespace) -> int:
    paths = _paths_from_ns(ns)
    paths.ensure_layout()
    pin_hf_env(paths)
    cfg = load_eval_config()
    items_path = ns.items
    if not Path(items_path).exists():
        print(f"No items at {items_path}.", file=sys.stderr)
        return 2
    constructs = {c.strip() for c in (ns.constructs or "").split(",") if c.strip()} or None
    from cm_evaluation.items import iter_items
    from cm_evaluation.models.qwen3_vl import Qwen3VLRunner
    from cm_evaluation.run import evaluate_items, new_run_dir

    model_id = ns.model_id or cfg.get("model_id") or "Qwen/Qwen3-VL-8B-Instruct"
    slug = model_id.split("/")[-1].lower()
    run_dir = Path(ns.run_dir) if ns.run_dir else new_run_dir(paths.results_root, slug)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "items": str(items_path),
                "constructs": sorted(constructs) if constructs else None,
                "limit": ns.limit,
                "include_class4": ns.include_class4,
                "data_root": str(paths.data_root),
                "frames_root": str(paths.frames_root) if paths.frames_root else None,
            },
            indent=2,
        )
    )
    print(f"run_dir={run_dir}")
    runner = Qwen3VLRunner(
        model_id,
        dtype=cfg.get("dtype") or "auto",
        device_map=cfg.get("device_map") or "auto",
        attn_implementation=cfg.get("attn_implementation"),
        local_files_only=ns.offline,
    )
    items = list(
        iter_items(
            items_path,
            constructs=constructs,
            include_class4=ns.include_class4,
            limit=ns.limit,
        )
    )
    if not items:
        print(f"No ok items under {items_path}", file=sys.stderr)
        return 2
    records = evaluate_items(
        runner,
        items,
        paths=paths,
        run_dir=run_dir,
        max_new_tokens_mcq=int(cfg.get("max_new_tokens_mcq") or 16),
        max_new_tokens_nav=int(cfg.get("max_new_tokens_nav") or 128),
        do_sample=bool(cfg.get("do_sample") or False),
        resume=not ns.no_resume,
    )
    summary_path = run_dir / "summary.json"
    print(summary_path.read_text() if summary_path.is_file() else f"n={len(records)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cm-eval",
        description="Evaluate cm-benchmark items on Qwen3-VL (and later other VLMs).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_paths = sub.add_parser("print-paths", help="Show the layout implied by --data-root")
    _add_data_root(p_paths)
    _add_frames_root(p_paths)
    p_paths.set_defaults(func=cmd_print_paths)

    p_dl = sub.add_parser("download-model", help="Download Qwen3-VL-8B-Instruct into --data-root")
    _add_data_root(p_dl)
    p_dl.add_argument("--model-id", default=None)
    p_dl.add_argument("--min-free-gb", type=float, default=None)
    p_dl.add_argument("--force", action="store_true")
    p_dl.add_argument("--offline", action="store_true")
    p_dl.add_argument(
        "--snapshot",
        action="store_true",
        help="Also copy weights to models/snapshots/ (uses extra disk; off by default)",
    )
    p_dl.set_defaults(func=cmd_download_model)

    p_smoke = sub.add_parser("smoke", help="Sequential images + one spatial question (no item JSON)")
    _add_data_root(p_smoke)
    p_smoke.add_argument("--images", nargs="+", required=True)
    p_smoke.add_argument("--question", required=True)
    p_smoke.add_argument("--model-id", default=None)
    p_smoke.add_argument("--offline", action="store_true")
    p_smoke.set_defaults(func=cmd_smoke)

    p_eval = sub.add_parser("evaluate", help="Score frozen items; write JSONL + per-construct stats")
    _add_data_root(p_eval)
    _add_frames_root(p_eval)
    p_eval.add_argument(
        "--items",
        type=Path,
        required=True,
        help="generate_items output: items JSON file, one scene folder, or a root of scene folders.",
    )
    p_eval.add_argument("--constructs", default=None, help="Comma-separated construct ids")
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--include-class4", action="store_true")
    p_eval.add_argument("--run-dir", type=Path, default=None)
    p_eval.add_argument("--model-id", default=None)
    p_eval.add_argument("--offline", action="store_true")
    p_eval.add_argument("--no-resume", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
