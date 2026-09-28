"""Thin wrapper around the smoke CLI (first Qwen3-VL proof).

    python test_qwen.py \\
      --data-root /path/to/eval-data \\
      --frames-root /path/to/navigation \\
      --episode 09_23_2026_16_31_04_526137
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cm_evaluation.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-VL sequential-image smoke test")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory for the Hugging Face cache / model weights.",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        required=True,
        help="Root whose children are episode folders with images/img_<t>.png.",
    )
    parser.add_argument(
        "--episode",
        required=True,
        help="Episode folder name under --frames-root.",
    )
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--question",
        default="Where is the Window relative to you right now?",
    )
    args, extra = parser.parse_known_args()
    images = [
        args.frames_root / args.episode / "images" / f"img_{i}.png" for i in args.frames
    ]
    argv = [
        "smoke",
        "--data-root",
        str(args.data_root),
        "--images",
        *[str(p) for p in images],
        "--question",
        args.question,
        *extra,
    ]
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
