#!/usr/bin/env python3
"""First evaluations: frozen cm-benchmark items on Qwen3-VL-8B-Instruct.

Requires --data-root and --items. Writes resume-safe JSONL under
``<data-root>/results/`` (or --results-root). Class 4 is skipped unless
``--include-class4``.
"""

import sys

from cm_evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
