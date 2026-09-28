#!/usr/bin/env python3
"""Download Qwen3-VL-8B-Instruct into --data-root (not ~/.cache)."""

import sys

from cm_evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["download-model", *sys.argv[1:]]))
