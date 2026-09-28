#!/usr/bin/env python3
"""Smoke: sequential first-person frames + one spatial question on Qwen3-VL."""

import sys

from cm_evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["smoke", *sys.argv[1:]]))
