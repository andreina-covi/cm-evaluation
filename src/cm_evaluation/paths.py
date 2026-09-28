"""Paths for weights, frames, items, and results.

Nothing is machine-specific. Callers pass roots into ``resolve_paths`` or the
CLI (same style as cm-benchmark: ``--csv_path_folder``, ``--output_path``).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_KEEP_HF_ENV = "CM_EVAL_KEEP_HF_ENV"

# Qwen3-VL-8B-Instruct snapshot is ~17.5 GB on disk (4 safetensor shards).
QWEN3_VL_8B_DISK_GB = 18.0


@dataclass(frozen=True)
class EvalPaths:
    data_root: Path
    hf_home: Path
    hf_hub_cache: Path
    results_root: Path
    frames_root: Optional[Path] = None

    @property
    def models_root(self) -> Path:
        return self.data_root / "models"

    def snapshot_dir(self, model_id: str) -> Path:
        """Optional portable copy: ``models/snapshots/<org>--<name>``."""
        slug = model_id.replace("/", "--")
        return self.models_root / "snapshots" / slug

    def ensure_layout(self) -> None:
        for path in (
            self.data_root,
            self.models_root,
            self.hf_home,
            self.hf_hub_cache,
            self.results_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def disk_usage(self) -> dict[str, float]:
        probe = self.data_root
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        return {
            "total_gb": usage.total / (1024**3),
            "used_gb": usage.used / (1024**3),
            "free_gb": usage.free / (1024**3),
        }


def _as_path(value: str | Path | None) -> Optional[Path]:
    if value is None or value == "":
        return None
    return Path(str(value)).expanduser()


def resolve_paths(
    *,
    data_root: str | Path,
    frames_root: str | Path | None = None,
    results_root: str | Path | None = None,
    hf_home: str | Path | None = None,
) -> EvalPaths:
    """Build path layout from caller-supplied roots.

    ``data_root`` is required. Other roots default under it:

    - ``hf_home`` → ``<data_root>/models/hf``
    - ``results_root`` → ``<data_root>/results``
    """
    root = _as_path(data_root)
    if root is None:
        raise ValueError("data_root is required")
    root = root.resolve()
    frames = _as_path(frames_root)
    hf = _as_path(hf_home) or (root / "models" / "hf")
    results = _as_path(results_root) or (root / "results")
    hf = hf.resolve()
    return EvalPaths(
        data_root=root,
        hf_home=hf,
        hf_hub_cache=(hf / "hub").resolve(),
        results_root=results.resolve(),
        frames_root=frames.resolve() if frames is not None else None,
    )


def pin_hf_env(paths: EvalPaths) -> dict[str, str]:
    """Point Hugging Face / Transformers caches at ``paths.hf_home``.

    Call this before importing ``transformers``. Skipped when
    ``CM_EVAL_KEEP_HF_ENV=1``.
    """
    applied: dict[str, str] = {}
    if os.environ.get(ENV_KEEP_HF_ENV) == "1":
        for key in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
            if key in os.environ:
                applied[key] = os.environ[key]
        return applied

    mapping = {
        "HF_HOME": str(paths.hf_home),
        "HF_HUB_CACHE": str(paths.hf_hub_cache),
        "HUGGINGFACE_HUB_CACHE": str(paths.hf_hub_cache),
        "TRANSFORMERS_CACHE": str(paths.hf_home / "transformers"),
    }
    for key, value in mapping.items():
        os.environ[key] = value
        applied[key] = value
    return applied


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    return data


def load_eval_config(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or REPO_ROOT
    return _load_yaml(root / "configs" / "eval.yaml")
