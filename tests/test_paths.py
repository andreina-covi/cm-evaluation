from pathlib import Path

import pytest

from cm_evaluation.paths import pin_hf_env, resolve_paths


def test_resolve_paths_requires_data_root() -> None:
    with pytest.raises((TypeError, ValueError)):
        resolve_paths()  # type: ignore[call-arg]


def test_resolve_paths_from_argument(tmp_path: Path) -> None:
    paths = resolve_paths(data_root=tmp_path / "eval-data")
    assert paths.data_root == (tmp_path / "eval-data").resolve()
    assert paths.hf_home == paths.data_root / "models" / "hf"
    assert paths.results_root == paths.data_root / "results"
    assert paths.frames_root is None


def test_resolve_paths_overrides(tmp_path: Path) -> None:
    frames = tmp_path / "nav"
    results = tmp_path / "out"
    hf = tmp_path / "hf-custom"
    paths = resolve_paths(
        data_root=tmp_path / "eval-data",
        frames_root=frames,
        results_root=results,
        hf_home=hf,
    )
    assert paths.frames_root == frames.resolve()
    assert paths.results_root == results.resolve()
    assert paths.hf_home == hf.resolve()
    assert paths.hf_hub_cache == (hf / "hub").resolve()


def test_pin_hf_env_overwrites_home_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CM_EVAL_KEEP_HF_ENV", raising=False)
    monkeypatch.setenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    paths = resolve_paths(data_root=tmp_path / "eval-data")
    applied = pin_hf_env(paths)
    assert applied["HF_HOME"] == str(paths.hf_home)
    assert str(paths.data_root) in applied["HF_HUB_CACHE"]


def test_pin_hf_env_keep(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CM_EVAL_KEEP_HF_ENV", "1")
    monkeypatch.setenv("HF_HOME", "/kept/hf")
    paths = resolve_paths(data_root=tmp_path / "eval-data")
    applied = pin_hf_env(paths)
    assert applied["HF_HOME"] == "/kept/hf"
