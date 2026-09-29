from cm_evaluation.models.qwen3_vl import _resolve_dtype
import pytest


def test_resolve_dtype_aliases() -> None:
    assert _resolve_dtype("auto") == "auto"
    assert _resolve_dtype("float16") == "float16"
    assert _resolve_dtype("torch.float16") == "float16"
    assert _resolve_dtype("bf16") == "bfloat16"


def test_resolve_dtype_unknown() -> None:
    with pytest.raises(ValueError):
        _resolve_dtype("int8")
