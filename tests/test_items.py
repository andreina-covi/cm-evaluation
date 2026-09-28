from pathlib import Path

import pytest

from cm_evaluation.items import iter_items, load_item_file, resolve_image_path
from cm_evaluation.protocol import format_user_text, is_mcq, wrap_item_for_eval


def test_load_and_filter_class4(tiny_items_json) -> None:
    items = load_item_file(tiny_items_json)
    assert len(items) == 3
    mcq_only = list(iter_items(tiny_items_json, include_class4=False))
    assert {i["item_id"] for i in mcq_only} == {"tiny-ego-001", "tiny-swm-001"}
    all_ok = list(iter_items(tiny_items_json, include_class4=True))
    assert len(all_ok) == 3


def test_construct_filter_and_limit(tiny_items_json) -> None:
    rows = list(
        iter_items(
            tiny_items_json,
            constructs={"egocentric_encoding"},
            include_class4=True,
            limit=1,
        )
    )
    assert len(rows) == 1
    assert rows[0]["item_id"] == "tiny-ego-001"


def test_format_appends_options() -> None:
    item = {
        "question": "Where is the Window relative to you?",
        "options": {"A": "left", "B": "right"},
    }
    text = format_user_text(item)
    assert "A. left" in text
    assert wrap_item_for_eval(item)["question"] == item["question"]


def test_is_mcq(tiny_items_json) -> None:
    items = {i["item_id"]: i for i in load_item_file(tiny_items_json)}
    assert is_mcq(items["tiny-ego-001"])
    assert not is_mcq(items["tiny-route-001"])


def test_resolve_image_path_against_frames_root(tmp_path: Path) -> None:
    image = tmp_path / "ep" / "images" / "img_0.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    got = resolve_image_path("ep/images/img_0.png", frames_root=tmp_path)
    assert got == image.resolve()
    with pytest.raises(FileNotFoundError):
        resolve_image_path("missing.png", frames_root=tmp_path)
