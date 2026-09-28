"""Load cm-benchmark item JSON (``items_<scene>.json`` roots)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional


def list_item_jsons(folder: str | Path) -> list[Path]:
    """Same layout as cm-benchmark ``generate_items --output_path``."""
    path = Path(folder)
    if path.is_file():
        if path.suffix != ".json":
            raise FileNotFoundError(f"Expected an items JSON file or folder, got {path}")
        return [path.resolve()]
    if not path.is_dir():
        raise NotADirectoryError(path)

    nested = sorted(
        child
        for child in path.glob("*/*.json")
        if child.is_file() and child.name.startswith("items_")
    )
    if nested:
        return [child.resolve() for child in nested]

    direct = sorted(
        child
        for child in path.glob("*.json")
        if child.is_file() and child.name.startswith("items_")
    )
    if direct:
        return [child.resolve() for child in direct]

    raise FileNotFoundError(
        f"No items_*.json under {path}. Pass generate_items output "
        "(items/<scene_id>/items_<scene_id>.json)."
    )


def load_item_file(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "items" in payload:
        return list(payload["items"])
    raise ValueError(f"Unrecognized items payload in {path}")


def iter_items(
    folder: str | Path,
    *,
    constructs: Optional[set[str]] = None,
    status: str = "ok",
    include_class4: bool = False,
    limit: Optional[int] = None,
) -> Iterator[dict[str, Any]]:
    from cm_evaluation.protocol import NAV_CONSTRUCTS

    yielded = 0
    for json_path in list_item_jsons(folder):
        for item in load_item_file(json_path):
            if status and item.get("status") != status:
                continue
            construct = item.get("construct")
            if constructs and construct not in constructs:
                continue
            if not include_class4 and construct in NAV_CONSTRUCTS:
                continue
            yield item
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def resolve_image_path(
    raw: str,
    *,
    frames_root: Optional[Path] = None,
    extra_roots: Optional[list[Path]] = None,
) -> Path:
    """Resolve an item ``image_paths`` entry against ``frames_root``."""
    path = Path(raw)
    if path.is_file():
        return path.resolve()

    candidates: list[Path] = []
    roots: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    if frames_root is not None:
        roots.append(frames_root)
    if extra_roots:
        roots.extend(extra_roots)
    if not path.is_absolute():
        for root in roots:
            candidates.append(root / raw)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    name = Path(raw).name
    if name.startswith("img_"):
        for root in roots:
            matches = list(root.glob(f"**/images/{name}"))
            if len(matches) == 1:
                return matches[0].resolve()
    raise FileNotFoundError(f"Image not found: {raw}")
