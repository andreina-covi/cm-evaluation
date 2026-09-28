from pathlib import Path

from cm_evaluation.items import load_item_file
from cm_evaluation.models.qwen3_vl import GenerationResult
from cm_evaluation.paths import EvalPaths
from cm_evaluation.run import evaluate_items, load_done_ids


class FakeRunner:
    def generate(self, image_paths, *, system, question, max_new_tokens, do_sample=False):
        return GenerationResult(text="A", n_images=len(list(image_paths)), max_new_tokens=max_new_tokens)


def test_evaluate_loop_resume(tmp_path: Path, tiny_items_json) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "img_0.png").write_bytes(b"png")
    (frames / "img_1.png").write_bytes(b"png")

    items = [i for i in load_item_file(tiny_items_json) if i["construct"] != "route_knowledge"]
    for item in items:
        item["image_paths"] = [str(frames / Path(p).name) for p in item["image_paths"]]

    paths = EvalPaths(
        data_root=tmp_path,
        hf_home=tmp_path / "hf",
        hf_hub_cache=tmp_path / "hf" / "hub",
        results_root=tmp_path / "results",
        frames_root=tmp_path,
    )
    run_dir = tmp_path / "run"
    records = evaluate_items(
        FakeRunner(),
        items,
        paths=paths,
        run_dir=run_dir,
        max_new_tokens_mcq=16,
        max_new_tokens_nav=128,
    )
    assert len(records) == 2
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "by_construct.csv").is_file()
    done = load_done_ids(run_dir / "predictions.jsonl")
    assert done == {"tiny-ego-001", "tiny-swm-001"}

    again = evaluate_items(
        FakeRunner(),
        items,
        paths=paths,
        run_dir=run_dir,
        max_new_tokens_mcq=16,
        max_new_tokens_nav=128,
        resume=True,
    )
    assert len(again) == 2
    lines = (run_dir / "predictions.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
