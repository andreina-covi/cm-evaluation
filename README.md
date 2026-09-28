# cm-evaluation

Evaluate [cm-benchmark](https://github.com/andreina-covi/cm-benchmark) spatial-cognition items on vision-language models.

Items are sequential first-person frames plus a question. The first model is **Qwen3-VL-8B-Instruct**. Classes 1–3 are multiple-choice; the predicted letter is compared to ground truth. Each run reports accuracy per construct.

Qwen3-VL-8B-Instruct needs about 18 GB of disk and 20 GB of VRAM (BF16).

## Setup

Python 3.12. Install a CUDA-matched `torch`, then this package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

`transformers>=4.57` is required for Qwen3-VL.

## Commands

`--data-root` is the directory for model weights and run outputs. `--items` is the cm-benchmark `generate_items` output. `--frames-root` is the episode image root.

```bash
python -m cm_evaluation print-paths \
  --data-root /path/to/eval-data

python -m cm_evaluation download-model \
  --data-root /path/to/eval-data

python -m cm_evaluation smoke \
  --data-root /path/to/eval-data \
  --images /path/to/episode/images/img_0.png /path/to/episode/images/img_1.png \
  --question "Where is the Window relative to you right now?"

python -m cm_evaluation evaluate \
  --data-root /path/to/eval-data \
  --items /path/to/items \
  --frames-root /path/to/navigation \
  --limit 4 \
  --constructs egocentric_encoding
```

```text
<data-root>/
  models/hf/              # model cache
  results/<model>/run_*/  # predictions.jsonl, summary.json, by_construct.csv
```

## Tests

```bash
pytest tests/ -q
```
