"""Qwen3-VL local runner (Transformers)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass
class GenerationResult:
    text: str
    n_images: int
    max_new_tokens: int


class Qwen3VLRunner:
    """Load once, generate many. Hugging Face caches must already be pinned."""

    def __init__(
        self,
        model_id: str,
        *,
        dtype: str = torch.float16,
        device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        local_files_only: bool = False,
        local_dir: Optional[Path] = None,
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        source = str(local_dir) if local_dir is not None else model_id
        kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": device_map,
            "local_files_only": local_files_only,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForImageTextToText.from_pretrained(source, **kwargs)
        self.processor = AutoProcessor.from_pretrained(
            source, local_files_only=local_files_only
        )
        self.model_id = model_id
        self.source = source

    def generate(
        self,
        image_paths: Sequence[str | Path],
        *,
        system: str,
        question: str,
        max_new_tokens: int,
        do_sample: bool = False,
    ) -> GenerationResult:
        content: list[dict[str, Any]] = []
        resolved: list[Path] = []
        for raw in image_paths:
            path = Path(raw).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            resolved.append(path)
            content.append({"type": "image", "image": path.as_uri()})
        content.append({"type": "text", "text": question})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        trimmed = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs.input_ids, generated_ids)
        ]
        text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return GenerationResult(
            text=text, n_images=len(resolved), max_new_tokens=max_new_tokens
        )
