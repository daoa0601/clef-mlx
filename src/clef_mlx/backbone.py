"""Qwen3.5 backbones that turn an encoded record into final hidden states.

``TextBackbone`` uses mlx-lm and drops the vision tower. ``VisionBackbone`` uses mlx-vlm (the
``vision`` extra) and keeps it, so records may carry images.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .encoding import EncodedRecord


def has_vision_weights(path: Path) -> bool:
    index = path / "model.safetensors.index.json"
    if index.exists():
        names = json.loads(index.read_text())["weight_map"]
    else:
        names = [name for file in path.glob("model*.safetensors") for name in mx.load(str(file))]
    return any("visual" in name or "vision_tower" in name for name in names)


class TextBackbone:
    def __init__(self, path: Path, lazy: bool = False) -> None:
        from mlx_lm.utils import load_model

        self.model, _ = load_model(path, lazy=lazy)

    @property
    def lm_head(self) -> nn.Module:
        return self.model.language_model.lm_head

    def __call__(self, encoded: EncodedRecord) -> mx.array:
        if encoded.media:
            raise ValueError("this model was loaded text-only; images need the vision backbone")
        return self.model.model(mx.array(encoded.input_ids)[None])[0]


class VisionBackbone:
    def __init__(self, path: Path, lazy: bool = False) -> None:
        from mlx_vlm.utils import load_model

        self.model = load_model(path, lazy=lazy)

    @property
    def lm_head(self) -> nn.Module:
        return self.model.language_model.lm_head

    def __call__(self, encoded: EncodedRecord) -> mx.array:
        input_ids = mx.array(encoded.input_ids)[None]
        media = {key: mx.array(value) for key, value in (encoded.media or {}).items()}
        # Merges image features into the embeddings and computes the 3D (t, h, w) rope positions.
        features = self.model.get_input_embeddings(input_ids, **media)
        hidden = self.model.language_model.model(
            input_ids, inputs_embeds=features.inputs_embeds, position_ids=features.position_ids
        )
        return hidden[0]
