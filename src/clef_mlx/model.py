"""Load Clef / Clef-Flash on MLX and answer typed questions."""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .backbone import TextBackbone, VisionBackbone, has_vision_weights
from .encoding import QUESTION_TYPES, EncodedRecord, encode_record
from .head import JointSchemaHead

HEAD_FILES = ("joint_head.safetensors", "joint_head_config.json")


def resolve(path_or_repo: str | Path) -> Path:
    path = Path(path_or_repo)
    if path.is_dir():
        return path
    # Skip the PyTorch helper; everything else (backbone, head, tokenizer) is needed.
    return Path(snapshot_download(str(path_or_repo), ignore_patterns=["*.py"]))


def _writable_view(source: Path, staging: Path) -> Path:
    """Mirror ``source`` with writable copies of its small files and links to everything else.

    mlx-vlm's convert copies ``*.json``/``*.py`` with their permissions and then overwrites
    ``tokenizer.json`` via ``save_pretrained``; HF cache blobs are read-only, so that fails.
    """
    for item in source.iterdir():
        if item.suffix in (".json", ".py", ".jinja"):
            shutil.copyfile(item, staging / item.name)
        else:
            (staging / item.name).symlink_to(item.resolve())
    return staging


def vision_available() -> bool:
    return importlib.util.find_spec("mlx_vlm") is not None


@dataclass
class Clef:
    backbone: TextBackbone | VisionBackbone
    head: JointSchemaHead
    tokenizer: Any
    processor: Any | None = None

    def encode(self, record: dict[str, Any], **kwargs: Any) -> EncodedRecord:
        return encode_record(self.tokenizer, record, processor=self.processor, **kwargs)

    def output_embedding_rows(self, ids: mx.array) -> mx.array:
        """Rows of the LM head weight, dequantized when the backbone is quantized."""
        lm_head = self.backbone.lm_head
        if isinstance(lm_head, nn.QuantizedLinear):
            return mx.dequantize(
                lm_head.weight[ids],
                lm_head.scales[ids],
                None if lm_head.biases is None else lm_head.biases[ids],
                group_size=lm_head.group_size,
                bits=lm_head.bits,
                mode=lm_head.mode,
            )
        return lm_head.weight[ids]

    def hidden_states(self, encoded: EncodedRecord) -> mx.array:
        """Final normed hidden states (HF ``last_hidden_state``) for one record, ``[length, hidden]``."""
        return self.backbone(encoded)

    def logits(self, encoded: EncodedRecord) -> list[mx.array]:
        logits = self.head(self.hidden_states(encoded), encoded, self.output_embedding_rows)
        mx.eval(logits)
        return logits

    def probabilities(self, record: dict[str, Any], **kwargs: Any) -> dict[str, dict[str, float]]:
        encoded = self.encode(record, **kwargs)
        return {
            question.question_id: dict(zip(question.option_ids, mx.softmax(logits, axis=-1).tolist()))
            for question, logits in zip(encoded.questions, self.logits(encoded))
        }

    def systemone(self, request: dict[str, Any], max_length: int = 16384) -> dict[str, Any]:
        """Answer a Jev/SystemOne ``/v1/systemone`` request body; ``images`` are PIL images."""
        questions = request.get("questions")
        if not isinstance(request.get("model"), str) or "state" not in request:
            raise ValueError("model and state are required")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("at least one question is required")
        for question_id, question in questions.items():
            if question.get("type") not in QUESTION_TYPES:
                raise ValueError(f"{question_id}: type must be noul, choice, or score")
            if question["type"] != "noul" and not question.get("criteria"):
                raise ValueError(f"{question_id}: criteria must not be empty")
        encoded = self.encode(request, max_length=max_length)
        answers = {
            question.question_id: systemone_answer(
                questions[question.question_id],
                dict(zip(question.option_ids, mx.softmax(logits.astype(mx.float32), axis=-1).tolist())),
            )
            for question, logits in zip(encoded.questions, self.logits(encoded))
        }
        return {
            "model": request["model"],
            "answers": answers,
            "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
        }


def systemone_answer(question: dict[str, Any], probabilities: dict[str, float]) -> dict[str, Any]:
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = [str(option) for option in question["criteria"]]
        choice = max(options, key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": round(probabilities[choice], 4),
            "probabilities": {option: round(probabilities[option], 4) for option in options},
        }
    levels = [str(index) for index in range(len(question["criteria"]))]
    return {
        "type": "score",
        "score": round(sum(index * probabilities[level] for index, level in enumerate(levels)), 4),
        "confidence": round(max(probabilities[level] for level in levels), 4),
        "legend": dict(zip(levels, question["criteria"])),
        "probabilities": {level: round(probabilities[level], 4) for level in levels},
    }


def load(path_or_repo: str | Path, lazy: bool = False, vision: bool | None = None) -> Clef:
    """Load a Clef release (HF repo id or local dir) or a directory written by :func:`convert`.

    ``vision=None`` keeps the vision tower whenever the checkpoint has one and the ``vision``
    extra (mlx-vlm) is installed; ``False`` forces the lighter text-only mlx-lm backbone.
    """
    path = resolve(path_or_repo)
    head = JointSchemaHead.from_release(path)
    if vision is None:
        vision = vision_available() and has_vision_weights(path)
    if not vision:
        return Clef(TextBackbone(path, lazy), head, AutoTokenizer.from_pretrained(path))
    if not has_vision_weights(path):
        raise ValueError(f"{path} has no vision weights; it was converted text-only")
    # mlx-vlm's numpy port of the Qwen3-VL processor: same tokens and pixels as HF, no torch.
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    processor = Qwen3VLProcessor.from_pretrained(str(path))
    return Clef(VisionBackbone(path, lazy), head, processor.tokenizer, processor)


def convert(
    path_or_repo: str | Path,
    mlx_path: str | Path,
    bits: int = 8,
    group_size: int = 64,
    quantize_lm_head: bool = False,
    vision: bool | None = None,
) -> Path:
    """Quantize the backbone and copy the joint head next to it.

    With vision (the default when mlx-vlm is installed) the conversion goes through mlx-vlm and
    keeps the vision tower in bf16; otherwise mlx-lm writes a smaller text-only checkpoint. The
    LM head stays in bf16 by default: the joint head reads its rows directly as option
    features, so quantization noise there feeds straight into the logits.
    """
    source = resolve(path_or_repo)
    mlx_path = Path(mlx_path)
    if mlx_path.exists():
        raise ValueError(f"{mlx_path} already exists")
    if vision is None:
        vision = vision_available()

    def keep_lm_head(module_path: str) -> bool:
        return not quantize_lm_head and module_path.endswith("lm_head")

    if vision:
        from mlx_vlm.convert import convert as mlx_vlm_convert
        from mlx_vlm.utils import skip_multimodal_module

        with tempfile.TemporaryDirectory() as staging:
            mlx_vlm_convert(
                str(_writable_view(source, Path(staging))),
                mlx_path=str(mlx_path),
                quantize=True,
                q_bits=bits,
                q_group_size=group_size,
                quant_predicate=lambda p, _m: not (skip_multimodal_module(p) or keep_lm_head(p)),
            )
    else:
        from mlx_lm import convert as mlx_lm_convert

        mlx_lm_convert(
            str(source),
            mlx_path=str(mlx_path),
            quantize=True,
            q_bits=bits,
            q_group_size=group_size,
            quant_predicate=lambda p, _m, *_: not keep_lm_head(p),
        )
    # The head, plus the release's processor config (mlx-vlm rewrites it with its own video defaults).
    for name in (*HEAD_FILES, "processor_config.json"):
        if (source / name).exists():
            shutil.copyfile(source / name, mlx_path / name)
    return mlx_path
