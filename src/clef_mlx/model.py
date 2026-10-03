"""Load Clef / Clef-Flash on MLX and answer typed questions."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.utils import load_model
from transformers import AutoTokenizer

from .encoding import QUESTION_TYPES, EncodedRecord, encode_record
from .head import JointSchemaHead

HEAD_FILES = ("joint_head.safetensors", "joint_head_config.json")


def resolve(path_or_repo: str | Path) -> Path:
    path = Path(path_or_repo)
    if path.is_dir():
        return path
    # Skip the PyTorch helper; everything else (backbone, head, tokenizer) is needed.
    return Path(snapshot_download(str(path_or_repo), ignore_patterns=["*.py"]))


@dataclass
class Clef:
    backbone: nn.Module
    head: JointSchemaHead
    tokenizer: Any

    def encode(self, record: dict[str, Any], **kwargs: Any) -> EncodedRecord:
        return encode_record(self.tokenizer, record, **kwargs)

    def output_embedding_rows(self, ids: mx.array) -> mx.array:
        """Rows of the LM head weight, dequantized when the backbone is quantized."""
        lm_head = self.backbone.language_model.lm_head
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
        return self.backbone.model(mx.array(encoded.input_ids)[None])[0]

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
        """Answer a Jev/SystemOne ``/v1/systemone`` request body (text-only)."""
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


def load(path_or_repo: str | Path, lazy: bool = False) -> Clef:
    """Load a Clef release (HF repo id or local dir) or a directory written by :func:`convert`."""
    path = resolve(path_or_repo)
    backbone, _ = load_model(path, lazy=lazy)
    return Clef(
        backbone=backbone,
        head=JointSchemaHead.from_release(path),
        tokenizer=AutoTokenizer.from_pretrained(path),
    )


def convert(
    path_or_repo: str | Path,
    mlx_path: str | Path,
    bits: int = 8,
    group_size: int = 64,
    quantize_lm_head: bool = False,
) -> Path:
    """Quantize the backbone with mlx-lm and copy the joint head next to it.

    The LM head stays in bf16 by default: the joint head reads its rows directly as option
    features, so quantization noise there feeds straight into the logits.
    """
    from mlx_lm import convert as mlx_lm_convert

    source = resolve(path_or_repo)
    mlx_path = Path(mlx_path)

    def quant_predicate(module_path: str, _module: nn.Module, *_: Any) -> bool:
        return quantize_lm_head or not module_path.endswith("lm_head")

    mlx_lm_convert(
        str(source),
        mlx_path=str(mlx_path),
        quantize=True,
        q_bits=bits,
        q_group_size=group_size,
        quant_predicate=quant_predicate,
    )
    for name in HEAD_FILES:
        shutil.copy2(source / name, mlx_path / name)
    return mlx_path
