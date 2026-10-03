"""Torch-free port of Clef's record encoding (``encode_record`` in ``joint_schema_model.py``).

The token layout must match the release exactly: the joint head reads hidden states at the
question and option spans computed here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
MEDIA_KEYS = ("pixel_values", "image_grid_thw")
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    question_type = str(question["type"])
    if question_type == "noul":
        criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]
    record_id: str
    media: dict[str, np.ndarray] | None = dataclass_field(default=None, compare=False, repr=False)


def _tokens(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _encode_media(processor: Any, record: dict[str, Any]) -> tuple[list[int], dict[str, np.ndarray] | None]:
    """Expand one placeholder per image into image-pad tokens and return the pixel inputs."""
    if record.get("videos"):
        raise NotImplementedError("clef-mlx does not support video inputs yet")
    images = list(record.get("images") or [])
    if not images:
        return [], None
    if processor is None:
        raise ValueError(
            "records with images need a vision backbone: install clef-mlx[vision] and load a release "
            "or a checkpoint converted without --text-only"
        )
    encoded = processor(
        text=[IMAGE_PLACEHOLDER * len(images) + "\n"],
        images=images,
        return_tensors="np",
        **(record.get("media_kwargs") or {}),
    )
    media = {key: np.asarray(encoded[key]) for key in MEDIA_KEYS}
    return np.asarray(encoded["input_ids"])[0].tolist(), media


def encode_record(
    tokenizer: Any,
    record: dict[str, Any],
    max_length: int = 16384,
    max_state_tokens: int | None = None,
    processor: Any | None = None,
) -> EncodedRecord:
    schema_ids = _tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    questions: list[EncodedQuestion] = []
    for question_index, (question_id, question) in enumerate(record["questions"].items()):
        schema_ids.extend(
            _tokens(
                tokenizer,
                f"\nFIELD {question_index + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: ",
            )
        )
        question_start = len(schema_ids)
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids.extend(_tokens(tokenizer, render(instructions)))
        question_end = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, "\nALLOWED OPTIONS:\n"))

        option_spans: list[tuple[int, int]] = []
        option_ids: list[str] = []
        for option_index, (option_id, description) in enumerate(question_options(question)):
            schema_ids.extend(_tokens(tokenizer, f"OPTION {option_index + 1}: "))
            option_start = len(schema_ids)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(_tokens(tokenizer, render(semantics)))
            option_spans.append((option_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids.extend(_tokens(tokenizer, "\n"))
        schema_ids.extend(_tokens(tokenizer, "END FIELD\n"))
        questions.append(
            EncodedQuestion(
                question_id=str(question_id),
                question_type=QUESTION_TYPES[str(question["type"])],
                question_span=(question_start, question_end),
                option_spans=tuple(option_spans),
                option_ids=tuple(option_ids),
            )
        )

    prefix_ids = _tokens(
        tokenizer,
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n",
    )
    suffix_ids = _tokens(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )
    media_ids, media = _encode_media(processor, record)
    prefix_ids = prefix_ids + media_ids
    state_ids = _tokens(tokenizer, render(record["state"]))
    if max_state_tokens is not None:
        state_ids = state_ids[:max_state_tokens]
    fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed_length > max_length:
        raise ValueError(f"schema requires {fixed_length} tokens before state; maximum is {max_length}")
    state_ids = state_ids[: max_length - fixed_length]
    offset = len(prefix_ids) + len(state_ids)
    shifted = tuple(
        EncodedQuestion(
            question_id=q.question_id,
            question_type=q.question_type,
            question_span=(q.question_span[0] + offset, q.question_span[1] + offset),
            option_spans=tuple((s + offset, e + offset) for s, e in q.option_spans),
            option_ids=q.option_ids,
        )
        for q in questions
    )
    input_ids = tuple(prefix_ids + state_ids + schema_ids + suffix_ids)
    if not input_ids or not shifted:
        raise ValueError("record produced no model input or questions")
    return EncodedRecord(
        input_ids=input_ids, questions=shifted, record_id=str(record.get("id", "unknown")), media=media
    )
