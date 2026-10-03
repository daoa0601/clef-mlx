"""Parity against the PyTorch reference shipped in the release (``joint_schema_model.py``).

Needs the release files in the HF cache (``hf download Cloudflare/clef-flash``) and torch
(dev dependency). The backbone is not exercised here; see ``scripts/compare_backbone.py``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys

import mlx.core as mx
import numpy as np
import pytest
from pathlib import Path

torch = pytest.importorskip("torch")
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from clef_mlx.encoding import encode_record  # noqa: E402
from clef_mlx.head import JointSchemaHead  # noqa: E402

REPO = os.environ.get("CLEF_REPO", "Cloudflare/clef-flash")

RECORDS = [
    {
        "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
        "questions": {
            "status": {
                "type": "choice",
                "instructions": "What is the invoice status?",
                "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."},
            },
            "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
        },
    },
    {
        "state": "Our checkout started returning errors and orders are blocked.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle the message?",
                "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
            },
            "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
            "outage": {"type": "noul"},
        },
    },
]


@pytest.fixture(scope="module")
def release():
    try:
        path = snapshot_download(
            REPO, allow_patterns=["*.py", "*.json", "joint_head.safetensors"], local_files_only=True
        )
    except Exception:
        pytest.skip(f"{REPO} is not in the local HF cache")
    spec = importlib.util.spec_from_file_location("joint_schema_model", f"{path}/joint_schema_model.py")
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    return path, reference, AutoTokenizer.from_pretrained(path)


@pytest.mark.parametrize("record", RECORDS)
def test_encoding_matches_reference(release, record):
    _, reference, tokenizer = release
    expected = reference.encode_record(tokenizer, record)
    actual = encode_record(tokenizer, record)
    assert actual.input_ids == expected.input_ids
    assert [dataclasses.astuple(q) for q in actual.questions] == [
        dataclasses.astuple(q) for q in expected.questions
    ]


@pytest.mark.parametrize("record", RECORDS)
def test_head_logits_match_reference(release, record):
    path, reference, tokenizer = release
    config = json.loads((Path(path) / "joint_head_config.json").read_text())
    torch_head = reference.JointSchemaHead(**config)
    torch_head.load_state_dict(load_file(f"{path}/joint_head.safetensors"), strict=True)
    torch_head = torch_head.float().eval()
    mlx_head = JointSchemaHead.from_release(path)

    encoded = encode_record(tokenizer, record)
    # Remap token ids to a compact table so the fake LM head stays small.
    vocab = sorted(set(encoded.input_ids))
    compact = dataclasses.replace(encoded, input_ids=tuple(vocab.index(i) for i in encoded.input_ids))
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((len(compact.input_ids), config["hidden_size"]), dtype=np.float32) * 2
    embedding = rng.standard_normal((len(vocab), config["hidden_size"]), dtype=np.float32) * 0.02

    with torch.inference_mode():
        expected = torch_head(
            torch.from_numpy(hidden)[None],
            torch.tensor(compact.input_ids)[None],
            torch.ones(1, len(compact.input_ids), dtype=torch.long),
            [compact],
            torch.from_numpy(embedding),
        )[0]
    table = mx.array(embedding)
    actual = mlx_head(mx.array(hidden), compact, lambda ids: table[ids])

    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(np.array(got), want.numpy(), rtol=1e-4, atol=1e-4)
