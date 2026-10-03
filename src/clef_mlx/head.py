"""MLX port of Clef's ``JointSchemaHead``.

Module and parameter names mirror the PyTorch release so ``joint_head.safetensors`` loads
with only two key renames (``nn.Sequential`` children live under ``layers`` in MLX).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .encoding import EncodedRecord


class TorchMultiheadAttention(nn.Module):
    """``torch.nn.MultiheadAttention`` (batch_first, packed in-projection, with bias)."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.in_proj_weight = mx.zeros((3 * width, width))
        self.in_proj_bias = mx.zeros((3 * width,))
        self.out_proj = nn.Linear(width, width)

    def __call__(self, queries: mx.array, keys: mx.array, values: mx.array) -> mx.array:
        wq, wk, wv = mx.split(self.in_proj_weight, 3, axis=0)
        bq, bk, bv = mx.split(self.in_proj_bias, 3, axis=0)
        q, k, v = queries @ wq.T + bq, keys @ wk.T + bk, values @ wv.T + bv

        def heads(x: mx.array) -> mx.array:
            batch, length, width = x.shape
            return x.reshape(batch, length, self.heads, width // self.heads).transpose(0, 2, 1, 3)

        q, k, v = heads(q), heads(k), heads(v)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
        batch, _, length, _ = out.shape
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = TorchMultiheadAttention(width, heads)
        self.feedforward_norm = nn.LayerNorm(width)
        # Dropout placeholders keep the torch Sequential indices (0 and 3 are the Linears).
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward), nn.GELU(), nn.Dropout(0.0), nn.Linear(feedforward, width), nn.Dropout(0.0)
        )

    def __call__(self, queries: mx.array, memory: mx.array) -> mx.array:
        memory = self.memory_norm(memory)
        queries = queries + self.attention(self.query_norm(queries), memory, memory)
        return queries + self.feedforward(self.feedforward_norm(queries))


class TransformerDecoderLayer(nn.Module):
    """``torch.nn.TransformerDecoderLayer`` with ``norm_first=True``, GELU, no masks."""

    def __init__(self, width: int, heads: int, feedforward: int) -> None:
        super().__init__()
        self.self_attn = TorchMultiheadAttention(width, heads)
        self.multihead_attn = TorchMultiheadAttention(width, heads)
        self.linear1 = nn.Linear(width, feedforward)
        self.linear2 = nn.Linear(feedforward, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)

    def __call__(self, x: mx.array, memory: mx.array) -> mx.array:
        y = self.norm1(x)
        x = x + self.self_attn(y, y, y)
        x = x + self.multihead_attn(self.norm2(x), memory, memory)
        return x + self.linear2(nn.gelu(self.linear1(self.norm3(x))))


def _normalize(x: mx.array, eps: float = 1e-12) -> mx.array:
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), eps)


def _cosine_similarity(a: mx.array, b: mx.array, eps: float = 1e-8) -> mx.array:
    a = a / mx.maximum(mx.linalg.norm(a, axis=-1, keepdims=True), eps)
    b = b / mx.maximum(mx.linalg.norm(b, axis=-1, keepdims=True), eps)
    return (a * b).sum(axis=-1)


class JointSchemaHead(nn.Module):
    def __init__(
        self, hidden_size: int, width: int, routing_layers: int, layers: int, heads: int, feedforward: int
    ) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(3, width)
        self.evidence_layers = [EvidenceRoutingLayer(width, heads, feedforward) for _ in range(routing_layers)]
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = [TransformerDecoderLayer(width, heads, feedforward) for _ in range(layers)]
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = nn.Sequential(
            nn.Linear(width * 4, width), nn.GELU(), nn.Dropout(0.0), nn.Linear(width, 1)
        )
        self.prior_logit_scale = mx.zeros(())
        self.joint_logit_scale = mx.zeros(())
        self.residual_gate = mx.zeros(())

    @classmethod
    def from_release(cls, path: str | Path, dtype: mx.Dtype = mx.float32) -> "JointSchemaHead":
        path = Path(path)
        head = cls(**json.loads((path / "joint_head_config.json").read_text()))
        weights = mx.load(str(path / "joint_head.safetensors"))
        weights = {
            re.sub(r"(feedforward|residual_scorer)\.(\d+)\.", r"\1.layers.\2.", key): value.astype(dtype)
            for key, value in weights.items()
        }
        head.load_weights(list(weights.items()), strict=True)
        head.eval()
        mx.eval(head.parameters())
        return head

    def __call__(
        self,
        hidden_states: mx.array,
        record: EncodedRecord,
        output_embedding_rows: Callable[[mx.array], mx.array],
    ) -> list[mx.array]:
        """Score one record.

        ``hidden_states`` is the backbone's final (normed) hidden state for the record, shape
        ``[length, hidden_size]``. ``output_embedding_rows(ids)`` returns rows of the LM head
        weight. Returns one logit vector per question.
        """
        hidden = self.hidden_norm(hidden_states.astype(self.hidden_norm.weight.dtype))
        memory = self.memory_projection(hidden)[None]
        global_vector = hidden[-1]

        def mean_span(span: tuple[int, int]) -> mx.array:
            return hidden[span[0] : span[1]].mean(axis=0)

        question_vectors = mx.stack([mean_span(q.question_span) for q in record.questions])
        type_ids = mx.array([q.question_type for q in record.questions])

        input_ids = mx.array(record.input_ids)
        option_counts = [len(q.option_spans) for q in record.questions]
        spans = [span for q in record.questions for span in q.option_spans]
        contexts = mx.stack([mean_span(span) for span in spans])
        lexical = mx.stack(
            [output_embedding_rows(input_ids[s:e]).astype(hidden.dtype).mean(axis=0) for s, e in spans]
        )
        question_per_option = mx.concatenate(
            [mx.broadcast_to(question_vectors[i], (n, question_vectors.shape[-1])) for i, n in enumerate(option_counts)]
        )

        routed = (
            self.option_context_projection(contexts)
            + self.option_lexical_projection(lexical)
            + self.option_question_projection(question_per_option)
        )[None]
        for layer in self.evidence_layers:
            routed = layer(routed, memory)
        routed = routed[0]
        boundaries = list(_cumsum(option_counts)[:-1])
        split_options = mx.split(routed, boundaries, axis=0) if boundaries else [routed]
        split_lexical = mx.split(lexical, boundaries, axis=0) if boundaries else [lexical]

        base_fields = self.question_projection(question_vectors)
        summaries = []
        for field, options in zip(base_fields, split_options):
            weights = mx.softmax((options @ field) / math.sqrt(options.shape[-1]), axis=0)
            summaries.append((weights[:, None] * options).sum(axis=0))
        fields = (
            base_fields
            + self.option_summary_norm(mx.stack(summaries))
            + self.global_projection(global_vector)[None]
            + self.type_embedding(type_ids)
        )[None]
        for layer in self.layers:
            fields = layer(fields, memory)
        fields = self.field_norm(fields[0])

        prior_scale = mx.exp(mx.minimum(self.prior_logit_scale, math.log(100.0)))
        joint_scale = mx.exp(mx.minimum(self.joint_logit_scale, math.log(100.0)))
        gate = mx.sigmoid(self.residual_gate)
        logits = []
        for index, (field, lexical_options, routed_options) in enumerate(zip(fields, split_lexical, split_options)):
            anchor = _normalize(question_vectors[index] + global_vector)
            prior = prior_scale * (_normalize(lexical_options) @ anchor)
            options = self.option_norm(routed_options)
            repeated = mx.broadcast_to(field, options.shape)
            features = mx.concatenate(
                [repeated, options, repeated * options, mx.abs(repeated - options)], axis=-1
            )
            residual = self.residual_scorer(features).squeeze(-1)
            joint = joint_scale * _cosine_similarity(repeated, options) + residual
            logits.append(prior + gate * joint)
        return logits


def _cumsum(values: list[int]) -> list[int]:
    total, out = 0, []
    for value in values:
        total += value
        out.append(total)
    return out
