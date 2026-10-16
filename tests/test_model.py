"""``Clef`` turns the head's logits into per-option probabilities; no weights needed."""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from clef_mlx import Clef, EncodedQuestion, EncodedRecord


def question(question_id: str, option_ids: tuple[str, ...]) -> EncodedQuestion:
    return EncodedQuestion(question_id, 0, (0, 1), tuple((0, 1) for _ in option_ids), option_ids)


ENCODED = EncodedRecord(
    input_ids=(1, 2, 3),
    questions=(question("outage", ("true", "false")), question("team", ("billing", "technical", "sales"))),
    record_id="record",
)


class FixedHead:
    def __call__(self, hidden, encoded, rows):
        assert encoded is ENCODED
        return [mx.array([math.log(3.0), 0.0]), mx.array([0.0, 0.0, math.log(2.0)])]


def test_option_probabilities_name_every_option_of_every_question():
    clef = Clef(backbone=lambda encoded: mx.zeros((3, 4)), head=FixedHead(), tokenizer=None)
    probabilities = clef.option_probabilities(ENCODED)
    assert list(probabilities) == ["outage", "team"]
    assert probabilities["outage"] == pytest.approx({"true": 0.75, "false": 0.25})
    assert probabilities["team"] == pytest.approx({"billing": 0.25, "technical": 0.25, "sales": 0.5})


def test_probabilities_encode_the_record_first(monkeypatch):
    clef = Clef(backbone=lambda encoded: mx.zeros((3, 4)), head=FixedHead(), tokenizer=None)
    records = []
    monkeypatch.setattr(clef, "encode", lambda record, **kwargs: records.append((record, kwargs)) or ENCODED)
    assert clef.probabilities({"state": "s"}, max_length=8)["outage"] == pytest.approx({"true": 0.75, "false": 0.25})
    assert records == [({"state": "s"}, {"max_length": 8})]
