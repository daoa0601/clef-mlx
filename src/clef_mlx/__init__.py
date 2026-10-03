"""Run Cloudflare's Clef / Clef-Flash decision models on Apple silicon with MLX."""

from .encoding import EncodedQuestion, EncodedRecord, encode_record
from .model import Clef, convert, load

__all__ = ["Clef", "EncodedQuestion", "EncodedRecord", "convert", "encode_record", "load"]
