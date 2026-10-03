# clef-mlx

Run Cloudflare's [Clef](https://huggingface.co/Cloudflare/clef) and
[Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) decision models on Apple silicon with
[MLX](https://github.com/ml-explore/mlx).

Clef is a Qwen3.5-architecture backbone plus a small "joint schema head" that scores every allowed
option of every question in one forward pass. Here the backbone runs on
[mlx-lm](https://github.com/ml-explore/mlx-lm)'s `qwen3_5` implementation, and the head and record
encoding are ported to MLX with no torch dependency at runtime.

**Text-only.** The release also accepts images and video. Those paths are not ported, so records
with `images` or `videos` raise `NotImplementedError`.

## Install

```bash
uv sync
```

## Use

```python
from clef_mlx import load

clef = load("Cloudflare/clef-flash")          # HF repo id, release dir, or a converted dir
print(clef.systemone({
    "model": "clef-flash",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle the message?",
                       "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
}))
```

`clef.probabilities(record)` returns `{question_id: {option_id: p}}` for the release's record format.

CLI:

```bash
echo '{"state": "Refund my duplicate charge", "questions": {"refund": {"type": "noul"}}}' \
  | uv run clef-mlx run --model Cloudflare/clef-flash
```

## Memory and quantization

| Model | bf16 | 8-bit | 4-bit |
|---|---|---|---|
| Clef-Flash (9B) | ~19 GB | 10 GB (measured) | 6.3 GB (measured) |
| Clef (27B) | ~54 GB | ~29 GB (est.) | ~17 GB (est.) |

Clef in bf16 does not fit on a 48 GB Mac, so quantize it first. The conversion downloads the full
bf16 release first (about 54 GB on disk):

```bash
uv run clef-mlx convert --model Cloudflare/clef --out ./clef-8bit          # default: 8-bit
uv run clef-mlx run --model ./clef-8bit request.json
```

`convert` quantizes the backbone through `mlx_lm.convert`, keeps the LM head in bf16 because the
joint head reads its rows directly as option features, and copies the joint head next to the
backbone. 8-bit is the default because on Clef-Flash it is indistinguishable from bf16 (below);
`--bits 4` works but moves borderline probabilities by a few points.

## Parity with the PyTorch release

```bash
uv sync --group dev
uv run pytest                                  # encoding + head vs reference (fp32, 1e-4)
uv run python scripts/compare_backbone.py --model Cloudflare/clef-flash
uv run python scripts/compare_backbone.py --model Cloudflare/clef-flash --mlx-model ./clef-flash-4bit
```

The script runs the release's own `joint_schema_model.py` (transformers + torch on MPS) and
clef-mlx on the same records, then reports per-option probability differences.

Measured on Clef-Flash (M4 Pro, 48 GB), 8 questions over 3 records, reference = release code in
bf16 on MPS:

| clef-mlx variant | argmax agreement | worst \|Δp\| | warm latency / record |
|---|---|---|---|
| bf16 | 8/8 | 0.006 | ~0.8 s |
| 8-bit | 8/8 | 0.004 | ~0.7 s |
| 4-bit | 8/8 | 0.092 | ~0.7 s |

Clef (27B) uses the same architecture and code path but has not been run through this check yet.

## Attribution

`src/clef_mlx/encoding.py` and `src/clef_mlx/head.py` are ports of `joint_schema_model.py` from the
[Cloudflare/clef](https://huggingface.co/Cloudflare/clef) release, which is licensed Apache-2.0.
