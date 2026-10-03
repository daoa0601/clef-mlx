# clef-mlx

Run Cloudflare's [Clef](https://huggingface.co/Cloudflare/clef) and
[Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) decision models on Apple silicon with
[MLX](https://github.com/ml-explore/mlx).

Clef is a Qwen3.5-architecture backbone plus a small "joint schema head" that scores every allowed
option of every question in one forward pass. Here the head and record encoding are ported to MLX,
and the backbone runs on one of two upstream implementations:

- **text**: [mlx-lm](https://github.com/ml-explore/mlx-lm)'s `qwen3_5` (the vision tower is dropped)
- **vision**: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)'s `qwen3_5` with its vision tower, so
  records can carry images. Optional, via the `vision` extra.

No torch at runtime in either mode. Video inputs are not ported yet and raise `NotImplementedError`.

## Install

```bash
uv sync                      # text only
uv sync --extra vision       # + images (pulls in mlx-vlm)
```

## Use

```python
from PIL import Image
from clef_mlx import load

clef = load("Cloudflare/clef-flash")   # HF repo id, release dir, or a converted dir
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

print(clef.probabilities({
    "state": {"task": "Review the attached receipt."},
    "images": [Image.open("receipt.jpg")],
    "questions": {"legible": {"type": "noul", "instructions": "Is the receipt total legible?"}},
}))
```

`load` picks the vision backbone automatically when the checkpoint has vision weights and the
`vision` extra is installed; pass `vision=False` to force the lighter text backbone.
`clef.probabilities(record)` returns `{question_id: {option_id: p}}` for the release's record format.

CLI (in request JSON, `images` are file paths):

```bash
echo '{"state": "Expense check", "images": ["receipt.png"],
       "questions": {"over_15": {"type": "noul", "instructions": "Is the total above 15 USD?"}}}' \
  | uv run clef-mlx run --model Cloudflare/clef-flash
```

## Memory and quantization

| Model | bf16 | 8-bit + vision | 8-bit text-only | 4-bit text-only |
|---|---|---|---|---|
| Clef-Flash (9B) | ~19 GB | 11 GB | 10 GB | 6.3 GB |
| Clef (27B) | ~54 GB | ~30 GB (est.) | ~29 GB (est.) | ~17 GB (est.) |

Clef in bf16 does not fit on a 48 GB Mac, so quantize it first. The conversion downloads the full
bf16 release first (about 54 GB on disk):

```bash
uv run clef-mlx convert --model Cloudflare/clef --out ./clef-8bit               # 8-bit, keeps vision
uv run clef-mlx convert --model Cloudflare/clef --out ./clef-8bit --text-only   # mlx-lm, no vision
uv run clef-mlx run --model ./clef-8bit request.json
```

`convert` quantizes the language model, keeps the vision tower and the LM head in bf16 (the joint
head reads LM head rows directly as option features), and copies the joint head and the release's
processor config next to the backbone. It goes through mlx-vlm when the `vision` extra is installed
and through mlx-lm with `--text-only`. 8-bit is the default because on Clef-Flash it is
indistinguishable from bf16 (below); `--bits 4` works but moves borderline probabilities by a few
points.

## Parity with the PyTorch release

```bash
uv sync --all-extras --group dev
uv run pytest                     # encoding (text + images) and head vs the release code
uv run python scripts/compare_backbone.py --model Cloudflare/clef-flash
uv run python scripts/compare_backbone.py --model Cloudflare/clef-flash --mlx-model ./clef-flash-8bit
```

The script runs the release's own `joint_schema_model.py` (transformers + torch on MPS, bf16) and
clef-mlx on the same records — 3 text records and 3 image records (a rendered receipt, a shapes
picture, and a two-image record) — then reports per-option probability differences. Image records
are skipped for text-only checkpoints.

Measured on Clef-Flash (M4 Pro, 48 GB):

| clef-mlx variant | questions | argmax agreement | worst \|Δp\| | warm latency / record |
|---|---|---|---|---|
| bf16, vision | 15 (8 text, 7 image) | 15/15 | 0.019 | 0.7 s text, 1.3–2.6 s image |
| 8-bit, vision | 15 | 15/15 | 0.017 | similar |
| bf16, text-only | 8 | 8/8 | 0.006 | ~0.8 s |
| 8-bit, text-only | 8 | 8/8 | 0.004 | ~0.7 s |
| 4-bit, text-only | 8 | 8/8 | 0.092 | ~0.7 s |

Image preprocessing uses mlx-vlm's numpy port of the Qwen3-VL processor. It produces the same
tokens as the HF processor; after resizing, a handful of pixels (~0.001%) differ by one 8-bit
level because PIL and torchvision round differently.

Clef (27B) uses the same architecture and code path but has not been run through this check yet.

## Attribution

`src/clef_mlx/encoding.py` and `src/clef_mlx/head.py` are ports of `joint_schema_model.py` from the
[Cloudflare/clef](https://huggingface.co/Cloudflare/clef) release, which is licensed Apache-2.0.
