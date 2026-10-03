"""End-to-end parity: PyTorch reference (transformers) vs clef-mlx on the same records.

Each side runs in its own subprocess so the two copies of the weights never share memory.

    uv run python scripts/compare_backbone.py --model Cloudflare/clef-flash
    uv run python scripts/compare_backbone.py --model Cloudflare/clef-flash --mlx-model ./clef-flash-4bit
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def receipt() -> Image.Image:
    image = Image.new("RGB", (480, 360), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    for row, line in enumerate(["ACME HARDWARE", "Hammer      12.99", "Nails        4.50", "TOTAL      $17.49"]):
        draw.text((30, 30 + row * 70), line, fill="black", font=font)
    return image


def shapes() -> Image.Image:
    image = Image.new("RGB", (517, 389), (40, 70, 200))
    ImageDraw.Draw(image).ellipse([140, 80, 380, 320], fill=(220, 30, 30))
    return image


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
            "outage": {"type": "noul", "instructions": "Is a service down?"},
        },
    },
    {
        "state": {
            "ticket": "Hi, I was charged twice for my March subscription. Can you refund the duplicate?",
            "customer_tier": "pro",
        },
        "questions": {
            "intent": {
                "type": "choice",
                "criteria": {
                    "refund": "Customer wants money back",
                    "cancel": "Customer wants to cancel",
                    "bug": "Customer reports a product defect",
                    "question": "General question",
                },
            },
            "sentiment": {"type": "score", "criteria": ["Angry", "Neutral", "Happy"]},
            "escalate": {"type": "noul", "instructions": "Does this need a human agent?"},
        },
    },
    {
        "state": {"task": "Review the attached receipt."},
        "images": [receipt()],
        "questions": {
            "legible": {"type": "noul", "instructions": "Is the receipt total legible?"},
            "over_15": {"type": "noul", "instructions": "Is the receipt total above 15 USD?"},
            "store": {
                "type": "choice",
                "instructions": "What kind of store issued the receipt?",
                "criteria": {"grocery": "Food store", "hardware": "Tools and supplies", "restaurant": "Meals"},
            },
        },
    },
    {
        "state": "Describe the picture.",
        "images": [shapes()],
        "questions": {
            "circle_color": {
                "type": "choice",
                "instructions": "What color is the circle?",
                "criteria": {"red": "Red", "green": "Green", "blue": "Blue"},
            },
            "triangle": {"type": "noul", "instructions": "Is there a triangle in the picture?"},
        },
    },
    {
        "state": {"ticket": "Customer says the item in the photo arrived broken.", "order": 1182},
        "images": [shapes(), receipt()],
        "questions": {
            "photo_count": {"type": "score", "criteria": ["No images", "One image", "Two images", "Three or more"]},
            "has_receipt": {"type": "noul", "instructions": "Is one of the images a receipt?"},
        },
    },
]


def run_reference(model: str, device: str) -> list[dict]:
    import torch
    from huggingface_hub import snapshot_download

    path = snapshot_download(model)
    spec = importlib.util.spec_from_file_location("joint_schema_model", f"{path}/joint_schema_model.py")
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    clef, processor = reference.load_release_model(path, device=device)
    results = []
    for record in RECORDS:
        encoded = reference.encode_record(processor.tokenizer, record, processor=processor)
        batch = reference.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device(device))
        start = time.perf_counter()
        with torch.inference_mode():
            logits = clef(batch)[0]
        elapsed = time.perf_counter() - start
        results.append(
            {
                "seconds": elapsed,
                "probabilities": {
                    q.question_id: dict(zip(q.option_ids, ql.float().softmax(-1).tolist()))
                    for q, ql in zip(encoded.questions, logits)
                },
            }
        )
    return results


def run_mlx(model: str, vision: bool | None) -> list[dict]:
    from clef_mlx import load

    clef = load(model, vision=vision)
    clef.probabilities(RECORDS[0])  # warm up kernels
    results = []
    for record in RECORDS:
        if record.get("images") and clef.processor is None:
            results.append(None)  # text-only backbone
            continue
        start = time.perf_counter()
        probabilities = clef.probabilities(record)
        results.append({"seconds": time.perf_counter() - start, "probabilities": probabilities})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Cloudflare/clef-flash", help="release used for the reference")
    parser.add_argument("--mlx-model", help="MLX model dir or repo (defaults to --model)")
    parser.add_argument("--device", default="mps", help="torch device for the reference")
    parser.add_argument("--side", choices=["reference", "mlx"], help=argparse.SUPPRESS)
    parser.add_argument("--cache", type=Path, help="reuse/write the reference results here")
    parser.add_argument("--text-only", action="store_true", help="MLX side: text backbone, skip image records")
    args = parser.parse_args()

    if args.side == "reference":
        json.dump(run_reference(args.model, args.device), sys.stdout)
        return
    if args.side == "mlx":
        json.dump(run_mlx(args.mlx_model or args.model, False if args.text_only else None), sys.stdout)
        return

    def side(name: str) -> list[dict]:
        command = [sys.executable, __file__, "--side", name, "--model", args.model, "--device", args.device]
        if args.mlx_model:
            command += ["--mlx-model", args.mlx_model]
        if args.text_only:
            command += ["--text-only"]
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
        return json.loads(output.strip().splitlines()[-1])

    if args.cache and args.cache.exists():
        reference = json.loads(args.cache.read_text())
    else:
        reference = side("reference")
        if args.cache:
            args.cache.write_text(json.dumps(reference))
    ours = side("mlx")

    worst = 0.0
    agree = total = 0
    for index, (want, got) in enumerate(zip(reference, ours)):
        if got is None:
            print(f"record {index}: skipped (image record, text-only backbone)")
            continue
        print(f"record {index}: reference {want['seconds']:.2f}s, mlx {got['seconds']:.2f}s")
        for question, expected in want["probabilities"].items():
            actual = got["probabilities"][question]
            diff = max(abs(expected[o] - actual[o]) for o in expected)
            worst = max(worst, diff)
            total += 1
            agree += max(expected, key=expected.get) == max(actual, key=actual.get)
            fmt = lambda p: " ".join(f"{o}={v:.3f}" for o, v in p.items())  # noqa: E731
            print(f"  {question:<12} ref  {fmt(expected)}")
            print(f"  {'':<12} mlx  {fmt(actual)}   max|Δp|={diff:.4f}")
    print(f"\nargmax agreement {agree}/{total}, worst max|Δp| {worst:.4f}")


if __name__ == "__main__":
    main()
