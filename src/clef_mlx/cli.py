"""``clef-mlx`` command line: convert a release to quantized MLX, or score one record's options."""

from __future__ import annotations

import argparse
import json
import sys
import time


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="clef-mlx")
    commands = parser.add_subparsers(dest="command", required=True)

    convert_parser = commands.add_parser("convert", help="quantize a Clef release for MLX")
    convert_parser.add_argument("--model", default="Cloudflare/clef")
    convert_parser.add_argument("--out", required=True)
    convert_parser.add_argument("--bits", type=int, default=8)
    convert_parser.add_argument("--group-size", type=int, default=64)
    convert_parser.add_argument("--quantize-lm-head", action="store_true")
    convert_parser.add_argument("--text-only", action="store_true", help="drop the vision tower (uses mlx-lm)")

    run_parser = commands.add_parser(
        "run", help="print a record's option probabilities (JSON file or stdin); images are file paths"
    )
    run_parser.add_argument("--model", default="Cloudflare/clef-flash")
    run_parser.add_argument("record", nargs="?", help="record JSON path; reads stdin when omitted")

    args = parser.parse_args(argv)

    from .model import convert, load

    if args.command == "convert":
        vision = False if args.text_only else None
        out = convert(args.model, args.out, args.bits, args.group_size, args.quantize_lm_head, vision=vision)
        print(f"wrote {out}")
        return

    record = json.load(open(args.record) if args.record else sys.stdin)
    if record.get("images"):
        from PIL import Image  # JSON carries image file paths; the model takes PIL images

        record["images"] = [Image.open(image).convert("RGB") for image in record["images"]]
    clef = load(args.model)
    start = time.perf_counter()
    encoded = clef.encode(record)
    probabilities = clef.option_probabilities(encoded)
    usage = {"input_tokens": len(encoded.input_ids), "latency_ms": round((time.perf_counter() - start) * 1000, 1)}
    print(json.dumps({"probabilities": probabilities, "usage": usage}, indent=2))


if __name__ == "__main__":
    main()
