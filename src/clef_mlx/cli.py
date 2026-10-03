"""``clef-mlx`` command line: convert a release to quantized MLX, or answer a SystemOne request."""

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

    run_parser = commands.add_parser("run", help="answer a /v1/systemone request body (JSON file or stdin)")
    run_parser.add_argument("--model", default="Cloudflare/clef-flash")
    run_parser.add_argument("request", nargs="?", help="request JSON path; reads stdin when omitted")

    args = parser.parse_args(argv)

    from .model import convert, load

    if args.command == "convert":
        out = convert(args.model, args.out, args.bits, args.group_size, args.quantize_lm_head)
        print(f"wrote {out}")
        return

    request = json.load(open(args.request) if args.request else sys.stdin)
    request.setdefault("model", args.model)
    clef = load(args.model)
    start = time.perf_counter()
    response = clef.systemone(request)
    response["usage"]["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
