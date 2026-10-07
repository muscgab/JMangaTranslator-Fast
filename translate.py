#!/usr/bin/env python3
"""Translate Japanese manga speech bubbles to Simplified Chinese, one bubble per line.

  python translate.py --model DIR "堪忍袋の緒が切れた！"
  python translate.py --model DIR --backend onnx < bubbles.txt > out.txt

Backends: auto (default), torch, cuda-graphs, onnx, coreml, mlx. Each input line is one bubble; line breaks inside a
bubble should be removed before translation. Empty lines give empty output lines.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jmt_fast import BACKENDS, load, pick  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="bubbles to translate (default: read lines from stdin)")
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent), help="release directory")
    ap.add_argument("--backend", default="auto", choices=("auto", *BACKENDS))
    ap.add_argument("--device", help="torch backend: cpu / cuda / mps (default: best available)")
    ap.add_argument("--threads", type=int, help="onnx backend: CPU threads (default: min(6, cores))")
    ap.add_argument("--timing", action="store_true", help="print per-bubble milliseconds to stderr")
    args = ap.parse_args()
    backend = pick() if args.backend == "auto" else args.backend
    kw = {}
    if backend == "torch" and args.device:
        kw["device"] = args.device
    if backend == "onnx" and args.threads:
        kw["threads"] = args.threads
    t0 = time.perf_counter()
    tr = load(args.model, backend, **kw)
    print(f"[{backend}] loaded in {time.perf_counter() - t0:.1f} s", file=sys.stderr)
    lines = args.text or (x.rstrip("\n") for x in sys.stdin)
    for line in lines:
        t = time.perf_counter()
        out = tr.translate(line) if line.strip() else ""
        print(out, flush=True)
        if args.timing:
            print(f"{1000 * (time.perf_counter() - t):.1f} ms", file=sys.stderr)


if __name__ == "__main__":
    main()
