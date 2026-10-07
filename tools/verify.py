#!/usr/bin/env python3
"""Run one backend over a jsonl of bubbles ({"id", "clean"} per line) and compare with a reference jsonl
({"id", "hyp"}): share of identical outputs, p50 / p90 milliseconds per bubble after --warm bubbles.

  python tools/verify.py --model DIR --backend onnx --data boxes.jsonl --ref ref.jsonl [--n 500] [--out hyps.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jmt_fast import BACKENDS, load  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", required=True, choices=BACKENDS)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--device")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(a.data, encoding="utf-8")][:a.n]
    t0 = time.perf_counter()
    tr = load(a.model, a.backend, **({"device": a.device} if a.device else {}))
    load_s = time.perf_counter() - t0
    for r in rows[:a.warm]:
        tr.translate(r["clean"])
    hyps, ms = [], []
    for r in rows:
        t = time.perf_counter()
        hyps.append(tr.translate(r["clean"]))
        ms.append(1000 * (time.perf_counter() - t))
    rep = {"backend": a.backend, "n": len(rows), "load_s": round(load_s, 1), "p50_ms": round(float(np.percentile(ms, 50)), 1),
           "p90_ms": round(float(np.percentile(ms, 90)), 1)}
    if a.ref:
        ref = {json.loads(x)["id"]: json.loads(x)["hyp"] for x in open(a.ref, encoding="utf-8")}
        same = [h == ref[r["id"]] for h, r in zip(hyps, rows) if r["id"] in ref]
        rep.update(compared=len(same), identical=int(sum(same)), identical_share=round(float(np.mean(same)), 4))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r, h, t in zip(rows, hyps, ms):
                f.write(json.dumps({"id": r["id"], "hyp": h, "ms": round(t, 2)}, ensure_ascii=False) + "\n")
    print(json.dumps(rep, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
