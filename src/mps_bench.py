#!/usr/bin/env python3
"""Per-box (batch=1) latency of Zakkuri v1 on Apple-silicon PyTorch MPS (user 2026-10-07: "你试试MPS加速").
Boxes: the first --n of sample500.jsonl (Manga109-s clean input), the first --warm translated once before timing.

  eager_fp32   run_ours.Ours on mps (ARMT.generate_ctx: the self-attention cache grows by one position per step,
               so every step has a new shape)
  static_fp32  bench.OursGraph(dev="mps", graphs=False): fixed-shape encoder per bucket (32/64/128) + one fixed-shape
               decoder step (KV cache T=128, cross 130), cache update / kana rules / argmax on the device
  static_fp16  the same in fp16 with the Core ML per-norm scales
  compiled_fp16  static_fp16 with torch.compile on the encoder / decoder modules
ms = wall time per box incl. tokenization, torch.mps.synchronize() before and after. Agreement with eager_fp32."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--modes", default="eager_fp32,static_fp32,static_fp16")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--warm", type=int, default=20)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(x) for x in open(Path(__file__).with_name("sample500.jsonl"), encoding="utf-8")][:args.n]


def build(mode):
    if mode == "eager_fp32":
        from run_ours import Ours
        o = Ours(bench.CKPT, str(bench.ENC), bench.DATA, "mps")
        o.model.float().eval()
        return o
    dt = torch.float32 if mode == "static_fp32" else torch.float16
    g = bench.OursGraph(dev="mps", dt=dt, graphs=False)
    if mode.startswith("compiled"):
        g.encs = {L: torch.compile(e) for L, e in g.encs.items()}
        g.dec = torch.compile(g.dec)
    return g


ref = None
for mode in args.modes.split(","):
    t0 = time.perf_counter()
    m = build(mode)
    load_s = time.perf_counter() - t0
    for r in rows[:args.warm]:
        m.translate([r["clean"]])
    torch.mps.synchronize()
    hyps, ms = [], []
    for r in rows:
        torch.mps.synchronize()
        t = time.perf_counter()
        hyps.append(m.translate([r["clean"]])[0])
        torch.mps.synchronize()
        ms.append(1000 * (time.perf_counter() - t))
    if mode == "eager_fp32":
        ref = hyps
    a = np.array(ms)
    rep = {"mode": mode, "n": len(a), "p50_ms": round(float(np.percentile(a, 50)), 1),
           "p90_ms": round(float(np.percentile(a, 90)), 1), "mean_ms": round(float(a.mean()), 1),
           "load_s": round(load_s, 1), "mps_gib": round(torch.mps.driver_allocated_memory() / 2**30, 2),
           "identical_to_eager": None if ref is None else round(float(np.mean([x == y for x, y in zip(hyps, ref)])), 4)}
    with open(args.out / f"{mode}.jsonl", "w", encoding="utf-8") as f:
        for r, h, t in zip(rows, hyps, ms):
            f.write(json.dumps({"id": r["id"], "hyp": h, "ms": round(t, 2)}, ensure_ascii=False) + "\n")
    print(json.dumps(rep), flush=True)
    del m
    torch.mps.empty_cache()
