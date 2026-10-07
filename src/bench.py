#!/usr/bin/env python3
"""Per-box (batch=1) latency on a modern NVIDIA GPU (user 2026-10-07): 500 Manga109-s boxes (clean input), drawn with
seed 20261007 from the 3,817-box comparison set (make_sample.py), the first --warm boxes translated once before timing.

  ours_eager  Zakkuri v1 as on the Titan Xp run: PyTorch eager fp32, ARMT.generate_ctx (run_ours.Ours)
  ours_graph  Zakkuri v1, fp16 with the per-norm scales of the Core ML export (norm_scales.json; fp16 overflows
              in the encoder's massive activations otherwise), encoder = EncoderStatic per length bucket (32/64/128)
              and one decoder step (DecoderExport, KV cache T=128, cross length 130) each captured as a CUDA graph;
              kana byte rules, argmax and the cache update run inside the step graph; the host replays one step per
              token and reads the token back (one sync per step). Same function as the Core ML / ANE path.
  ours_graph32  the same graphs in fp32 (the norm scales are powers of two, so in fp32 they change nothing). Tesla P4
              (Pascal, 10-07): 32.0 ms p50 vs 30.0 ms for ours_graph -- cuBLAS half GEMMs accumulate in fp32, so the
              fp16 path still works there
  nano        NanoSakura-2.2-0.2B, fp32 (run_ours.Nano)
  gal         GalTransl-v4-4B-2601 Q6_K via llama-server, one request at a time (run_galtransl.Client)

ms = wall time of one box: tokenization -> generation -> CUDA sync -> detokenization (same definition as run_ours.timed).
Output: <out>/<system>.jsonl (id, hyp, ms, ...) and <out>/<system>.summary.json."""
from __future__ import annotations

import argparse
import json
import os
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/sakura_cmp"))
sys.path.insert(0, str(ROOT / "ar_mt"))
from train import BOS, EOS, PAD, decode_ids  # noqa: E402

CKPT = ROOT / "artifacts/r2_20261006/weights/r2_step54000.pt"
ENC = ROOT / "artifacts/ar_mt_20261005/weights/dapt_v1_model"
DATA = ROOT / "artifacts/ar_mt_20261005/data_v2_ext"
SCALES = ROOT / "artifacts/coreml_20261007/norm_scales.json"
NANO = ROOT / "models/nanosakura_2_2_188m"


class OursGraph:
    def __init__(self, buckets=(32, 64, 128), T: int = 128, dev: str = "cuda", dt=torch.float16, graphs: bool = True):
        from run_ours import Ours
        import export_coreml as ex
        o = Ours(CKPT, str(ENC), DATA, "cpu")
        m = o.model.float().eval()
        saved = json.loads(SCALES.read_text())
        for n, mod in m.named_modules():
            if n in saved:
                mod._s = saved[n]["s"]
        self.tok, self.sp, self.dat_ids, self.step = o.tok, o.sp, o.dat_ids, o.step
        self.buckets, self.T, self.M = buckets, T, 2 + max(buckets)
        self.dt = dt
        self.encs = {L: ex.EncoderStatic(m, L).to(dev, dt).eval() for L in buckets}
        self.dec = ex.DecoderExport(m, 1.0).to(dev, dt).eval()
        self.tok_emb = m.encoder.embeddings.tok_embeddings.to(dev, dt)
        self.emb_s = (m.emb.weight.detach() * math.sqrt(m.d)).to(dev, dt)
        self.pos = m.pos.weight.detach().to(dev, dt)
        H, hd = m.layers[0].h, m.layers[0].hd
        self.rules = [(p2, p1, mk) for p2, p1, mk in m.kana_byte_rules(o.byte_compact, self.emb_s.shape[0], dev)]
        z = lambda *s: torch.zeros(*s, device=dev, dtype=dt)  # noqa: E731
        self.cross = [z(1, H, self.M, hd) for _ in range(4)]
        self.cmask = torch.full((1, 1, 1, self.M), ex.NEG, device=dev, dtype=dt)
        self.caches = [z(1, H, T, hd) for _ in range(4)]
        self.smask = torch.full((1, 1, 1, T), ex.NEG, device=dev, dtype=dt)
        self.ids = {L: torch.full((1, L), 3, dtype=torch.long, device=dev) for L in buckets}
        self.kmask = {L: torch.zeros((1, 1, 1, L), device=dev, dtype=dt) for L in buckets}
        lt = lambda v: torch.full((1,), v, dtype=torch.long, device=dev)  # noqa: E731
        self.t, self.cur, self.prev1, self.prev2, self.nxt = lt(0), lt(BOS), lt(-1), lt(-1), lt(0)
        self.NEG = ex.NEG
        t0 = time.perf_counter()
        if graphs:
            self.g_enc = {L: self._capture(lambda L=L: self._enc(L)) for L in buckets}
            self._reset()
            self.g_dec = self._capture(self._dec_step)
        else:                                            # debug path (CPU / no capture): same functions, eager
            self.g_enc = {L: type("G", (), {"replay": staticmethod(lambda L=L: self._enc(L))})() for L in buckets}
            self.g_dec = type("G", (), {"replay": staticmethod(self._dec_step)})()
        self.capture_s = round(time.perf_counter() - t0, 1)

    @torch.no_grad()
    def _enc(self, L):
        outs = self.encs[L](self.tok_emb(self.ids[L]), self.kmask[L])
        for buf, x in zip(self.cross, outs):
            buf[:, :, :2 + L].copy_(x)

    @torch.no_grad()
    def _dec_step(self):
        x = (self.emb_s[self.cur] + self.pos[self.t])[None]                                  # [1, 1, d]
        logits, *news = self.dec(x, *self.caches, self.smask, *self.cross, self.cmask)
        lg = logits.view(-1).float()
        for p2, p1, mk in self.rules:
            hit = self.prev1.eq(p1) if p2 is None else self.prev1.eq(p1) & self.prev2.eq(p2)
            lg = lg.masked_fill(hit & mk, float("-inf"))
        nxt = lg.argmax().view(1)
        for c, nw in zip(self.caches, news):
            c.index_copy_(2, self.t, nw)
        self.smask.index_fill_(3, self.t, 0.0)
        self.prev2.copy_(self.prev1)
        self.prev1.copy_(nxt)
        self.cur.copy_(nxt)
        self.nxt.copy_(nxt)
        self.t.add_(1)

    def _capture(self, fn):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._reset()
                fn()
        torch.cuda.current_stream().wait_stream(s)
        self._reset()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g

    def _reset(self):
        self.smask.fill_(self.NEG)
        self.t.zero_()
        self.cur.fill_(BOS)
        self.prev1.fill_(-1)
        self.prev2.fill_(-1)

    @torch.no_grad()
    def translate(self, texts):
        text, = texts
        ids = self.tok(text, add_special_tokens=True, truncation=True, max_length=256)["input_ids"]
        n = len(ids)
        L = next((b for b in self.buckets if b >= n), self.buckets[-1])
        ids, n = ids[:L], min(n, L)
        a = torch.full((1, L), 3, dtype=torch.long)
        a[0, :n] = torch.tensor(ids)
        km = torch.zeros((1, 1, 1, L), dtype=self.dt)
        km[..., n:] = self.NEG
        cm = torch.full((1, 1, 1, self.M), self.NEG, dtype=self.dt)
        cm[..., :2 + n] = 0.0
        self.ids[L].copy_(a, non_blocking=True)
        self.kmask[L].copy_(km, non_blocking=True)
        self.cmask.copy_(cm, non_blocking=True)
        self.g_enc[L].replay()
        self._reset()
        out = []
        for _ in range(min(3 * n + 10, self.T)):
            self.g_dec.replay()
            v = int(self.nxt.item())
            if v in (EOS, PAD):
                break
            out.append(v)
        self.last = {"ntok": n, "nout": len(out)}
        return [decode_ids(out, self.dat_ids, self.sp)]


def load(system: str):
    if system == "ours_eager":
        from run_ours import Ours
        return Ours(CKPT, str(ENC), DATA, "cuda")
    if system == "ours_graph":
        return OursGraph()
    if system == "ours_graph32":
        return OursGraph(dt=torch.float32)
    if system == "nano":
        from run_ours import Nano
        return Nano(NANO, "cuda")
    from run_galtransl import Client
    return Client(os.environ.get("GAL_URL", "http://127.0.0.1:18080"), 20261006)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=("ours_eager", "ours_graph", "ours_graph32", "nano", "gal"))
    ap.add_argument("--sample", type=Path, default=Path(__file__).with_name("sample500.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in open(args.sample, encoding="utf-8")][:args.limit or None]
    t0 = time.perf_counter()
    m = load(args.system)
    load_s = round(time.perf_counter() - t0, 1)
    gal = args.system == "gal"

    def one(text):
        if gal:
            r = m(text)
            return r["hyp"], r["ms"], {k: r.get(k) for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms")}
        torch.cuda.synchronize()
        t = time.perf_counter()
        h = m.translate([text])[0]
        torch.cuda.synchronize()
        return h, round(1000 * (time.perf_counter() - t), 2), dict(getattr(m, "last", {}))

    for r in rows[:args.warm]:
        one(r["clean"])
    res = []
    with open(args.out / f"{args.system}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            h, ms, extra = one(r["clean"])
            rec = {"id": r["id"], "hyp": h, "ms": ms, **extra}
            res.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    a = np.array([x["ms"] for x in res])
    cpu = subprocess.run("lscpu | grep 'Model name' | head -1", shell=True, capture_output=True, text=True).stdout
    summ = {"system": args.system, "n": len(a), "warm": args.warm, "p50_ms": round(float(np.percentile(a, 50)), 1),
            "p90_ms": round(float(np.percentile(a, 90)), 1), "mean_ms": round(float(a.mean()), 1),
            "total_s": round(float(a.sum()) / 1000, 1), "load_s": load_s,
            "capture_s": getattr(m, "capture_s", None),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "cpu": cpu.split(":", 1)[-1].strip(), "torch": torch.__version__, "python": platform.python_version(),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if not gal else None}
    (args.out / f"{args.system}.summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
