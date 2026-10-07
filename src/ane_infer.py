#!/usr/bin/env python3
"""Lean Core ML runtime of the all-ANE export (no PyTorch in the process), for resource measurement.

  pack   (needs torch, run once) -> <out>/host/{enc_emb,dec_emb,pos}.npy (fp16) + host.json (dat_ids, byte pieces, ids)
  run    numpy + coremltools + tokenizers + sentencepiece only: translate Manga109 clean boxes with
         encoder_static_fp16 (L32/L64/L128) + decoder_fp16 on CPU_AND_NE; reports wall p50/p90, process CPU time per
         box and peak memory footprint (macOS rusage), and agreement with a given eval_*.jsonl.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.modules.setdefault("tensorflow", None)        # see export_coreml.py: TF import deadlocks next to tokenizers
ROOT = Path(__file__).resolve().parents[1]
NEG = -1e4
BOS, EOS, PAD = 1, 2, 0


def pack(out: Path):
    import torch
    sys.path.insert(0, str(ROOT / "ar_mt"))
    from train import BOS as B_, EOS as E_, PAD as P_
    ck = torch.load(ROOT / "artifacts/r2_20261006/weights/r2_step54000.pt", map_location="cpu", weights_only=False, mmap=True)
    sd = ck["model"]
    d = sd["emb.weight"].shape[1]
    h = out / "host"
    h.mkdir(parents=True, exist_ok=True)
    np.save(h / "enc_emb.npy", sd["encoder.embeddings.tok_embeddings.weight"].float().numpy().astype(np.float16))
    np.save(h / "dec_emb.npy", (sd["emb.weight"].float() * d ** 0.5).numpy().astype(np.float16))
    np.save(h / "pos.npy", sd["pos.weight"].float().numpy().astype(np.float16))
    vocab = json.loads((ROOT / "artifacts/ar_mt_20261005/data_v2_ext/vocab.json").read_text())
    lut = np.zeros(vocab["dat_vocab"], dtype=np.int64)
    for i, x in enumerate(vocab["dat_ids"]):
        if x >= 0:
            lut[x] = i
    (h / "host.json").write_text(json.dumps({"dat_ids": vocab["dat_ids"], "byte_compact": [int(lut[x]) for x in vocab["byte_piece_ids"]],
                                             "bos": B_, "eos": E_, "pad": P_, "heads": 12, "head_dim": 64, "T": 128,
                                             "buckets": [32, 64, 128]}))
    print("packed", h)


class ANE:
    def __init__(self, out: Path):
        import coremltools as ct
        import sentencepiece as spm
        from tokenizers import Tokenizer
        h = out / "host"
        self.cfg = json.loads((h / "host.json").read_text())
        self.enc_emb = np.load(h / "enc_emb.npy", mmap_mode="r")                    # rows are read on demand
        self.dec_emb = np.load(h / "dec_emb.npy").astype(np.float32)
        self.pos = np.load(h / "pos.npy").astype(np.float32)
        self.tok = Tokenizer.from_file(str(ROOT / "artifacts/ar_mt_20261005/weights/dapt_v1_model/tokenizer.json"))
        self.sp = spm.SentencePieceProcessor(model_file=str(ROOT / "artifacts/preview200m_training_20260923/tokenizer/joint.model"))
        cu = ct.ComputeUnit.CPU_AND_NE
        self.buckets = self.cfg["buckets"]
        self.enc = {b: ct.models.MLModel(str(out / "encoder_static_fp16.mlpackage"), function_name=f"L{b}", compute_units=cu)
                    for b in self.buckets}
        self.dec = ct.models.MLModel(str(out / "decoder_fp16.mlpackage"), compute_units=cu)
        B = self.cfg["byte_compact"]
        V = self.dec_emb.shape[0]

        def mask(vals):
            m = np.zeros(V, bool)
            m[[B[v] for v in vals]] = True
            return m
        self.rules = [(None, B[0xE3], mask([0x81, 0x82, 0x83])), (B[0xE3], B[0x87], mask(range(0xB0, 0xC0))),
                      (B[0xEF], B[0xBD], mask(range(0xA6, 0xC0))), (B[0xEF], B[0xBE], mask(range(0x80, 0x9E)))]
        self.H, self.hd, self.T = self.cfg["heads"], self.cfg["head_dim"], self.cfg["T"]
        self.M = 2 + max(self.buckets)

    def __call__(self, text: str) -> str:
        c = self.cfg
        ids = self.tok.encode(text).ids[:256]
        n = len(ids)
        Lb = next((b for b in self.buckets if b >= n), self.buckets[-1])
        ids, n = ids[:Lb], min(n, Lb)
        arr = np.full(Lb, 3, dtype=np.int64)
        arr[:n] = ids
        x = np.asarray(self.enc_emb[arr], dtype=np.float32)[None]
        km = np.full((1, 1, 1, Lb), NEG, np.float32)
        km[..., :n] = 0.0
        r = self.enc[Lb].predict({"x": x, "kmask": km})
        cr = {}
        for k in ("ck0", "cv0", "ck1", "cv1"):
            z = np.zeros((1, self.H, self.M, self.hd), np.float32)
            z[:, :, :n + 2] = r[k][:, :, :n + 2]
            cr[k] = z
        cmask = np.full((1, 1, 1, self.M), NEG, np.float32)
        cmask[..., :n + 2] = 0.0
        caches = {k: np.zeros((1, self.H, self.T, self.hd), np.float32) for k in ("kc0", "vc0", "kc1", "vc1")}
        smask = np.full((1, 1, 1, self.T), NEG, np.float32)
        prev1 = prev2 = -1
        tok, out = c["bos"], []
        for t in range(min(3 * n + 10, 512, self.T + 1)):
            xin = (self.dec_emb[tok] + self.pos[t])[None, None]
            o = self.dec.predict({"x": xin, **caches, "smask": smask, **cr, "cmask": cmask})
            lg = o["logits"].reshape(-1).astype(np.float32)
            for p2, p1, mk in self.rules:
                if prev1 == p1 and (p2 is None or prev2 == p2):
                    lg[mk] = -np.inf
            nxt = int(lg.argmax())
            if nxt in (c["eos"], c["pad"]):
                break
            out.append(nxt)
            if t < self.T:
                for k, nk in (("kc0", "k0"), ("vc0", "v0"), ("kc1", "k1"), ("vc1", "v1")):
                    caches[k][:, :, t] = o[nk][:, :, 0]
                smask[..., t] = 0.0
            prev2, prev1, tok = prev1, nxt, nxt
        dat = c["dat_ids"]
        return self.sp.decode([int(dat[i]) for i in out if dat[i] >= 4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pack", "run"))
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/coreml_20261007")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--compare", type=Path, help="eval_*.jsonl with ref_armt to check agreement")
    a = ap.parse_args()
    if a.mode == "pack":
        return pack(a.out)
    sys.path.insert(0, str(ROOT / "benchmarks/sakura_cmp"))
    from common import load_m109
    recs = load_m109()[:a.n]
    t0 = time.perf_counter()
    m = ANE(a.out)
    load_s = time.perf_counter() - t0
    for r in recs[:10]:
        m(r["clean"])
    ru0, ms, hyps = resource.getrusage(resource.RUSAGE_SELF), [], []
    for r in recs:
        t = time.perf_counter()
        hyps.append(m(r["clean"]))
        ms.append(1000 * (time.perf_counter() - t))
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    rep = {"n": len(recs), "load_s": round(load_s, 1), "ms_p50": round(float(np.median(ms)), 1),
           "ms_p90": round(float(np.percentile(ms, 90)), 1), "cpu_ms_per_box": round(1000 * cpu / len(recs), 1),
           "cpu_util_of_one_core": round(cpu / (sum(ms) / 1000), 2), "max_rss_mib": round(ru1.ru_maxrss / 2**20, 1)}
    if a.compare:
        ref = {json.loads(x)["id"]: json.loads(x)["ref_armt"] for x in open(a.compare, encoding="utf-8")}
        rep["identical_to_armt"] = round(float(np.mean([h == ref[r["id"]] for h, r in zip(hyps, recs) if r["id"] in ref])), 4)
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
