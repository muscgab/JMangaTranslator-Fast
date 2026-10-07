#!/usr/bin/env python3
"""Zakkuri v1 in MLX (Apple-silicon GPU), batch=1 (user 2026-10-07: "你继续尝试" after PyTorch MPS topped out near 40 ms).

Same function as export_coreml.EncoderStatic / DecoderExport (validated against ARMT on Core ML / CUDA):
  encoder  ModernBERT-ja (25 layers; global attention every 3rd layer with RoPE theta 160000, the others local with
           |i-j| <= 64 and theta 10000), exact GELU, LayerNorm without bias; depth fusion of h0..h24, SwiGLU bridge,
           2 null tokens, kv2 per decoder layer. One mx.compile trace per length bucket (32 / 64 / 128).
  decoder  one step with a fixed self-attention cache (T = 128, written at position t with a where-mask), the cross
           K/V of the bucket, kana byte rules, argmax; mx.compile'd. Steps are chained lazily --chunk at a time and
           evaluated once per chunk (tokens after <eos> are discarded). mx.compile is off by default (slower here).
Weights fp16 (--dtype), every norm computed in fp32 (the encoder carries activations up to ~2.9e3, x^2 overflows fp16).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ar_mt"))
sys.path.insert(0, str(ROOT / "benchmarks/sakura_cmp"))
CKPT = ROOT / "artifacts/r2_20261006/weights/r2_step54000.pt"
ENC = ROOT / "artifacts/ar_mt_20261005/weights/dapt_v1_model"
DATA = ROOT / "artifacts/ar_mt_20261005/data_v2_ext"
NEG = -1e4
BOS, EOS, PAD = 1, 2, 3


class ZakkuriMLX:
    def __init__(self, dtype=mx.float16, buckets=(32, 64, 128), T: int = 128, chunk: int = 4, compile_: bool = False, exact: bool = False):
        import torch
        import sentencepiece as spm
        from transformers import AutoConfig, AutoTokenizer
        from model import ARMT
        from train import DAT_TOK, decode_ids
        self.decode_ids = decode_ids
        ck = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
        sd = ck["model"]
        self.dt, self.buckets, self.T, self.chunk, self.exact = dtype, buckets, T, chunk, exact
        W = lambda k, dt=dtype: mx.array(sd[k].float().numpy()).astype(dt)  # noqa: E731
        c = AutoConfig.from_pretrained(str(ENC))
        self.nl, self.H, self.hd, self.win, self.eps = c.num_hidden_layers, c.num_attention_heads, c.hidden_size // c.num_attention_heads, c.sliding_window, c.norm_eps
        self.theta = {True: c.rope_parameters["full_attention"]["rope_theta"], False: c.rope_parameters["sliding_attention"]["rope_theta"]}
        self.tok_emb = W("encoder.embeddings.tok_embeddings.weight")
        self.emb_norm = W("encoder.embeddings.norm.weight", mx.float32)
        self.enc = []
        for i in range(self.nl):
            p = f"encoder.layers.{i}."
            self.enc.append({"attn_norm": W(p + "attn_norm.weight", mx.float32) if i else None,
                             "Wqkv": W(p + "attn.Wqkv.weight").T, "Wo": W(p + "attn.Wo.weight").T,
                             "mlp_norm": W(p + "mlp_norm.weight", mx.float32),
                             "Wi": W(p + "mlp.Wi.weight").T, "Wo2": W(p + "mlp.Wo.weight").T,
                             "glob": i % c.global_attn_every_n_layers == 0})
        self.final_norm = W("encoder.final_norm.weight", mx.float32)
        self.br = (W("bridge.0.weight", mx.float32), W("bridge.1.gate_up.weight").T, W("bridge.1.down.weight").T,
                   W("bridge.2.weight", mx.float32))
        D = sd["fusion.logits"].shape[1]
        self.fnorms = [W(f"fusion.norms.{j}.weight", mx.float32) for j in range(D)]
        self.fw = mx.softmax(W("fusion.logits", mx.float32), axis=-1).astype(dtype)          # [J, D]
        self.gamma = W("fusion.gamma")
        self.fwo = W("fusion.wo.weight").T
        self.null = W("null")
        self.d = sd["emb.weight"].shape[1]
        self.emb = W("emb.weight")
        self.emb_s = (self.emb.astype(mx.float32) * math.sqrt(self.d)).astype(dtype)
        self.pos = W("pos.weight")
        self.dec = []
        for j in range(ck["cfg"]["dec_layers"]):
            p = f"layers.{j}."
            self.dec.append({k: (W(p + k + ".weight", mx.float32) if k.startswith("n") else W(p + k + ".weight").T)
                             for k in ("n1", "n2", "n3", "qkv", "o1", "q2", "kv2", "o2")})
            self.dec[-1]["gu"] = W(p + "ffn.gate_up.weight").T
            self.dec[-1]["down"] = W(p + "ffn.down.weight").T
        self.norm = W("norm.weight", mx.float32)
        vocab = json.loads((DATA / "vocab.json").read_text())
        self.dat_ids = vocab["dat_ids"]
        lut = np.zeros(vocab["dat_vocab"], dtype=np.int64)
        for i, x in enumerate(self.dat_ids):
            if x >= 0:
                lut[x] = i
        bc = [int(lut[x]) for x in vocab["byte_piece_ids"]]
        rr = ARMT.kana_byte_rules(bc, self.emb.shape[0], "cpu")
        self.rule_keys = [(p2, p1) for p2, p1, _ in rr]
        self.rule_masks = [mx.array(mk.numpy()) for _, _, mk in rr]
        self.sp = spm.SentencePieceProcessor(model_file=str(DAT_TOK))
        self.tok = AutoTokenizer.from_pretrained(str(ENC))
        self.far = {L: mx.where(mx.abs(mx.arange(L)[:, None] - mx.arange(L)[None]) > self.win, NEG, 0.0).astype(dtype)[None, None]
                    for L in range(1, buckets[-1] + 1)}
        # mx.compile with the weights captured as constants was SLOWER here (10-07, M1 Pro, L=32: encoder 22.9 vs
        # 13.4 ms, step 3.1 vs 1.4 ms), so plain lazy evaluation is the default.
        if compile_ == "args":                         # weights as explicit inputs of the compiled function
            self._names = ["tok_emb", "emb_norm", "enc", "final_norm", "br", "fnorms", "fw", "gamma", "fwo", "null", "dec",
                           "emb", "emb_s", "pos", "norm", "rule_masks", "far"]
            self.br = list(self.br)
            for p in self.enc:
                p["glob_"] = p.pop("glob")
            self._glob = [p.pop("glob_") for p in self.enc]
            self.far = [self.far[L] for L in range(1, buckets[-1] + 1)]
            ce, cs = mx.compile(self._bound(self._encode)), mx.compile(self._bound(self._step))
            self._enc_c = lambda *a: ce(self._state(), *a)
            self._step_c = lambda *a: cs(self._state(), *a)
        else:
            self._glob = [p.pop("glob") for p in self.enc]
            self.far = [self.far[L] for L in range(1, buckets[-1] + 1)]
            self._enc_c = mx.compile(self._encode) if compile_ else self._encode
            self._step_c = mx.compile(self._step) if compile_ else self._step
        del sd, ck

    # ---- pieces ----
    def _state(self):
        return [getattr(self, n) for n in self._names]

    def _bound(self, fn):
        def f(state, *a):                              # trace with the state's (tracer) arrays bound to self
            saved = self._state()
            for n, v in zip(self._names, state):
                setattr(self, n, v)
            try:
                return fn(*a)
            finally:
                for n, v in zip(self._names, saved):
                    setattr(self, n, v)
        return f

    def _far(self, L):
        return self.far[L - 1]

    def ln(self, x, w):
        return mx.fast.layer_norm(x.astype(mx.float32), w, None, self.eps).astype(self.dt)

    def rms(self, x, w):
        return mx.fast.rms_norm(x.astype(mx.float32), w, 1e-6).astype(self.dt)

    def heads(self, x, n):
        return x.reshape(1, n, self.H, self.hd).transpose(0, 2, 1, 3)

    def _encode(self, ids, kmask):
        L = ids.shape[1]
        h = self.ln(self.tok_emb[ids], self.emb_norm)                                       # [1, L, E]
        states = [h]
        for i, p in enumerate(self.enc):
            glob = self._glob[i]
            a = h if i == 0 else self.ln(h, p["attn_norm"])
            qkv = (a @ p["Wqkv"]).reshape(1, L, 3, self.H, self.hd)
            q, k, v = (qkv[:, :, j].transpose(0, 2, 1, 3) for j in range(3))
            base = self.theta[glob]
            q = mx.fast.rope(q, self.hd, traditional=False, base=base, scale=1.0, offset=0)
            k = mx.fast.rope(k, self.hd, traditional=False, base=base, scale=1.0, offset=0)
            mask = kmask if glob else kmask + self._far(L)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5, mask=mask)
            h = h + o.transpose(0, 2, 1, 3).reshape(1, L, -1) @ p["Wo"]
            x1, x2 = mx.split(self.ln(h, p["mlp_norm"]) @ p["Wi"], 2, axis=-1)
            h = h + (_gelu(x1) * x2) @ p["Wo2"]
            states.append(h)
        last = self.ln(h, self.final_norm)
        n0, gu, dn, n2 = self.br
        g, u = mx.split(self.rms(last, n0) @ gu, 2, axis=-1)
        base = self.rms((_silu(g) * u) @ dn, n2)
        normed = mx.stack([self.rms(s, n) for s, n in zip(states[:len(self.fnorms)], self.fnorms)])   # [D, 1, L, E]
        out = []
        for j, p in enumerate(self.dec):
            f = (self.fw[j][:, None, None, None] * normed).sum(0)
            mem = base + self.gamma[j] * (f @ self.fwo)
            mem = mx.concatenate([self.null[None], mem], axis=1)                              # [1, 2 + L, d]
            kk, vv = mx.split(mem @ p["kv2"], 2, axis=-1)
            out += [self.heads(kk, 2 + L), self.heads(vv, 2 + L)]
        return out

    def _step(self, tok, t, prev1, prev2, kc0, vc0, kc1, vc1, ck0, cv0, ck1, cv1, cmask):
        x = (self.emb_s[tok] + self.pos[t])[None]                                           # [1, 1, d]
        ar = mx.arange(self.T)
        smask = mx.concatenate([mx.where(ar < t, 0.0, NEG), mx.zeros((1,))]).astype(self.dt)[None, None, None]
        upd = (ar == t)[None, None, :, None]
        caches, news = ((kc0, vc0), (kc1, vc1)), []
        for p, (kc, vc), (ck, cv) in zip(self.dec, caches, ((ck0, cv0), (ck1, cv1))):
            q, k, v = mx.split(self.rms(x, p["n1"]) @ p["qkv"], 3, axis=-1)
            q, k, v = self.heads(q, 1), self.heads(k, 1), self.heads(v, 1)
            K, V = mx.concatenate([kc, k], axis=2), mx.concatenate([vc, v], axis=2)
            o = mx.fast.scaled_dot_product_attention(q, K, V, scale=self.hd ** -0.5, mask=smask)
            x = x + o.transpose(0, 2, 1, 3).reshape(1, 1, -1) @ p["o1"]
            q2 = self.heads(self.rms(x, p["n2"]) @ p["q2"], 1)
            o = mx.fast.scaled_dot_product_attention(q2, ck, cv, scale=self.hd ** -0.5, mask=cmask)
            x = x + o.transpose(0, 2, 1, 3).reshape(1, 1, -1) @ p["o2"]
            g, u = mx.split(self.rms(x, p["n3"]) @ p["gu"], 2, axis=-1)
            x = x + (_silu(g) * u) @ p["down"]
            news += [mx.where(upd, k, kc), mx.where(upd, v, vc)]
        lg = (self.rms(x, self.norm) @ self.emb.T).reshape(-1).astype(mx.float32)
        for (p2, p1), mk in zip(self.rule_keys, self.rule_masks):
            hit = (prev1 == p1) if p2 is None else (prev1 == p1) & (prev2 == p2)
            lg = mx.where(hit & mk, -mx.inf, lg)
        nxt = mx.argmax(lg).reshape(1).astype(mx.int32)
        return [nxt, t + 1, nxt, prev1, *news]

    # ---- host ----
    def translate(self, texts):
        text, = texts
        ids = self.tok(text, add_special_tokens=True, truncation=True, max_length=256)["input_ids"]
        n = len(ids)
        L = min(n, self.buckets[-1]) if self.exact else next((b for b in self.buckets if b >= n), self.buckets[-1])
        ids, n = ids[:L], min(n, L)
        a = np.full((1, L), 3, dtype=np.int32)
        a[0, :n] = ids
        km = np.zeros((1, 1, 1, L), dtype=np.float32)
        km[..., n:] = NEG
        cross = self._enc_c(mx.array(a), mx.array(km).astype(self.dt))
        M = 2 + L
        cm = np.full((1, 1, 1, M), NEG, dtype=np.float32)
        cm[..., :2 + n] = 0.0
        cmask = mx.array(cm).astype(self.dt)
        z = mx.zeros((1, self.H, self.T, self.hd), dtype=self.dt)
        state = [mx.array([BOS], dtype=mx.int32), mx.array([0], dtype=mx.int32), mx.array([-1], dtype=mx.int32),
                 mx.array([-1], dtype=mx.int32), z, z, z, z]
        steps = min(3 * n + 10, self.T)
        out, done, s = [], False, 0
        while s < steps and not done:
            toks = []
            for _ in range(min(self.chunk, steps - s)):
                r = self._step_c(*state, *cross, cmask)
                toks.append(r[0])
                state = [r[0], r[1], r[2], r[3], *r[4:]]
                s += 1
            mx.eval(toks, state)
            for tk in toks:
                v = int(tk.item())
                if v in (EOS, PAD):
                    done = True
                    break
                out.append(v)
        self.last = {"ntok": n, "nout": len(out)}
        return [self.decode_ids(out, self.dat_ids, self.sp)]


def _gelu(x):
    return 0.5 * x * (1 + mx.erf(x / math.sqrt(2)))


def _silu(x):
    return x * mx.sigmoid(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--compile", default="0", help="0 = lazy eval, 1 = mx.compile (weights as constants), args = weights as inputs")
    ap.add_argument("--exact", type=int, default=0, help="1 = encode at the true length (no bucket padding)")
    ap.add_argument("--ref", type=Path, help="jsonl with id/hyp to compare (e.g. MPS eager_fp32)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(x) for x in open(Path(__file__).with_name("sample500.jsonl"), encoding="utf-8")][:args.n]
    t0 = time.perf_counter()
    m = ZakkuriMLX(getattr(mx, args.dtype), chunk=args.chunk, compile_={"0": False, "1": True}.get(args.compile, args.compile), exact=bool(args.exact))
    load_s = time.perf_counter() - t0
    for r in rows[:args.warm]:
        m.translate([r["clean"]])
    hyps, ms = [], []
    for r in rows:
        t = time.perf_counter()
        hyps.append(m.translate([r["clean"]])[0])
        ms.append(1000 * (time.perf_counter() - t))
    a = np.array(ms)
    ref = {json.loads(x)["id"]: json.loads(x)["hyp"] for x in open(args.ref, encoding="utf-8")} if args.ref else None
    rep = {"mode": f"mlx_{args.dtype}_chunk{args.chunk}" + ({"0": "", "1": "_compiled"}.get(args.compile, "_compiled_args")) + ("_exact" if args.exact else ""), "n": len(a), "p50_ms": round(float(np.percentile(a, 50)), 1),
           "p90_ms": round(float(np.percentile(a, 90)), 1), "mean_ms": round(float(a.mean()), 1), "load_s": round(load_s, 1),
           "peak_gib": round(mx.get_peak_memory() / 2**30, 2) if hasattr(mx, "get_peak_memory") else None,
           "identical_to_ref": None if ref is None else round(float(np.mean([h == ref[r["id"]] for r, h in zip(rows, hyps)])), 4)}
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / f"{rep['mode']}.jsonl", "w", encoding="utf-8") as f:
        for r, h, t in zip(rows, hyps, ms):
            f.write(json.dumps({"id": r["id"], "hyp": h, "ms": round(t, 2)}, ensure_ascii=False) + "\n")
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
