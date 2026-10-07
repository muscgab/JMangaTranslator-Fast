#!/usr/bin/env python3
"""Core ML / ANE export of the single-block ARMT (R2 54k): an encoder model and a one-step decoder model, plus a
host-side greedy loop identical to ARMT.generate_ctx with an empty prefix (kana byte rules included).

encoder   ids [1, L] int32 (L in --buckets, padded with pad id 3; mask derived inside as ids != 3)
          -> ck0, cv0, ck1, cv1 [1, H, 2 + L, 64]   cross-attention K/V of both decoder layers
          (ModernBERT with eager attention, bidirectional sliding window |i - j| <= 64 on sliding layers, RoPE tables
          as constants; bridge, gamma depth fusion, null tokens, kv2 of each decoder layer).
decoder   x [1, 1, d] (host: emb[tok] * sqrt(d) + pos[t]), self K/V caches [1, H, T, 64] x 2 layers with additive mask
          smask [1, 1, 1, T] (positions < t), cross K/V padded to M = 2 + max bucket with additive cmask [1, 1, 1, M]
          -> logits [1, 1, V], k0, v0, k1, v1 [1, H, 1, 64] (host writes them into the caches at t).
fp16      every LayerNorm / RMSNorm divides its input by a calibrated per-norm power of two s and uses eps / s^2 (the
          same function): x^2 stays below the fp16 limit where |h| reaches ~2,865 (encoder layers 14-24), while small
          inputs keep s = 1 (a global s = 64 underflowed x^2 at the embedding norm: 10.6 % error, 2026-10-07).
Modes:
  check    torch fp32: export modules vs ARMT (memories / logits) and host loop vs Ours.translate on --n boxes
  convert  write <out>/{encoder,decoder}_<prec>.mlpackage
  eval     Core ML host loop on Manga109 clean (--n boxes): agreement with ARMT outputs, chrF, latency per box
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

# coremltools imports tensorflow when it is installed; TF's native library deadlocks / aborts on an absl mutex next to
# sentencepiece / tokenizers (2026-10-07). The export does not need TF, so hide it before coremltools is imported.
sys.modules.setdefault("tensorflow", None)
try:
    import coremltools  # noqa: F401
except ImportError:
    pass
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ar_mt"))
sys.path.insert(0, str(ROOT / "benchmarks/sakura_cmp"))
from model import ARMT  # noqa: E402
from train import BOS, EOS, PAD, decode_ids  # noqa: E402

NEG = -1e4


CALIB: dict | None = None          # module -> max |input| while calibrating (torch fp32 only, never while tracing)


def norm_scaled(x, mod, eps: float, center: bool):
    """LayerNorm (center=True, no bias) / RMSNorm of x with weight mod.weight, computed on x / s with eps / s^2 (the
    same function). s = mod._s is a per-norm power of two from calibrate(): 1 where inputs are small (dividing them
    would underflow x^2 in fp16), larger where the residual stream carries massive activations."""
    if CALIB is not None:
        CALIB[mod] = max(CALIB.get(mod, 0.0), float(x.detach().abs().max()))
    s = getattr(mod, "_s", 1.0)
    w = mod.weight
    x = x * (1.0 / s)
    if center:
        x = x - x.mean(-1, keepdim=True)
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps / (s * s)) * w


def rope(x, cos, sin):
    x1, x2 = x.chunk(2, -1)                      # rotate_half without shape-derived ints (coremltools aten::Int)
    return x * cos + torch.cat((-x2, x1), -1) * sin


class EncoderExport(nn.Module):
    def __init__(self, m: ARMT, lmax: int, s: float):
        super().__init__()
        enc, c = m.encoder, m.encoder.config
        self.c, self.s, self.lmax = c, s, lmax
        self.tok = enc.embeddings.tok_embeddings
        self.emb_norm = enc.embeddings.norm
        self.layers = enc.layers
        self.final_norm = enc.final_norm
        self.H, self.hd = c.num_attention_heads, c.hidden_size // c.num_attention_heads
        for kind in ("full_attention", "sliding_attention"):             # no length-dependent slicing: RoPE angles
            theta = c.rope_parameters[kind]["rope_theta"]                  # and the band mask come from positions
            inv = 1.0 / theta ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd)
            ang = torch.arange(lmax, dtype=torch.float32)[:, None] * torch.cat((inv, inv))[None]     # fp32 table:
            self.register_buffer(f"cos_{kind}", ang.cos(), persistent=False)       # angles up to ~lmax rad would
            self.register_buffer(f"sin_{kind}", ang.sin(), persistent=False)       # lose ~0.06 rad in fp16
        self.bridge, self.fusion, self.null = m.bridge, m.fusion, m.null
        self.register_buffer("fw", m.fusion.logits.detach().softmax(-1), persistent=False)       # [J, D]
        self.kv2 = nn.ModuleList(layer.kv2 for layer in m.layers)
        self.eps = c.norm_eps

    def ln(self, x, mod):
        return norm_scaled(x, mod, self.eps, True)

    def rms(self, x, mod):
        return norm_scaled(x, mod, mod.eps, False)

    def forward(self, ids):
        h = self.ln(self.tok(ids.long()), self.emb_norm)
        valid = (ids != 3).to(h.dtype)                                                # [1, L]
        pos = torch.cumsum(torch.ones_like(valid), 1)[0] - 1.0                        # [L] = 0 .. L-1
        key = (1.0 - valid)[:, None, None, :] * NEG
        far = ((pos[:, None] - pos[None, :]).abs() > self.c.sliding_window).to(h.dtype) * NEG
        masks = {"full_attention": key, "sliding_attention": key + far[None, None]}
        trig = {}
        for kind in ("full_attention", "sliding_attention"):
            pi = pos.long()                                                           # table lookup by position
            trig[kind] = (F.embedding(pi, getattr(self, f"cos_{kind}")).to(h.dtype),
                          F.embedding(pi, getattr(self, f"sin_{kind}")).to(h.dtype))
        return self.body(h, masks, trig)

    def body(self, h, masks, trig):
        states = [h]
        for i, layer in enumerate(self.layers):
            kind = layer.attention_type
            a = h if i == 0 else self.ln(h, layer.attn_norm)
            qkv = layer.attn.Wqkv(a).view(1, -1, 3, self.H, self.hd)
            q, k, v = (qkv[:, :, j].transpose(1, 2) for j in range(3))
            cos, sin = trig[kind]
            q, k = rope(q, cos, sin), rope(k, cos, sin)
            p = torch.softmax(q @ k.transpose(2, 3) * self.hd ** -0.5 + masks[kind], -1)
            h = h + layer.attn.Wo((p @ v).transpose(1, 2).reshape(1, -1, self.H * self.hd))
            x1, x2 = layer.mlp.Wi(self.ln(h, layer.mlp_norm)).chunk(2, -1)
            h = h + layer.mlp.Wo(F.gelu(x1) * x2)
            states.append(h)
        last = self.ln(h, self.final_norm)
        b = self.bridge                                                               # RMSNorm, SwiGLU, RMSNorm
        base = self.rms(b[1](self.rms(last, b[0])), b[2])                             # [1, L, d]
        normed = torch.stack([self.rms(st, n) for st, n in zip(states[:len(self.fusion.norms)], self.fusion.norms)])
        out = []
        for j, kv2 in enumerate(self.kv2):
            f = (self.fw[j][:, None, None, None] * normed).sum(0)                     # weighted depth sum
            mem = base + self.fusion.gamma[j] * self.fusion.wo(f)
            mem = torch.cat((self.null[None], mem), 1)                                 # [1, 2 + L, d]
            k, v = kv2(mem).chunk(2, -1)
            out += [k.view(1, -1, self.H, self.hd).transpose(1, 2), v.view(1, -1, self.H, self.hd).transpose(1, 2)]
        return tuple(out)


class EncoderStatic(EncoderExport):
    """All-ANE encoder for one fixed length L: the token-embedding lookup moves to the host (input x = raw token
    embeddings [1, L, E] before the embedding LayerNorm), the padding mask is an input (kmask [1, 1, 1, L], 0 / NEG),
    RoPE tables and the sliding-window band are constants of this L. No gather / cast / comparison left in the graph."""

    def __init__(self, m: ARMT, L: int):
        super().__init__(m, L, 1.0)
        pos = torch.arange(L, dtype=torch.float32)
        self.register_buffer("far", torch.where((pos[:, None] - pos[None, :]).abs() > self.c.sliding_window, NEG, 0.0)
                             [None, None], persistent=False)                                   # [1, 1, L, L]

    def forward(self, x, kmask):
        h = self.ln(x, self.emb_norm)
        masks = {"full_attention": kmask, "sliding_attention": kmask + self.far}
        trig = {k: (getattr(self, f"cos_{k}"), getattr(self, f"sin_{k}")) for k in masks}
        return self.body(h, masks, trig)


class DecoderExport(nn.Module):
    def __init__(self, m: ARMT, s: float):
        super().__init__()
        self.layers, self.norm, self.s = m.layers, m.norm, s
        self.register_buffer("emb_t", m.emb.weight.detach().T.contiguous(), persistent=False)   # [d, V]
        self.H, self.hd = m.layers[0].h, m.layers[0].hd

    def rms(self, x, mod):
        return norm_scaled(x, mod, mod.eps, False)

    def heads(self, x):
        return x.view(1, 1, self.H, self.hd).transpose(1, 2)

    def forward(self, x, kc0, vc0, kc1, vc1, smask, ck0, cv0, ck1, cv1, cmask):
        news = []
        zero = torch.zeros_like(smask[..., :1])
        for layer, kc, vc, ck, cv in zip(self.layers, (kc0, kc1), (vc0, vc1), (ck0, ck1), (cv0, cv1)):
            q, k, v = layer.qkv(self.rms(x, layer.n1)).chunk(3, -1)
            q, k, v = self.heads(q), self.heads(k), self.heads(v)
            K, V = torch.cat((kc, k), 2), torch.cat((vc, v), 2)
            p = torch.softmax(q @ K.transpose(2, 3) * self.hd ** -0.5 + torch.cat((smask, zero), -1), -1)
            x = x + layer.o1((p @ V).transpose(1, 2).reshape(1, 1, -1))
            q2 = self.heads(layer.q2(self.rms(x, layer.n2)))
            p = torch.softmax(q2 @ ck.transpose(2, 3) * self.hd ** -0.5 + cmask, -1)
            x = x + layer.o2((p @ cv).transpose(1, 2).reshape(1, 1, -1))
            x = x + layer.ffn(self.rms(x, layer.n3))
            news += [k, v]
        return (self.rms(x, self.norm) @ self.emb_t, *news)


class Host:
    """Greedy loop of ARMT.generate_ctx (empty prefix) around an encoder / decoder step backend."""

    def __init__(self, m: ARMT, vocab: dict, tok, sp, buckets, T: int, enc_fn, dec_fn):
        self.tok, self.sp, self.buckets, self.T = tok, sp, buckets, T
        self.enc_fn, self.dec_fn = enc_fn, dec_fn
        self.d = m.d
        self.emb = m.emb.weight.detach().float().numpy() * math.sqrt(m.d)
        self.pos = m.pos.weight.detach().float().numpy()
        self.dat_ids = vocab["dat_ids"]
        lut = np.zeros(vocab["dat_vocab"], dtype=np.int64)
        for i, dd in enumerate(self.dat_ids):
            if dd >= 0:
                lut[dd] = i
        bc = [int(lut[x]) for x in vocab["byte_piece_ids"]]
        self.rules = [(p2, p1, mk.numpy()) for p2, p1, mk in ARMT.kana_byte_rules(bc, self.emb.shape[0], "cpu")]
        self.M = 2 + max(buckets)
        self.H, self.hd = m.layers[0].h, m.layers[0].hd

    def __call__(self, text: str):
        ids = self.tok(text, add_special_tokens=True, truncation=True, max_length=256)["input_ids"]
        n = len(ids)
        Lb = next((b for b in self.buckets if b >= n), None)
        if Lb is None:                                       # longer than the largest bucket: keep the head
            ids, n, Lb = ids[:self.buckets[-1]], self.buckets[-1], self.buckets[-1]
        arr = np.full((1, Lb), 3, dtype=np.int32)
        arr[0, :n] = ids
        t0 = time.perf_counter()
        cross = self.enc_fn(arr)                             # 4 x [1, H, 2 + Lb, hd]
        t_enc = time.perf_counter() - t0
        cr = []
        for c in cross:
            z = np.zeros((1, self.H, self.M, self.hd), dtype=np.float32)
            z[:, :, :c.shape[2]] = c
            cr.append(z)
        cmask = np.full((1, 1, 1, self.M), NEG, dtype=np.float32)
        cmask[..., :2 + n] = 0.0
        caches = [np.zeros((1, self.H, self.T, self.hd), dtype=np.float32) for _ in range(4)]
        smask = np.full((1, 1, 1, self.T), NEG, dtype=np.float32)
        steps = min(3 * n + 10, 512, self.T + 1)
        prev1 = prev2 = -1
        tok, out = BOS, []
        for t in range(steps):
            x = (self.emb[tok] + self.pos[t])[None, None].astype(np.float32)
            logits, *news = self.dec_fn(x, caches, smask, cr, cmask)
            lg = logits.reshape(-1).astype(np.float32)
            for p2, p1, mk in self.rules:
                if prev1 == p1 and (p2 is None or prev2 == p2):
                    lg[mk] = -np.inf
            nxt = int(lg.argmax())
            if nxt == EOS or nxt == PAD:
                break
            out.append(nxt)
            if t < self.T:
                for c, nw in zip(caches, news):
                    c[:, :, t] = nw[:, :, 0]
                smask[..., t] = 0.0
            prev2, prev1, tok = prev1, nxt, nxt
        return {"hyp": decode_ids(out, self.dat_ids, self.sp), "ntok": n, "nout": len(out),
                "enc_ms": round(1000 * t_enc, 2), "ms": round(1000 * (time.perf_counter() - t0), 2)}


def load(args):
    from run_ours import Ours
    o = Ours(args.ckpt, args.encoder, args.data, "cpu")
    o.model.float().eval()
    return o


def torch_backends(o, enc, dec):
    def enc_fn(arr):
        with torch.no_grad():
            return [t.numpy() for t in enc(torch.from_numpy(arr))]

    def dec_fn(x, caches, smask, cr, cmask):
        with torch.no_grad():
            r = dec(torch.from_numpy(x), *map(torch.from_numpy, caches), torch.from_numpy(smask),
                    *map(torch.from_numpy, cr), torch.from_numpy(cmask))
        return [t.numpy() for t in r]
    return enc_fn, dec_fn


def calibrate(m, o, enc, dec, vocab, buckets, T, texts, cap: float, path: Path):
    """Per-norm scales from the max |input| seen on texts (torch fp32, encoder + host decoding loop):
    s = 2^ceil(log2(max(1, absmax / cap))). Saved to path (module name -> s, absmax) and set as mod._s."""
    global CALIB
    names = {mod: n for n, mod in m.named_modules()}
    if path.exists():
        saved = json.loads(path.read_text())
        for mod, n in names.items():
            if n in saved:
                mod._s = saved[n]["s"]
        return saved
    CALIB = {}
    host = Host(m, vocab, o.tok, o.sp, buckets, T, *torch_backends(o, enc, dec))
    for t in texts:
        host(t)
    seen, CALIB = CALIB, None
    out = {}
    for mod, a in seen.items():
        mod._s = float(2 ** math.ceil(math.log2(max(1.0, a / cap))))
        out[names[mod]] = {"s": mod._s, "absmax": round(a, 2)}
    path.write_text(json.dumps(out, indent=1) + "\n")
    return out


def coreml_backends(out: Path, prec: str, units: str, dprec: str | None = None, dunits: str | None = None,
                    static: bool = False, table=None, buckets=()):
    import coremltools as ct
    if static:                     # all-ANE encoder: one function per bucket, embedding lookup + padding mask on host
        fm = {b: ct.models.MLModel(str(out / f"encoder_static_{prec}.mlpackage"), function_name=f"L{b}",
                                   compute_units=getattr(ct.ComputeUnit, units)) for b in buckets}
    else:
        em = ct.models.MLModel(str(out / f"encoder_{prec}.mlpackage"), compute_units=getattr(ct.ComputeUnit, units))
    dm = ct.models.MLModel(str(out / f"decoder_{dprec or prec}.mlpackage"),
                           compute_units=getattr(ct.ComputeUnit, dunits or units))
    names = ("ck0", "cv0", "ck1", "cv1")

    def enc_fn(arr):
        if static:
            L = arr.shape[1]
            km = np.where(arr == 3, NEG, 0.0).astype(np.float32)[:, None, None, :]
            r = fm[L].predict({"x": table[arr], "kmask": km})
        else:
            r = em.predict({"ids": arr})
        return [r[k] for k in names]

    def dec_fn(x, caches, smask, cr, cmask):
        feed = {"x": x, "kc0": caches[0], "vc0": caches[1], "kc1": caches[2], "vc1": caches[3], "smask": smask,
                "ck0": cr[0], "cv0": cr[1], "ck1": cr[2], "cv1": cr[3], "cmask": cmask}
        r = dm.predict(feed)
        return [r["logits"], r["k0"], r["v0"], r["k1"], r["v1"]]
    return enc_fn, dec_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("check", "convert", "eval", "encerr"))
    ap.add_argument("--ckpt", type=Path, default=ROOT / "artifacts/r2_20261006/weights/r2_step54000.pt")
    ap.add_argument("--encoder", default=str(ROOT / "artifacts/ar_mt_20261005/weights/dapt_v1_model"))
    ap.add_argument("--data", type=Path, default=ROOT / "artifacts/ar_mt_20261005/data_v2_ext")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/coreml_20261007")
    ap.add_argument("--buckets", default="32,64,128")
    ap.add_argument("--T", type=int, default=128, help="self-attention cache length (max output tokens ~ T + 1)")
    ap.add_argument("--norm-cap", type=float, default=32.0, help="calibrated norm scale keeps max|x|/s <= cap")
    ap.add_argument("--prec", choices=("fp32", "fp16"), default="fp16")
    ap.add_argument("--units", default="CPU_AND_NE", help="ALL / CPU_ONLY / CPU_AND_GPU / CPU_AND_NE")
    ap.add_argument("--dec-prec", choices=("fp32", "fp16"), help="decoder precision (default --prec)")
    ap.add_argument("--dec-units", help="decoder compute units (default --units)")
    ap.add_argument("--backend", choices=("torch", "coreml"), default="coreml")
    ap.add_argument("--static", type=int, default=0, help="1 = all-ANE encoder (encoder_static_<prec>.mlpackage)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(4)
    buckets = [int(b) for b in args.buckets.split(",")]
    o = load(args)
    m = o.model
    enc = EncoderExport(m, max(buckets), 1.0).eval()
    dec = DecoderExport(m, 1.0).eval()
    vocab = json.loads((args.data / "vocab.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    from common import load_m109, load_murasaki
    recs = load_m109()
    # calibration texts: Manga109 boxes 200-999 (evaluation uses the first 200) + 300 Murasaki segments (longer)
    ctexts = [r["clean"] for r in recs[200:1000]] + [s_ for r in load_murasaki() for s_ in r["segs"]][:300]
    scales = calibrate(m, o, enc, dec, vocab, buckets, args.T, ctexts, args.norm_cap, args.out / "norm_scales.json")
    print(json.dumps({"norm_scales": {str(k): v for k, v in sorted(
        __import__("collections").Counter(x["s"] for x in scales.values()).items())}}), flush=True)

    if args.mode == "check":
        # 1) memories: export encoder vs ARMT._encode (+ null, kv2), on real boxes padded to a bucket
        worst = {}
        for r in recs[:args.n]:
            ids = o.tok(r["clean"], add_special_tokens=True, truncation=True, max_length=256)["input_ids"]
            n = len(ids)
            Lb = next(b for b in buckets if b >= n)
            arr = torch.full((1, Lb), 3, dtype=torch.int32)
            arr[0, :n] = torch.tensor(ids)
            with torch.no_grad():
                got = enc(arr)
                s = torch.tensor([ids])
                mems, _ = m.memories(s, torch.ones_like(s, dtype=torch.bool), False)
                ref = [t for layer, mem in zip(m.layers, mems) for t in layer.cross_kv(mem)]
            for name, a, b in zip(("ck0", "cv0", "ck1", "cv1"), got, ref):
                e = float((a[:, :, :n + 2] - b).abs().max() / b.abs().max())
                worst[name] = max(worst.get(name, 0.0), e)
        print(json.dumps({"check": "encoder rel max err over valid positions", **worst}), flush=True)
        # 2) full host loop (torch backends) vs Ours.translate
        host = Host(m, vocab, o.tok, o.sp, buckets, args.T, *torch_backends(o, enc, dec))
        same = 0
        diffs = []
        for r in recs[:args.n]:
            a = host(r["clean"])["hyp"]
            b = o.translate([r["clean"]])[0]
            same += a == b
            if a != b and len(diffs) < 5:
                diffs.append([r["clean"], a, b])
        print(json.dumps({"check": "host loop vs ARMT", "n": args.n, "identical": same, "diffs": diffs},
                         ensure_ascii=False), flush=True)
        # 3) fp16 simulation in torch (CPU): any inf / nan in the encoder outputs?
        import copy
        enc16 = copy.deepcopy(enc).half().eval()
        bad = 0
        for r in recs[:min(args.n, 50)]:
            ids = o.tok(r["clean"], add_special_tokens=True, truncation=True, max_length=256)["input_ids"]
            arr = torch.full((1, next(b for b in buckets if b >= len(ids))), 3, dtype=torch.int32)
            arr[0, :len(ids)] = torch.tensor(ids)
            with torch.no_grad():
                bad += any(not torch.isfinite(t).all() for t in enc16(arr))
        print(json.dumps({"check": "torch fp16 encoder non-finite outputs", "boxes": min(args.n, 50), "bad": bad}),
              flush=True)
        return

    if args.mode == "encerr":
        # encoder outputs of the Core ML model (--prec / --units) and of torch fp16 (CPU) vs torch fp32, valid positions
        enc_fn, _ = coreml_backends(args.out, args.prec, args.units)
        import copy
        enc16 = copy.deepcopy(enc).half().eval()
        err = {"coreml": [], "torch_fp16": []}
        for r in recs[:args.n]:
            ids = o.tok(r["clean"], add_special_tokens=True, truncation=True, max_length=256)["input_ids"]
            n = len(ids)
            arr = np.full((1, next(b for b in buckets if b >= n)), 3, dtype=np.int32)
            arr[0, :n] = ids
            with torch.no_grad():
                ref = [t[:, :, :n + 2].numpy() for t in enc(torch.from_numpy(arr))]
                t16 = [t[:, :, :n + 2].float().numpy() for t in enc16(torch.from_numpy(arr))]
            cm = [c[:, :, :n + 2].astype(np.float32) for c in enc_fn(arr)]
            for name, got in (("coreml", cm), ("torch_fp16", t16)):
                err[name].append([float(np.linalg.norm(g - r_) / np.linalg.norm(r_)) for g, r_ in zip(got, ref)])
        rep = {k: {"rel_l2_mean": np.round(np.mean(v, 0), 5).tolist(), "rel_l2_max": np.round(np.max(v, 0), 5).tolist()}
               for k, v in err.items()}
        print(json.dumps({"encerr": f"{args.prec}_{args.units}", "n": args.n, "outputs": "ck0 cv0 ck1 cv1", **rep}),
              flush=True)
        return

    if args.mode == "convert" and args.static:
        import coremltools as ct
        prec = ct.precision.FLOAT16 if args.prec == "fp16" else ct.precision.FLOAT32
        E_ = m.encoder.config.hidden_size
        desc = ct.utils.MultiFunctionDescriptor()
        parts = []
        for b in buckets:
            es = EncoderStatic(m, b).eval()
            ex = (torch.randn(1, b, E_) * 0.05, torch.zeros(1, 1, 1, b))
            with torch.no_grad():
                ts = torch.jit.trace(es, ex, check_trace=False)
            mb = ct.convert(ts, inputs=[ct.TensorType(name="x", shape=(1, b, E_)), ct.TensorType(name="kmask", shape=(1, 1, 1, b))],
                            outputs=[ct.TensorType(name=k) for k in ("ck0", "cv0", "ck1", "cv1")],
                            convert_to="mlprogram", compute_precision=prec, minimum_deployment_target=ct.target.macOS15)
            pth = args.out / f"_enc_static_L{b}_{args.prec}.mlpackage"
            mb.save(str(pth))
            parts.append(pth)
            desc.add_function(str(pth), src_function_name="main", target_function_name=f"L{b}")
        desc.default_function_name = f"L{buckets[0]}"
        ct.utils.save_multifunction(desc, str(args.out / f"encoder_static_{args.prec}.mlpackage"))
        import shutil
        for pth in parts:
            shutil.rmtree(pth)
        print(json.dumps({"converted": "encoder_static", "functions": [f"L{b}" for b in buckets]}), flush=True)
        return

    if args.mode == "convert":
        import coremltools as ct
        prec = ct.precision.FLOAT16 if args.prec == "fp16" else ct.precision.FLOAT32
        ex = torch.full((1, buckets[0]), 3, dtype=torch.int32)
        ex[0, :5] = torch.tensor([6, 100, 200, 300, 4])
        with torch.no_grad():
            te = torch.jit.trace(enc, ex, check_trace=False)
        t0 = time.time()
        me = ct.convert(te, inputs=[ct.TensorType(name="ids", shape=ct.EnumeratedShapes(
            shapes=[[1, b] for b in buckets], default=[1, buckets[0]]), dtype=np.int32)],
            outputs=[ct.TensorType(name=k) for k in ("ck0", "cv0", "ck1", "cv1")],
            convert_to="mlprogram", compute_precision=prec, minimum_deployment_target=ct.target.macOS15)
        me.save(str(args.out / f"encoder_{args.prec}.mlpackage"))
        print(json.dumps({"converted": "encoder", "secs": round(time.time() - t0, 1)}), flush=True)
        H, hd, d, M, T = dec.H, dec.hd, m.d, 2 + max(buckets), args.T
        shapes = {"x": (1, 1, d), "kc0": (1, H, T, hd), "vc0": (1, H, T, hd), "kc1": (1, H, T, hd), "vc1": (1, H, T, hd),
                  "smask": (1, 1, 1, T), "ck0": (1, H, M, hd), "cv0": (1, H, M, hd), "ck1": (1, H, M, hd),
                  "cv1": (1, H, M, hd), "cmask": (1, 1, 1, M)}
        exs = tuple(torch.zeros(s) for s in shapes.values())
        with torch.no_grad():
            td = torch.jit.trace(dec, exs, check_trace=False)
        t0 = time.time()
        md = ct.convert(td, inputs=[ct.TensorType(name=k, shape=s) for k, s in shapes.items()],
                        outputs=[ct.TensorType(name=k) for k in ("logits", "k0", "v0", "k1", "v1")],
                        convert_to="mlprogram", compute_precision=prec, minimum_deployment_target=ct.target.macOS15)
        md.save(str(args.out / f"decoder_{args.prec}.mlpackage"))
        print(json.dumps({"converted": "decoder", "secs": round(time.time() - t0, 1)}), flush=True)
        return

    # eval: Manga109 clean, first --n boxes; reference = ARMT (Ours.translate, CPU fp32)
    try:
        import sacrebleu
    except ImportError:                                      # base env has none: chrF is added later (comet env)
        sacrebleu = None
    from common import M109
    refs = {json.loads(x)["id"]: json.loads(x)["reference"] for x in open(M109 / "refs.jsonl", encoding="utf-8")}
    table = m.encoder.embeddings.tok_embeddings.weight.detach().float().numpy() if args.static else None
    fns = torch_backends(o, enc, dec) if args.backend == "torch" else \
        coreml_backends(args.out, args.prec, args.units, args.dec_prec, args.dec_units, bool(args.static), table, buckets)
    host = Host(m, vocab, o.tok, o.sp, buckets, args.T, *fns)
    for r in recs[:10]:                                      # warm-up (model load / ANE compile)
        host(r["clean"])
    rows = []
    for r in recs[:args.n]:
        x = host(r["clean"])
        x.update(id=r["id"], ref_armt=o.translate([r["clean"]])[0])
        rows.append(x)
    tag = args.tag or f"{args.backend}_{args.prec}_{args.units}" + \
        (f"__dec_{args.dec_prec or args.prec}_{args.dec_units or args.units}" if args.dec_prec or args.dec_units else "")
    with open(args.out / f"eval_{tag}.jsonl", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    ms = np.array([x["ms"] for x in rows])
    em = np.array([x["enc_ms"] for x in rows])
    nout = np.array([x["nout"] for x in rows])
    hyp, ref_armt = [x["hyp"] for x in rows], [x["ref_armt"] for x in rows]
    gold = [refs[x["id"]] for x in rows]
    rep = {"tag": tag, "n": len(rows), "identical_to_armt": round(float(np.mean([a == b for a, b in zip(hyp, ref_armt)])), 4),
           "chrf": sacrebleu and round(sacrebleu.corpus_chrf(hyp, [gold]).score, 2),
           "chrf_armt": sacrebleu and round(sacrebleu.corpus_chrf(ref_armt, [gold]).score, 2),
           "ms_p50": round(float(np.median(ms)), 1), "ms_p90": round(float(np.percentile(ms, 90)), 1),
           "enc_ms_p50": round(float(np.median(em)), 1),
           "dec_ms_per_step": round(float(((ms - em) / (nout + 1)).mean()), 2), "mean_out_tokens": round(float(nout.mean()), 1),
           "bad_outputs": sum(not h.strip() for h in hyp)}
    (args.out / f"eval_{tag}.json").write_text(json.dumps(rep, ensure_ascii=False) + "\n")
    print(json.dumps(rep, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
