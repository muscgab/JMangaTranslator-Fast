"""MLX backend (Apple-silicon GPU), batch size 1, no PyTorch needed.

Same function as torch_model.EncoderStatic / DecoderStep: ModernBERT-ja (global attention every 3rd layer with RoPE
theta 160000, the other layers local with |i - j| <= 64 and theta 10000), exact GELU, LayerNorm without bias; depth
fusion, SwiGLU bridge, 2 null tokens, kv2 per decoder layer. One decoder step with a fixed self-attention cache
(T = 128, written at position t with a where-mask), kana byte rules and argmax on the device; steps are chained lazily
`chunk` at a time and evaluated once per chunk (tokens after <eos> are discarded). Weights fp16, every normalization
in fp32 (the encoder carries activations up to ~2.9e3; x^2 overflows fp16). compile="args" compiles the encoder and
the step with the weights passed as inputs (compiling with the weights captured as constants was slower).
"""
from __future__ import annotations

import json
import math

import mlx.core as mx
import numpy as np

from .common import NEG, Release, Text, kana_rules


def _gelu(x):
    return 0.5 * x * (1 + mx.erf(x / math.sqrt(2)))


def _silu(x):
    return x * mx.sigmoid(x)


class MLX:
    def __init__(self, root, dtype=mx.float16, buckets=(32, 64, 128), T: int = 128, chunk: int = 8,
                 compile_: str | bool = "args"):
        self.rel = rel = Release(root)
        self.text = Text(rel)
        sd = mx.load(str(rel.root / "model.safetensors"))
        c = json.loads((rel.root / "tokenizer/config.json").read_text())
        self.dt, self.buckets, self.T, self.chunk = dtype, buckets, T, chunk
        W = lambda k, dt=dtype: sd[k].astype(dt)  # noqa: E731
        self.nl, self.H = c["num_hidden_layers"], c["num_attention_heads"]
        win = c.get("sliding_window") or c["local_attention"] // 2      # transformers: sliding_window = local_attention // 2
        self.hd, self.win, self.eps = c["hidden_size"] // self.H, win, c["norm_eps"]
        rp = c.get("rope_parameters") or {}
        self.theta = {True: rp.get("full_attention", {}).get("rope_theta", c.get("global_rope_theta")),
                      False: rp.get("sliding_attention", {}).get("rope_theta", c.get("local_rope_theta"))}
        every = c["global_attn_every_n_layers"]
        self.tok_emb = W("encoder.embeddings.tok_embeddings.weight")
        self.emb_norm = W("encoder.embeddings.norm.weight", mx.float32)
        self.enc = []
        for i in range(self.nl):
            p = f"encoder.layers.{i}."
            self.enc.append({"attn_norm": W(p + "attn_norm.weight", mx.float32) if i else None,
                             "Wqkv": W(p + "attn.Wqkv.weight").T, "Wo": W(p + "attn.Wo.weight").T,
                             "mlp_norm": W(p + "mlp_norm.weight", mx.float32),
                             "Wi": W(p + "mlp.Wi.weight").T, "Wo2": W(p + "mlp.Wo.weight").T})
        self._glob = [i % every == 0 for i in range(self.nl)]
        self.final_norm = W("encoder.final_norm.weight", mx.float32)
        self.br = [W("bridge.0.weight", mx.float32), W("bridge.1.gate_up.weight").T, W("bridge.1.down.weight").T,
                   W("bridge.2.weight", mx.float32)]
        D = sd["fusion.logits"].shape[1]
        self.fnorms = [W(f"fusion.norms.{j}.weight", mx.float32) for j in range(D)]
        self.fw = mx.softmax(W("fusion.logits", mx.float32), axis=-1).astype(dtype)          # [J, D]
        self.gamma = W("fusion.gamma")
        self.fwo = W("fusion.wo.weight").T
        self.null = W("null")
        self.d = sd["emb.weight"].shape[1]
        self.emb = W("emb.weight")
        self.emb_s = (self.emb.astype(mx.float32) * math.sqrt(self.d)).astype(dtype)      # scaled from the fp16 table
        self.pos = W("pos.weight")
        self.dec = []
        for j in range(rel.cfg["dec_layers"]):
            p = f"layers.{j}."
            self.dec.append({k: (W(p + k + ".weight", mx.float32) if k.startswith("n") else W(p + k + ".weight").T)
                             for k in ("n1", "n2", "n3", "qkv", "o1", "q2", "kv2", "o2")})
            self.dec[-1]["gu"] = W(p + "ffn.gate_up.weight").T
            self.dec[-1]["down"] = W(p + "ffn.down.weight").T
        self.norm = W("norm.weight", mx.float32)
        rr = kana_rules(rel.byte_compact, self.emb.shape[0])
        self.rule_keys = [(p2, p1) for p2, p1, _ in rr]
        self.rule_masks = [mx.array(mk) for _, _, mk in rr]
        self.far = [mx.where(mx.abs(mx.arange(L)[:, None] - mx.arange(L)[None]) > self.win, NEG, 0.0).astype(dtype)[None, None]
                    for L in range(1, buckets[-1] + 1)]
        del sd
        if compile_ == "args":
            self._names = ["tok_emb", "emb_norm", "enc", "final_norm", "br", "fnorms", "fw", "gamma", "fwo", "null", "dec",
                           "emb", "emb_s", "pos", "norm", "rule_masks", "far"]
            ce, cs = mx.compile(self._bound(self._encode)), mx.compile(self._bound(self._step))
            self._enc_c = lambda *a: ce(self._state(), *a)
            self._step_c = lambda *a: cs(self._state(), *a)
        else:
            self._enc_c = mx.compile(self._encode) if compile_ else self._encode
            self._step_c = mx.compile(self._step) if compile_ else self._step

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
            mask = kmask if glob else kmask + self.far[L - 1]
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

    def translate(self, text: str) -> str:
        rel = self.rel
        ids = self.text.encode(text)
        n = len(ids)
        L = next((b for b in self.buckets if b >= n), self.buckets[-1])
        ids, n = ids[:L], min(n, L)
        a = np.full((1, L), rel.cfg["src_pad"], dtype=np.int32)
        a[0, :n] = ids
        km = np.zeros((1, 1, 1, L), dtype=np.float32)
        km[..., n:] = NEG
        cross = self._enc_c(mx.array(a), mx.array(km).astype(self.dt))
        cm = np.full((1, 1, 1, 2 + L), NEG, dtype=np.float32)
        cm[..., :2 + n] = 0.0
        cmask = mx.array(cm).astype(self.dt)
        z = mx.zeros((1, self.H, self.T, self.hd), dtype=self.dt)
        state = [mx.array([rel.bos], dtype=mx.int32), mx.array([0], dtype=mx.int32), mx.array([-1], dtype=mx.int32),
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
                if v in (rel.eos, rel.pad):
                    done = True
                    break
                out.append(v)
        return self.text.decode(out)
