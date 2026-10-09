"""PyTorch model: ModernBERT-ja encoder (transformers) + depth fusion + SwiGLU bridge + 2 null tokens + a 2-layer
autoregressive decoder. Module names match model.safetensors. Also the fixed-shape encoder / one-step decoder modules
used by the CUDA Graphs backend and by the ONNX / Core ML exports (fp16 runs need norm_scales.json)."""
from __future__ import annotations

import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import NEG, Release


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_in: int, hidden: int, d_out: int):
        super().__init__()
        self.gate_up = nn.Linear(d_in, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d_out, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, -1)
        return self.down(F.silu(g) * u)


class Fusion(nn.Module):
    """Decoder layer j reads base + gamma_j * Wo(sum_l softmax(w_j)_l RMSNorm_l(h_l)) over encoder states h_0..h_{L-1}."""

    def __init__(self, e: int, d: int, dec_layers: int, depths: int):
        super().__init__()
        self.norms = nn.ModuleList(RMSNorm(e) for _ in range(depths))
        self.wo = nn.Linear(e, d, bias=False)
        self.gamma = nn.Parameter(torch.zeros(dec_layers, d))
        self.logits = nn.Parameter(torch.zeros(dec_layers, depths))

    def forward(self, states):
        h = torch.stack([n(s) for n, s in zip(self.norms, states)], 0)            # [D, B, N, E]
        f = torch.einsum("jl,lbne->jbne", self.logits.softmax(-1).to(h.dtype), h)
        return self.gamma[:, None, None].to(h.dtype) * self.wo(f)                  # [J, B, N, d]


class DecoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.n1, self.n2, self.n3 = RMSNorm(d), RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o1 = nn.Linear(d, d, bias=False)
        self.q2 = nn.Linear(d, d, bias=False)
        self.kv2 = nn.Linear(d, 2 * d, bias=False)
        self.o2 = nn.Linear(d, d, bias=False)
        self.ffn = SwiGLU(d, ffn, d)

    def heads(self, x):
        return x.view(x.shape[0], x.shape[1], self.h, self.hd).transpose(1, 2)

    def merge(self, y):
        return y.transpose(1, 2).reshape(y.shape[0], y.shape[2], -1)

    def cross_kv(self, mem):
        k, v = self.kv2(mem).chunk(2, -1)
        return self.heads(k), self.heads(v)

    def step(self, x, t, cache, ck, cv, cmask):
        """One position; cache [2, B, H, T, hd] holds the self-attention K/V of earlier positions."""
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        cache[0, :, :, t] = self.heads(k)[:, :, 0]
        cache[1, :, :, t] = self.heads(v)[:, :, 0]
        x = x + self.o1(self.merge(F.scaled_dot_product_attention(
            self.heads(q), cache[0, :, :, : t + 1], cache[1, :, :, : t + 1])))
        x = x + self.o2(self.merge(F.scaled_dot_product_attention(
            self.heads(self.q2(self.n2(x))), ck, cv, attn_mask=cmask)))
        return x + self.ffn(self.n3(x))


class Translator(nn.Module):
    def __init__(self, cfg: dict, encoder_dir: str):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        enc_cfg = AutoConfig.from_pretrained(encoder_dir)
        self.encoder = AutoModel.from_config(enc_cfg, attn_implementation="sdpa")
        e = enc_cfg.hidden_size
        d = self.d = e
        L = cfg["dec_layers"]
        self.cfg = cfg
        self.bridge = nn.Sequential(RMSNorm(e), SwiGLU(e, cfg["bridge_hidden"], d), RMSNorm(d))
        self.fusion = Fusion(e, d, L, enc_cfg.num_hidden_layers)
        self.null = nn.Parameter(torch.zeros(cfg["null_tokens"], d))
        self.emb = nn.Embedding(cfg["vocab"], d)
        self.pos = nn.Embedding(cfg["max_tgt"], d)
        self.layers = nn.ModuleList(DecoderLayer(d, d // 64, cfg["ffn"]) for _ in range(L))
        self.norm = RMSNorm(d)

    def memories(self, src, src_mask):
        out = self.encoder(input_ids=src, attention_mask=src_mask.long(), output_hidden_states=True)
        base = self.bridge(out.last_hidden_state)
        mems = base[None].expand(len(self.layers), -1, -1, -1) + self.fusion(list(out.hidden_states[:-1]))
        b = src.shape[0]
        mems = torch.cat((self.null[None, None].expand(len(self.layers), b, -1, -1).to(mems.dtype), mems), 2)
        mask = torch.cat((torch.ones(b, self.null.shape[0], dtype=torch.bool, device=src.device), src_mask), 1)
        return mems, mask[:, None, None, :]

    def embed(self, ids, start: int = 0):
        pos = torch.arange(start, start + ids.shape[1], device=ids.device)
        return self.emb(ids) * math.sqrt(self.d) + self.pos(pos)[None]


def load(rel: Release, device: str = "cpu") -> Translator:
    """Build on the meta device so the random initialization (about 30 s on a laptop CPU) is skipped, then take the
    checkpoint tensors as parameters. The RoPE tables are not in the checkpoint and are rebuilt."""
    from safetensors.torch import load_file
    # Import the transformers model classes before entering the meta device: a first import inside it fails on
    # torch 2.8 with a circular torch._dynamo import (seen on Windows, torch 2.8.0 + transformers 5.19.0).
    from transformers import AutoModel  # noqa: F401
    sd = load_file(str(rel.root / "model.safetensors"))
    with torch.device("meta"):
        m = Translator(rel.cfg, str(rel.root / "tokenizer"))
    m.load_state_dict(sd, strict=True, assign=True)
    rot = getattr(m.encoder, "rotary_emb", None)
    if rot is not None:
        m.encoder.rotary_emb = type(rot)(config=m.encoder.config)
    if any(x.is_meta for x in [*m.parameters(), *m.buffers()]):     # other transformers layouts: slow, safe path
        m = Translator(rel.cfg, str(rel.root / "tokenizer"))
        m.load_state_dict(sd, strict=True)
    return m.float().eval().to(device)


def kana_rules(rel: Release, vocab: int, device):
    from .common import kana_rules as np_rules
    return [(p2, p1, torch.from_numpy(m).to(device)) for p2, p1, m in np_rules(rel.byte_compact, vocab)]


# ------------------------------------------------------------------ fixed-shape modules (CUDA Graphs, ONNX, Core ML)
def apply_norm_scales(m: Translator, rel: Release) -> None:
    """Set mod._s from norm_scales.json (power-of-two input scales; exact in fp32, needed in fp16)."""
    saved = json.loads((rel.root / "norm_scales.json").read_text())
    for name, mod in m.named_modules():
        if name in saved:
            mod._s = saved[name]["s"]


def norm_scaled(x, mod, eps: float, center: bool):
    """LayerNorm (center=True, no bias) / RMSNorm of x computed on x / s with eps / s^2 (the same function);
    s = mod._s keeps x^2 inside the fp16 range where the encoder carries massive activations (|h| up to ~2.9e3)."""
    s = getattr(mod, "_s", 1.0)
    x = x * (1.0 / s)
    if center:
        x = x - x.mean(-1, keepdim=True)
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps / (s * s)) * mod.weight


def rope(x, cos, sin):
    x1, x2 = x.chunk(2, -1)
    return x * cos + torch.cat((-x2, x1), -1) * sin


class EncoderExport(nn.Module):
    """ModernBERT + fusion + bridge + null tokens + kv2 of both decoder layers -> ck0, cv0, ck1, cv1 [1, H, 2 + L, hd].
    Eager attention, bidirectional sliding window |i - j| <= sliding_window on local layers, RoPE from fp32 tables."""

    def __init__(self, m: Translator, lmax: int):
        super().__init__()
        enc, c = m.encoder, m.encoder.config
        self.c, self.lmax = c, lmax
        self.tok = enc.embeddings.tok_embeddings
        self.emb_norm = enc.embeddings.norm
        self.layers = enc.layers
        self.final_norm = enc.final_norm
        self.H, self.hd = c.num_attention_heads, c.hidden_size // c.num_attention_heads
        for kind in ("full_attention", "sliding_attention"):
            rp = getattr(c, "rope_parameters", None)                   # transformers >= 5; older: *_rope_theta
            theta = rp[kind]["rope_theta"] if rp else (c.global_rope_theta if kind == "full_attention" else c.local_rope_theta)
            inv = 1.0 / theta ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd)
            ang = torch.arange(lmax, dtype=torch.float32)[:, None] * torch.cat((inv, inv))[None]
            self.register_buffer(f"cos_{kind}", ang.cos(), persistent=False)
            self.register_buffer(f"sin_{kind}", ang.sin(), persistent=False)
        self.bridge, self.fusion, self.null = m.bridge, m.fusion, m.null
        self.register_buffer("fw", m.fusion.logits.detach().softmax(-1), persistent=False)       # [J, D]
        self.kv2 = nn.ModuleList(layer.kv2 for layer in m.layers)
        self.eps = c.norm_eps

    def ln(self, x, mod):
        return norm_scaled(x, mod, self.eps, True)

    def rms(self, x, mod):
        return norm_scaled(x, mod, mod.eps, False)

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
        b = self.bridge
        base = self.rms(b[1](self.rms(last, b[0])), b[2])                             # [1, L, d]
        normed = torch.stack([self.rms(st, n) for st, n in zip(states[:len(self.fusion.norms)], self.fusion.norms)])
        out = []
        for j, kv2 in enumerate(self.kv2):
            f = (self.fw[j][:, None, None, None] * normed).sum(0)
            mem = base + self.fusion.gamma[j] * self.fusion.wo(f)
            mem = torch.cat((self.null[None], mem), 1)                                 # [1, 2 + L, d]
            k, v = kv2(mem).chunk(2, -1)
            out += [k.view(1, -1, self.H, self.hd).transpose(1, 2), v.view(1, -1, self.H, self.hd).transpose(1, 2)]
        return tuple(out)

    def forward(self, ids):                       # ids [1, L] padded with src_pad (3); length axis may be dynamic
        h = self.ln(self.tok(ids.long()), self.emb_norm)
        valid = (ids != 3).to(h.dtype)
        pos = torch.cumsum(torch.ones_like(valid), 1)[0] - 1.0
        key = (1.0 - valid)[:, None, None, :] * NEG
        far = ((pos[:, None] - pos[None, :]).abs() > self.c.sliding_window).to(h.dtype) * NEG
        masks = {"full_attention": key, "sliding_attention": key + far[None, None]}
        trig = {}
        for kind in ("full_attention", "sliding_attention"):
            pi = pos.long()
            trig[kind] = (F.embedding(pi, getattr(self, f"cos_{kind}")).to(h.dtype),
                          F.embedding(pi, getattr(self, f"sin_{kind}")).to(h.dtype))
        return self.body(h, masks, trig)


class EncoderStatic(EncoderExport):
    """Encoder for one fixed length L: input x = raw token embeddings [1, L, E], padding mask kmask [1, 1, 1, L]."""

    def __init__(self, m: Translator, L: int):
        super().__init__(m, L)
        pos = torch.arange(L, dtype=torch.float32)
        self.register_buffer("far", torch.where((pos[:, None] - pos[None, :]).abs() > self.c.sliding_window, NEG, 0.0)
                             [None, None], persistent=False)

    def forward(self, x, kmask):
        h = self.ln(x, self.emb_norm)
        masks = {"full_attention": kmask, "sliding_attention": kmask + self.far}
        trig = {k: (getattr(self, f"cos_{k}"), getattr(self, f"sin_{k}")) for k in masks}
        return self.body(h, masks, trig)


class DecoderStep(nn.Module):
    """One decoder step: x [1, 1, d], self caches [1, H, T, hd] with additive smask [1, 1, 1, T], cross K/V [1, H, M, hd]
    with additive cmask [1, 1, 1, M] -> logits [1, 1, V], k0, v0, k1, v1 [1, H, 1, hd] (written into the caches at t)."""

    def __init__(self, m: Translator):
        super().__init__()
        self.layers, self.norm = m.layers, m.norm
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
