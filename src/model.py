"""ModernBERT-ja encoder + shallow autoregressive decoder for ja->zh block translation.

encoder   HF ModernBERT (bidirectional), all hidden states returned.
memory    base = Bridge(last_hidden_state); decoder layer j reads
          memory_j = [null x2 ; base + gamma_j * Wo(sum_l softmax(w_j)_l RMSNorm_l(h_l))]   over h_0..h_{L-1}
          (static depth fusion of mangaOCR-NAR; gamma starts at 0 so memory_j == base at init).
bridge    "swiglu": RMSNorm -> SwiGLU(E -> hidden -> d) -> RMSNorm
          "mlp":    RMSNorm -> Linear(E -> hidden) -> GELU -> Linear(hidden -> d) -> RMSNorm   (OCR bridge)
          "linear": RMSNorm -> Linear(E -> d) -> RMSNorm
decoder   pre-RMSNorm blocks: causal self-attention, cross-attention to memory_j, SwiGLU; learned positions;
          output tied to the (compact) target embedding table. Inputs are scaled by sqrt(d) as in the DAT,
          so DAT embedding rows can initialise the table.
context   (cfg ctx_max_dist > 0) every block is encoded once (encode_blocks); a sample's memory is
          [null x2 ; current block ; earlier blocks] gathered from the encoded blocks (assemble), each position plus
          dist[k] (k = 0 current block, k = 1.. blocks back, zero-initialised). The decoder input is
          <ctx_zh> zh(earlier blocks, <sep>-joined) <bos> target; the loss covers the target only.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


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


def make_bridge(kind: str, e: int, d: int, hidden: int) -> nn.Module:
    if kind == "swiglu":
        return nn.Sequential(RMSNorm(e), SwiGLU(e, hidden, d), RMSNorm(d))
    if kind == "mlp":
        return nn.Sequential(RMSNorm(e), nn.Linear(e, hidden, bias=False), nn.GELU(), nn.Linear(hidden, d, bias=False),
                             RMSNorm(d))
    if kind == "linear":
        return nn.Sequential(RMSNorm(e), nn.Linear(e, d, bias=False), RMSNorm(d))
    raise ValueError(kind)


class Fusion(nn.Module):
    def __init__(self, e: int, d: int, dec_layers: int, depths: int):
        super().__init__()
        self.norms = nn.ModuleList(RMSNorm(e) for _ in range(depths))
        self.wo = nn.Linear(e, d, bias=False)                 # random init so gamma receives gradient
        self.gamma = nn.Parameter(torch.zeros(dec_layers, d))
        self.logits = nn.Parameter(torch.zeros(dec_layers, depths))

    def forward(self, states):
        h = torch.stack([n(s) for n, s in zip(self.norms, states)], 0)            # [D,B,N,E]
        f = torch.einsum("jl,lbne->jbne", self.logits.softmax(-1).to(h.dtype), h)
        return self.gamma[:, None, None].to(h.dtype) * self.wo(f)                  # [J,B,N,d]


class DecoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        self.h, self.hd, self.dropout = heads, d // heads, dropout
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

    def drop(self, y):
        return F.dropout(y, self.dropout, self.training)

    def cross_kv(self, mem):
        k, v = self.kv2(mem).chunk(2, -1)
        return self.heads(k), self.heads(v)

    def forward(self, x, ck, cv, cmask):
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        x = x + self.drop(self.o1(self.merge(F.scaled_dot_product_attention(
            self.heads(q), self.heads(k), self.heads(v), is_causal=True))))
        x = x + self.drop(self.o2(self.merge(F.scaled_dot_product_attention(
            self.heads(self.q2(self.n2(x))), ck, cv, attn_mask=cmask))))
        return x + self.drop(self.ffn(self.n3(x)))

    def step(self, x, t, cache, ck, cv, cmask):
        """One position for every row; cache [2,B,H,T,hd] holds self-attention K/V of earlier positions."""
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        cache[0, :, :, t] = self.heads(k)[:, :, 0]
        cache[1, :, :, t] = self.heads(v)[:, :, 0]
        x = x + self.o1(self.merge(F.scaled_dot_product_attention(
            self.heads(q), cache[0, :, :, : t + 1], cache[1, :, :, : t + 1])))
        x = x + self.o2(self.merge(F.scaled_dot_product_attention(
            self.heads(self.q2(self.n2(x))), ck, cv, attn_mask=cmask)))
        return x + self.ffn(self.n3(x))


class ARMT(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.cfg = cfg
        enc_cfg = AutoConfig.from_pretrained(cfg["encoder"])
        self.encoder = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa", dtype=torch.float32) \
            if cfg.get("load_encoder_weights", True) else AutoModel.from_config(enc_cfg, attn_implementation="sdpa")
        e = enc_cfg.hidden_size
        d = cfg.get("width") or e
        self.d, L = d, cfg["dec_layers"]
        self.bridge = make_bridge(cfg["bridge"], e, d, cfg["bridge_hidden"])
        self.fusion = Fusion(e, d, L, enc_cfg.num_hidden_layers) if cfg["fusion"] else None
        self.null = nn.Parameter(torch.randn(cfg["null_tokens"], d) * 0.02) if cfg["null_tokens"] else None
        self.emb = nn.Embedding(cfg["vocab"], d)
        self.pos = nn.Embedding(cfg["max_tgt"], d)
        self.layers = nn.ModuleList(DecoderLayer(d, d // 64, cfg["ffn"], cfg["dropout"]) for _ in range(L))
        self.norm = RMSNorm(d)
        self.dist = nn.Embedding(cfg["ctx_max_dist"] + 1, d) if cfg.get("ctx_max_dist") else None
        nn.init.normal_(self.emb.weight, std=d ** -0.5)
        nn.init.normal_(self.pos.weight, std=0.02)
        if self.dist is not None:
            nn.init.zeros_(self.dist.weight)
        for lin in [m for m in self.modules() if isinstance(m, nn.Linear) and not self._in_encoder(m)]:
            nn.init.normal_(lin.weight, std=0.02)
        for layer in self.layers:                                          # depth-scaled residual outputs
            for lin in (layer.o1, layer.o2, layer.ffn.down):
                nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * L))

    def _in_encoder(self, module) -> bool:
        return any(module is m for m in self.encoder.modules())

    # ------------------------------------------------------------------ init helpers
    @torch.no_grad()
    def init_embeddings(self, table: torch.Tensor, rows: list[int]) -> int:
        """Copy DAT embedding rows (compact index i <- DAT id rows[i]; rows[i] < 0 keeps the random init)."""
        n = 0
        for i, r in enumerate(rows):
            if r >= 0:
                self.emb.weight[i] = table[r].to(self.emb.weight.dtype)
                n += 1
        return n

    def decoder_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    # ------------------------------------------------------------------ forward pieces
    def memories(self, src, src_mask, encoder_grad: bool = True):
        with torch.set_grad_enabled(encoder_grad and torch.is_grad_enabled()):
            out = self.encoder(input_ids=src, attention_mask=src_mask.long(), output_hidden_states=self.fusion is not None)
        base = self.bridge(out.last_hidden_state)
        mems = base[None].expand(len(self.layers), -1, -1, -1)
        if self.fusion is not None:
            mems = mems + self.fusion(list(out.hidden_states[:-1]))
        if self.dist is not None:                         # single-block path = current block (distance 0)
            mems = mems + self.dist.weight[0].to(mems.dtype)
        mask = src_mask
        if self.null is not None:
            b = src.shape[0]
            mems = torch.cat((self.null[None, None].expand(len(self.layers), b, -1, -1).to(mems.dtype), mems), 2)
            mask = torch.cat((torch.ones(b, self.null.shape[0], dtype=torch.bool, device=src.device), src_mask), 1)
        return mems, mask[:, None, None, :]

    def _encode(self, src, src_mask, encoder_grad: bool):
        with torch.set_grad_enabled(encoder_grad and torch.is_grad_enabled()):
            out = self.encoder(input_ids=src, attention_mask=src_mask.long(), output_hidden_states=self.fusion is not None)
        base = self.bridge(out.last_hidden_state)
        mems = base[None].expand(len(self.layers), -1, -1, -1)
        if self.fusion is not None:
            mems = mems + self.fusion(list(out.hidden_states[:-1]))
        return mems

    def encode_blocks(self, src, src_mask, encoder_grad: bool = True, groups: int = 1):
        """Encode every block once and keep only real tokens: returns (packed [J, N, d], start [U], L) where block u
        occupies packed[:, start[u] : start[u] + len_u]. groups > 1 encodes length-sorted groups, each padded only to
        its own longest block."""
        U, L = src.shape
        lens = src_mask.sum(1)
        order = torch.argsort(lens) if groups > 1 and U >= 2 * groups else torch.arange(U, device=src.device)
        parts = torch.tensor_split(order, groups) if groups > 1 and U >= 2 * groups else [order]
        chunks, start = [], torch.zeros(U, dtype=torch.long, device=src.device)
        off = 0
        for part in parts:
            if part.numel() == 0:
                continue
            lg = int(lens[part].max())
            m = self._encode(src[part, :lg], src_mask[part, :lg], encoder_grad)            # [J, g, lg, d]
            pm = src_mask[part, :lg]
            chunks.append(m[:, pm])                                                       # row-major: block by block
            pl = lens[part]
            start[part] = off + torch.cumsum(pl, 0) - pl
            off += int(pl.sum())
        return torch.cat(chunks, 1), start, L

    def assemble(self, blocks, idx, dist, valid):
        """Gather sample memories from encoded blocks (encode_blocks output).
        idx [B, M] = u * L + p (block u, position p; pad entries arbitrary, masked by valid), dist [B, M] distance ids,
        valid [B, M] bool. Returns mems [J, B, n_null + M, d] and mask [B, 1, 1, n_null + M]."""
        packed, start, L = blocks
        J, d = packed.shape[0], packed.shape[-1]
        flat = torch.where(valid, start[idx // L] + idx % L, torch.zeros_like(idx))
        mems = packed[:, flat]                                                           # [J, B, M, d]
        if self.dist is not None:
            mems = mems + self.dist(dist)[None].to(mems.dtype)
        mask = valid
        if self.null is not None:
            b = idx.shape[0]
            mems = torch.cat((self.null[None, None].expand(J, b, -1, -1).to(mems.dtype), mems), 2)  # noqa: E501
            mask = torch.cat((torch.ones(b, self.null.shape[0], dtype=torch.bool, device=idx.device), valid), 1)
        return mems, mask[:, None, None, :]

    def decode_logits(self, mems, cmask, tgt_in):
        x = F.dropout(self.embed(tgt_in), self.cfg["dropout"], self.training)
        for layer, mem in zip(self.layers, mems):
            ck, cv = layer.cross_kv(mem)
            x = layer(x, ck, cv, cmask)
        return self.norm(x) @ self.emb.weight.T

    def loss_ctx(self, src, src_mask, idx, dist, valid, tgt_in, gold, pad: int, smoothing: float,
                 encoder_grad: bool = True, groups: int = 1):
        """gold = next-token targets with prefix positions set to pad (ignored). Logits are computed only at target
        positions. groups > 1: blocks are encoded in length groups and samples are decoded in length groups, each
        padded to its own longest decoder input / memory."""
        blocks = self.encode_blocks(src, src_mask, encoder_grad, groups)
        tlen = tgt_in.ne(pad).sum(1)
        order = torch.argsort(tlen) if groups > 1 else torch.arange(tgt_in.shape[0], device=tgt_in.device)
        tot_s = tot_p = 0.0
        n_all = gold.ne(pad).sum()
        for part in (torch.tensor_split(order, groups) if groups > 1 else [order]):
            if part.numel() == 0:
                continue
            T = int(tlen[part].max())
            M = int(valid[part].sum(1).max())
            mems, cmask = self.assemble(blocks, idx[part, :M], dist[part, :M], valid[part, :M])
            x = F.dropout(self.embed(tgt_in[part, :T]), self.cfg["dropout"], self.training)
            for layer, mem in zip(self.layers, mems):
                ck, cv = layer.cross_kv(mem)
                x = layer(x, ck, cv, cmask)
            g = gold[part, :T]
            ok = g.ne(pad)
            logits = (self.norm(x[ok]) @ self.emb.weight.T).float()
            tot_s = tot_s + F.cross_entropy(logits, g[ok], reduction="sum", label_smoothing=smoothing)
            tot_p = tot_p + F.cross_entropy(logits.detach(), g[ok], reduction="sum")
        return tot_s / n_all, tot_p / n_all, n_all

    @torch.no_grad()
    def generate_ctx(self, mems, cmask, prefixes, bos: int, eos: int, pad: int, max_len: int, byte_compact=None):
        """Greedy decoding after a per-row forced prefix (list of id lists ending before <bos>; [] = no context).
        Rows keep absolute positions from 0, so each row reads prefix, <bos>, then its own output."""
        b = len(prefixes)
        dev = cmask.device
        rules = self.kana_byte_rules(byte_compact, self.emb.weight.shape[0], dev) if byte_compact else []
        lp = torch.tensor([len(p) for p in prefixes], device=dev)
        P = int(lp.max()) if b else 0
        forced = torch.full((b, P + 1), bos, dtype=torch.long, device=dev)
        for r, p in enumerate(prefixes):
            if p:
                forced[r, :len(p)] = torch.tensor(p, device=dev)
        cross = [layer.cross_kv(mem) for layer, mem in zip(self.layers, mems)]
        steps = min(P + max_len, self.cfg["max_tgt"])                       # outputs start at t = len(prefix)
        caches = [torch.zeros(2, b, layer.h, steps, layer.hd, device=dev, dtype=mems.dtype) for layer in self.layers]
        prev1 = torch.full((b,), -1, dtype=torch.long, device=dev)
        prev2 = prev1.clone()
        done = torch.zeros(b, dtype=torch.bool, device=dev)
        tok = forced[:, :1]
        out = []
        for t in range(steps):
            x = self.embed(tok, t)
            for layer, cache, (ck, cv) in zip(self.layers, caches, cross):
                x = layer.step(x, t, cache, ck, cv, cmask)
            gen = t >= lp                                                  # this step's output is a target token
            logits = (self.norm(x) @ self.emb.weight.T)[:, -1]
            for p2, p1, m in rules:
                hit = prev1.eq(p1) if p2 is None else prev1.eq(p1) & prev2.eq(p2)
                logits = logits.masked_fill((hit & gen)[:, None] & m[None], float("-inf"))
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
            out.append(torch.where(gen, nxt, torch.full_like(nxt, -1)))
            done |= gen & nxt.eq(eos)
            if bool(done.all()) or t + 1 >= steps:
                break
            nf = forced[:, min(t + 1, P)]
            feed = torch.where(t + 1 <= lp, nf, nxt)                       # still inside prefix (incl. <bos>) -> forced
            prev2, prev1 = prev1, torch.where(gen, nxt, prev1)
            tok = feed[:, None]
        seqs = torch.stack(out, 1).tolist() if out else [[] for _ in range(b)]
        res = []
        for s in seqs:
            r = []
            for x in s:
                if x == -1:
                    continue
                if x in (eos, pad):
                    break
                r.append(x)
            res.append(r)
        return res

    def embed(self, ids, start: int = 0):
        pos = torch.arange(start, start + ids.shape[1], device=ids.device)
        return self.emb(ids) * math.sqrt(self.d) + self.pos(pos)[None]

    def forward(self, src, src_mask, tgt_in, encoder_grad: bool = True):
        mems, cmask = self.memories(src, src_mask, encoder_grad)
        x = F.dropout(self.embed(tgt_in), self.cfg["dropout"], self.training)
        for layer, mem in zip(self.layers, mems):
            ck, cv = layer.cross_kv(mem)
            x = layer(x, ck, cv, cmask)
        return self.norm(x) @ self.emb.weight.T

    def loss(self, src, src_mask, tgt, pad: int, smoothing: float, encoder_grad: bool = True):
        logits = self(src, src_mask, tgt[:, :-1], encoder_grad).float()
        gold = tgt[:, 1:]
        valid = gold.ne(pad)
        nll = F.cross_entropy(logits.flatten(0, 1), gold.flatten(), reduction="none", label_smoothing=smoothing)
        plain = F.cross_entropy(logits.flatten(0, 1).detach(), gold.flatten(), reduction="none")
        n = valid.sum()
        return (nll * valid.flatten()).sum() / n, (plain * valid.flatten()).sum() / n, n

    @staticmethod
    def kana_byte_rules(byte_compact: list[int], vocab: int, device):
        """Masks that stop byte-fallback pieces from spelling kana in UTF-8:
        E3 81|82|83 xx (U+3040-30FF), E3 87 B0-BF (U+31F0-31FF), EF BD A6-BF and EF BE 80-9D (half-width)."""
        B = torch.tensor(byte_compact, device=device)

        def mask(byte_values):
            m = torch.zeros(vocab, dtype=torch.bool, device=device)
            m[B[list(byte_values)]] = True
            return m
        return [(None, int(B[0xE3]), mask([0x81, 0x82, 0x83])),
                (int(B[0xE3]), int(B[0x87]), mask(range(0xB0, 0xC0))),
                (int(B[0xEF]), int(B[0xBD]), mask(range(0xA6, 0xC0))),
                (int(B[0xEF]), int(B[0xBE]), mask(range(0x80, 0x9E)))]

    @torch.no_grad()
    def generate(self, src, src_mask, bos: int, eos: int, pad: int, max_len: int, byte_compact=None):
        """Batched greedy decoding with KV cache; returns compact id lists without BOS/EOS.
        byte_compact (compact ids of <0x00>..<0xFF>) enables the kana byte-sequence ban."""
        b = src.shape[0]
        rules = self.kana_byte_rules(byte_compact, self.emb.weight.shape[0], src.device) if byte_compact else []
        prev1 = torch.full((b,), -1, dtype=torch.long, device=src.device)
        prev2 = prev1.clone()
        mems, cmask = self.memories(src, src_mask, encoder_grad=False)
        cross = [layer.cross_kv(mem) for layer, mem in zip(self.layers, mems)]
        steps = min(max_len, self.cfg["max_tgt"] - 1)
        caches = [torch.zeros(2, b, layer.h, steps, layer.hd, device=src.device, dtype=mems.dtype) for layer in self.layers]
        tok = torch.full((b, 1), bos, dtype=torch.long, device=src.device)
        done = torch.zeros(b, dtype=torch.bool, device=src.device)
        out = []
        for t in range(steps):
            x = self.embed(tok, t)
            for layer, cache, (ck, cv) in zip(self.layers, caches, cross):
                x = layer.step(x, t, cache, ck, cv, cmask)
            logits = (self.norm(x) @ self.emb.weight.T)[:, -1]
            for p2, p1, m in rules:
                hit = prev1.eq(p1) if p2 is None else prev1.eq(p1) & prev2.eq(p2)
                logits = logits.masked_fill(hit[:, None] & m[None], float("-inf"))
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
            prev2, prev1 = prev1, nxt
            out.append(nxt)
            done |= nxt.eq(eos)
            if bool(done.all()):
                break
            tok = nxt[:, None]
        seqs = torch.stack(out, 1).tolist() if out else [[] for _ in range(b)]
        res = []
        for s in seqs:
            r = []
            for x in s:
                if x in (eos, pad):
                    break
                r.append(x)
            res.append(r)
        return res
