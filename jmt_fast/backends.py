"""Inference backends. Every backend translates one speech bubble at a time (batch size 1) with greedy decoding and
the kana byte-sequence ban; translate(text) -> str.

  TorchEager   PyTorch fp32, CPU / CUDA / MPS; the reference implementation (source up to 256 tokens)
  CudaGraphs   PyTorch fp16 on NVIDIA GPUs: encoder per length bucket (32 / 64 / 128) and one decoder step captured
               as CUDA graphs; kana rules, argmax and the cache update run inside the step graph
  OnnxCPU      ONNX Runtime fp32 on CPU, encoder at the exact source length; no PyTorch needed
  CoreML       Core ML fp16 on the Apple Neural Engine (encoder buckets 32 / 64 / 128); no PyTorch needed
The fixed-shape backends (CudaGraphs, OnnxCPU, CoreML, MLX) keep the first 128 source tokens and at most 128 output
tokens; the self-attention cache of the exported decoder holds 128 positions.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .common import NEG, Release, StepLoop, Text


class TorchEager:
    def __init__(self, root, device: str | None = None):
        import torch
        from . import torch_model
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.rel, self.device = Release(root), device
        self.text = Text(self.rel)
        self.m = torch_model.load(self.rel, device)
        self.rules = torch_model.kana_rules(self.rel, self.m.emb.weight.shape[0], device)

    def translate(self, text: str) -> str:
        torch, m, rel = self.torch, self.m, self.rel
        ids = self.text.encode(text)
        n = len(ids)
        with torch.inference_mode():
            src = torch.tensor([ids], device=self.device)
            mems, cmask = m.memories(src, torch.ones_like(src, dtype=torch.bool))
            cross = [layer.cross_kv(mem) for layer, mem in zip(m.layers, mems)]
            steps = min(3 * n + 10, rel.cfg["max_tgt"])
            caches = [torch.zeros(2, 1, layer.h, steps, layer.hd, device=self.device) for layer in m.layers]
            tok = torch.tensor([[rel.bos]], device=self.device)
            prev1 = prev2 = -1
            out = []
            for t in range(steps):
                x = m.embed(tok, t)
                for layer, cache, (ck, cv) in zip(m.layers, caches, cross):
                    x = layer.step(x, t, cache, ck, cv, cmask)
                logits = (m.norm(x) @ m.emb.weight.T)[:, -1]
                for p2, p1, banned in self.rules:
                    if prev1 == p1 and (p2 is None or prev2 == p2):
                        logits = logits.masked_fill(banned[None], float("-inf"))
                nxt = int(logits.argmax(-1))
                if nxt in (rel.eos, rel.pad):
                    break
                out.append(nxt)
                prev2, prev1 = prev1, nxt
                tok = torch.tensor([[nxt]], device=self.device)
        return self.text.decode(out)


class CudaGraphs:
    def __init__(self, root, buckets=(32, 64, 128), T: int = 128, fp16: bool = True):
        import torch
        from . import torch_model as tm
        self.torch = torch
        dev, dt = "cuda", torch.float16 if fp16 else torch.float32
        self.rel = rel = Release(root)
        self.text = Text(rel)
        m = tm.load(rel, "cpu")
        tm.apply_norm_scales(m, rel)                     # powers of two: no effect in fp32, required in fp16
        self.buckets, self.T, self.M, self.dt = buckets, T, 2 + max(buckets), dt
        self.encs = {L: tm.EncoderStatic(m, L).to(dev, dt).eval() for L in buckets}
        self.dec = tm.DecoderStep(m).to(dev, dt).eval()
        self.tok_emb = m.encoder.embeddings.tok_embeddings.to(dev, dt)
        self.emb_s = (m.emb.weight.detach() * math.sqrt(m.d)).to(dev, dt)
        self.pos = m.pos.weight.detach().to(dev, dt)
        self.rules = tm.kana_rules(rel, self.emb_s.shape[0], dev)
        H, hd = m.layers[0].h, m.layers[0].hd
        z = lambda *s: torch.zeros(*s, device=dev, dtype=dt)  # noqa: E731
        self.cross = [z(1, H, self.M, hd) for _ in range(4)]
        self.cmask = torch.full((1, 1, 1, self.M), NEG, device=dev, dtype=dt)
        self.caches = [z(1, H, T, hd) for _ in range(4)]
        self.smask = torch.full((1, 1, 1, T), NEG, device=dev, dtype=dt)
        self.ids = {L: torch.full((1, L), rel.cfg["src_pad"], dtype=torch.long, device=dev) for L in buckets}
        self.kmask = {L: torch.zeros((1, 1, 1, L), device=dev, dtype=dt) for L in buckets}
        lt = lambda v: torch.full((1,), v, dtype=torch.long, device=dev)  # noqa: E731
        self.t, self.cur, self.prev1, self.prev2, self.nxt = lt(0), lt(rel.bos), lt(-1), lt(-1), lt(0)
        self.g_enc = {L: self._capture(lambda L=L: self._enc(L)) for L in buckets}
        self._reset()
        self.g_dec = self._capture(self._dec_step)

    def _enc(self, L):
        with self.torch.no_grad():
            outs = self.encs[L](self.tok_emb(self.ids[L]), self.kmask[L])
            for buf, x in zip(self.cross, outs):
                buf[:, :, :2 + L].copy_(x)

    def _dec_step(self):
        with self.torch.no_grad():
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
        torch = self.torch
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
        self.smask.fill_(NEG)
        self.t.zero_()
        self.cur.fill_(self.rel.bos)
        self.prev1.fill_(-1)
        self.prev2.fill_(-1)

    def translate(self, text: str) -> str:
        torch, rel = self.torch, self.rel
        ids = self.text.encode(text)
        n = len(ids)
        L = next((b for b in self.buckets if b >= n), self.buckets[-1])
        ids, n = ids[:L], min(n, L)
        a = torch.full((1, L), rel.cfg["src_pad"], dtype=torch.long)
        a[0, :n] = torch.tensor(ids)
        km = torch.zeros((1, 1, 1, L), dtype=self.dt)
        km[..., n:] = NEG
        cm = torch.full((1, 1, 1, self.M), NEG, dtype=self.dt)
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
            if v in (rel.eos, rel.pad):
                break
            out.append(v)
        return self.text.decode(out)


class OnnxCPU:
    def __init__(self, root, threads: int | None = None, spin: bool = False):
        import onnxruntime as ort
        self.rel = rel = Release(root)
        self.text = Text(rel)
        d = rel.root / "onnx"
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = threads or min(6, os.cpu_count() or 1)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.add_session_config_entry("session.intra_op.allow_spinning", "1" if spin else "0")
        mk = lambda f: ort.InferenceSession(str(d / f), so, providers=["CPUExecutionProvider"])  # noqa: E731
        self.enc, self.dec = mk("encoder_fp32.onnx"), mk("decoder_fp32.onnx")
        z = np.load(d / "host_fp32.npz")
        emb, pos, H = z["emb"], z["pos"], int(z["heads"])
        self.Lmax, T = 128, 128
        names = [o.name for o in self.dec.get_outputs()]
        step = lambda feed: dict(zip(names, self.dec.run(None, feed)))  # noqa: E731
        self.loop = StepLoop(rel, emb * math.sqrt(emb.shape[1]), pos, H, emb.shape[1] // H, T, 2 + self.Lmax, step)

    def translate(self, text: str) -> str:
        ids = self.text.encode(text)[:self.Lmax]
        r = self.enc.run(None, {"ids": np.asarray([ids], dtype=np.int64)})
        cross = dict(zip(("ck0", "cv0", "ck1", "cv1"), r))
        return self.text.decode(self.loop(cross, len(ids)))


class CoreML:
    def __init__(self, root, compute_units: str = "CPU_AND_NE"):
        import json
        import sys
        sys.modules.setdefault("tensorflow", None)   # coremltools imports TF if present; TF deadlocks next to tokenizers
        import coremltools as ct
        self.rel = rel = Release(root)
        self.text = Text(rel)
        d = rel.root / "coreml"
        h = json.loads((d / "host/host.json").read_text())
        self.enc_emb = np.load(d / "host/enc_emb.npy", mmap_mode="r")
        dec_emb = np.load(d / "host/dec_emb.npy").astype(np.float32)             # already scaled by sqrt(d)
        pos = np.load(d / "host/pos.npy").astype(np.float32)
        cu = getattr(ct.ComputeUnit, compute_units)
        self.buckets = h["buckets"]
        self.enc = {b: ct.models.MLModel(str(d / "encoder_static_fp16.mlpackage"), function_name=f"L{b}", compute_units=cu)
                    for b in self.buckets}
        dec = ct.models.MLModel(str(d / "decoder_fp16.mlpackage"), compute_units=cu)
        self.loop = StepLoop(rel, dec_emb, pos, h["heads"], h["head_dim"], h["T"], 2 + max(self.buckets), dec.predict)

    def translate(self, text: str) -> str:
        ids = self.text.encode(text)
        n = len(ids)
        L = next((b for b in self.buckets if b >= n), self.buckets[-1])
        ids, n = ids[:L], min(n, L)
        arr = np.full(L, self.rel.cfg["src_pad"], dtype=np.int64)
        arr[:n] = ids
        km = np.full((1, 1, 1, L), NEG, np.float32)
        km[..., :n] = 0.0
        r = self.enc[L].predict({"x": np.asarray(self.enc_emb[arr], dtype=np.float32)[None], "kmask": km})
        return self.text.decode(self.loop({k: r[k] for k in ("ck0", "cv0", "ck1", "cv1")}, n))
