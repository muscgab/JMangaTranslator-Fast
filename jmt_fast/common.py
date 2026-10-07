"""Host-side pieces shared by every backend: the release directory, Japanese tokenization, Chinese detokenization,
the kana byte-sequence ban and the numpy greedy loop used by the ONNX Runtime and Core ML backends.

Release directory layout
  config.json          decoder hyperparameters and special ids
  model.safetensors    all weights, fp32 (PyTorch / CUDA Graphs / MLX backends)
  tokenizer/           Japanese tokenizer (ModernBERT-ja with added characters) and the encoder architecture
  joint.model          Chinese SentencePiece model
  vocab.json           decoder vocabulary: compact id -> SentencePiece id (dat_ids), byte pieces
  norm_scales.json     per-normalization power-of-two scales for the fp16 CUDA Graphs backend
  onnx/                encoder_fp32.onnx, decoder_fp32.onnx, host_fp32.npz
  coreml/              encoder_static_fp16.mlpackage, decoder_fp16.mlpackage, host/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NEG = -1e4                    # additive mask value used by every exported graph


class Release:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.cfg = json.loads((self.root / "config.json").read_text())
        vocab = json.loads((self.root / "vocab.json").read_text())
        self.dat_ids: list[int] = vocab["dat_ids"]
        lut = np.zeros(vocab["dat_vocab"], dtype=np.int64)
        for i, d in enumerate(self.dat_ids):
            if d >= 0:
                lut[d] = i
        self.byte_compact = [int(lut[x]) for x in vocab["byte_piece_ids"]]   # compact ids of <0x00> .. <0xFF>
        self.bos, self.eos, self.pad = self.cfg["bos"], self.cfg["eos"], self.cfg["pad"]


class Text:
    """Japanese source -> encoder token ids ([CLS] ... [SEP], at most max_src_tokens), compact output ids -> Chinese."""

    def __init__(self, rel: Release):
        import sentencepiece as spm
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(rel.root / "tokenizer/tokenizer.json"))
        self.tok.no_padding()
        self.tok.enable_truncation(rel.cfg["max_src_tokens"])
        self.sp = spm.SentencePieceProcessor(model_file=str(rel.root / "joint.model"))
        self.dat_ids = rel.dat_ids

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode([int(self.dat_ids[i]) for i in ids if self.dat_ids[i] >= 4])


def kana_rules(byte_compact: list[int], vocab: int) -> list[tuple[int | None, int, np.ndarray]]:
    """(prev2, prev1, banned) triples that stop byte-fallback pieces from spelling kana in UTF-8:
    E3 81|82|83 xx (U+3040-30FF), E3 87 B0-BF (U+31F0-31FF), EF BD A6-BF and EF BE 80-9D (half-width).
    After the byte pieces prev2 (None = any), prev1, the next piece must not be in banned."""
    B = byte_compact

    def mask(values) -> np.ndarray:
        m = np.zeros(vocab, dtype=bool)
        m[[B[v] for v in values]] = True
        return m
    return [(None, B[0xE3], mask([0x81, 0x82, 0x83])), (B[0xE3], B[0x87], mask(range(0xB0, 0xC0))),
            (B[0xEF], B[0xBD], mask(range(0xA6, 0xC0))), (B[0xEF], B[0xBE], mask(range(0x80, 0x9E)))]


class StepLoop:
    """Greedy decoding around an exported one-step decoder (ONNX / Core ML): self-attention caches [1, H, T, hd]
    written at position t, additive masks smask [1, 1, 1, T] and cmask [1, 1, 1, M], cross K/V padded to M.
    step(feed) takes and returns dicts with the graph's input / output names."""

    def __init__(self, rel: Release, emb_scaled: np.ndarray, pos: np.ndarray, heads: int, head_dim: int,
                 T: int, M: int, step):
        self.rel, self.emb, self.pos = rel, emb_scaled, pos
        self.H, self.hd, self.T, self.M, self.step = heads, head_dim, T, M, step
        self.rules = kana_rules(rel.byte_compact, emb_scaled.shape[0])

    def __call__(self, cross: dict[str, np.ndarray], n: int) -> list[int]:
        """cross: ck0, cv0, ck1, cv1 with 2 + L positions (2 null tokens + L source positions, n of them real)."""
        cr = {}
        for k, c in cross.items():
            z = np.zeros((1, self.H, self.M, self.hd), dtype=np.float32)
            z[:, :, :2 + n] = c[:, :, :2 + n]
            cr[k] = z
        cmask = np.full((1, 1, 1, self.M), NEG, dtype=np.float32)
        cmask[..., :2 + n] = 0.0
        caches = {k: np.zeros((1, self.H, self.T, self.hd), dtype=np.float32) for k in ("kc0", "vc0", "kc1", "vc1")}
        smask = np.full((1, 1, 1, self.T), NEG, dtype=np.float32)
        prev1 = prev2 = -1
        tok, out = self.rel.bos, []
        for t in range(min(3 * n + 10, self.rel.cfg["max_tgt"], self.T + 1)):
            x = (self.emb[tok] + self.pos[t])[None, None].astype(np.float32)
            o = self.step({"x": x, **caches, "smask": smask, **cr, "cmask": cmask})
            lg = np.asarray(o["logits"], dtype=np.float32).reshape(-1)
            for p2, p1, banned in self.rules:
                if prev1 == p1 and (p2 is None or prev2 == p2):
                    lg[banned] = -np.inf
            nxt = int(lg.argmax())
            if nxt in (self.rel.eos, self.rel.pad):
                break
            out.append(nxt)
            if t < self.T:
                for kc, kn in (("kc0", "k0"), ("vc0", "v0"), ("kc1", "k1"), ("vc1", "v1")):
                    caches[kc][:, :, t] = np.asarray(o[kn])[:, :, 0]
                smask[..., t] = 0.0
            prev2, prev1, tok = prev1, nxt, nxt
        return out
