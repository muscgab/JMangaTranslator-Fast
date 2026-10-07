#!/usr/bin/env python3
"""Our single-block model (R2 checkpoint) -- or, with --system nano, NanoSakura-2.2 -- on the comparison tasks, on CUDA fp32 (Titan Xp has no bf16).

Every translation is produced alone (batch=1) and timed (ms: tokenization -> generation -> CUDA sync -> detokenization),
so each output row carries its own speed. Tasks:
  m109         Manga109 boxes, inputs clean / nar / mocr          -> hyp/m109.<tag>.<input>.jsonl
  murasaki     every split_ja segment alone, joined per paragraph -> hyp/murasaki.<tag>.seg.jsonl (seg_ms per segment)
  luna         standard sets dev / heldout / luna1k / manga200    -> hyp/luna_<set>.<tag>.single.jsonl
  throughput   Manga109 clean in length-sorted batches of --bsz   -> throughput.<tag>.json (+ agreement with batch=1)
The first --warm clean boxes are translated once before timing (CUDA / allocator warm-up).
Same generation path as benchmarks/manga109_ocr/eval_m109.py (assemble + generate_ctx with empty prefixes).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LUNA_SETS, OUT, ROOT, load_luna, load_m109, load_murasaki, write_jsonl  # noqa: E402

sys.path.insert(0, str(ROOT / "ar_mt"))
from model import ARMT  # noqa: E402
from train import BOS, DAT_TOK, EOS, PAD, decode_ids  # noqa: E402


class Ours:
    def __init__(self, ckpt: Path, encoder: str, data: Path, device: str):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.step = ck.get("step")
        self.model = ARMT(dict(ck["cfg"], load_encoder_weights=False, encoder=encoder))
        self.model.load_state_dict(ck["model"])
        self.model.to(device).eval()
        self.device = device
        vocab = json.loads((data / "vocab.json").read_text())
        self.dat_ids = vocab["dat_ids"]
        lut = np.zeros(vocab["dat_vocab"], dtype=np.int64)
        for i, d in enumerate(self.dat_ids):
            if d >= 0:
                lut[d] = i
        self.byte_compact = [int(lut[x]) for x in vocab["byte_piece_ids"]] if "byte_piece_ids" in vocab else None
        self.sp = spm.SentencePieceProcessor(model_file=str(DAT_TOK))
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(encoder)

    @torch.no_grad()
    def translate(self, texts: list[str]) -> list[str]:
        ids = [self.tok(x, add_special_tokens=True, truncation=True, max_length=256)["input_ids"] for x in texts]
        L = max(map(len, ids))
        s = torch.full((len(ids), L), 3, dtype=torch.long)
        for u, x in enumerate(ids):
            s[u, :len(x)] = torch.tensor(x)
        m = torch.arange(L)[None] < torch.tensor([len(x) for x in ids])[:, None]
        idx = torch.arange(len(ids))[:, None] * L + torch.arange(L)[None]
        s, m, idx = s.to(self.device), m.to(self.device), idx.to(self.device)
        mems, cmask = self.model.assemble(self.model.encode_blocks(s, m, False), idx, torch.zeros_like(idx), m)
        out = self.model.generate_ctx(mems, cmask, [[] for _ in ids], BOS, EOS, PAD, max_len=3 * L + 10,
                                      byte_compact=self.byte_compact)
        return [decode_ids(o, self.dat_ids, self.sp) for o in out]

    def batched(self, texts: list[str], bsz: int) -> list[str]:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))      # length-sorted batches, original order back
        res = [""] * len(texts)
        for j in range(0, len(order), bsz):
            part = order[j:j + bsz]
            for i, h in zip(part, self.translate([texts[i] for i in part])):
                res[i] = h
        return res


class Nano:
    """NanoSakura-2.2-0.2B (telecomadm1145, MIT, revision 5f09a522), loaded exactly as benchmarks/manga200/
    evaluate_nanosakura.py: upstream modeling_mamba2_s2s.py, strict load with explicit tied-weight aliases;
    source + "<eos>", greedy, max_new_tokens 256, decoder starts with BOS. fp32."""

    def __init__(self, model_dir: Path, device: str):
        from safetensors.torch import load_file
        from transformers import PreTrainedTokenizerFast
        sys.path.insert(0, str(model_dir))
        from modeling_mamba2_s2s import Mamba2Seq2SeqConfig, Mamba2Seq2SeqForConditionalGeneration
        self.cfg = Mamba2Seq2SeqConfig(**json.loads((model_dir / "config.json").read_text()))
        self.model = Mamba2Seq2SeqForConditionalGeneration(self.cfg)
        self.model.tie_weights()
        state = load_file(str(model_dir / "model.safetensors"), device="cpu")
        aliases = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
        stored = [k for k in aliases if k in state]
        for k in aliases:
            state[k] = state[stored[0]]
        self.model.load_state_dict(state, strict=True)
        self.model.tie_weights()
        assert sum(p.numel() for p in self.model.parameters()) == 188099360
        self.model.eval().to(device=device, dtype=torch.float32)
        self.tok = PreTrainedTokenizerFast.from_pretrained(str(model_dir), local_files_only=True)
        self.device, self.step = device, "nanosakura-2.2@5f09a522"

    @torch.inference_mode()
    def translate(self, texts: list[str]) -> list[str]:
        enc = [self.tok.encode(x + "<eos>") for x in texts]
        c = self.cfg
        src = torch.full((len(enc), max(map(len, enc))), c.pad_token_id, dtype=torch.long, device=self.device)
        for j, e in enumerate(enc):
            src[j, :len(e)] = torch.tensor(e, device=self.device)
        y = self.model.generate(input_ids=src, attention_mask=src.ne(c.pad_token_id), max_new_tokens=256, do_sample=False,
                                num_beams=1, decoder_start_token_id=c.bos_token_id, eos_token_id=c.eos_token_id,
                                pad_token_id=c.pad_token_id, use_cache=True)
        return self.tok.batch_decode(y.cpu().tolist(), skip_special_tokens=True)

    batched = Ours.batched


def timed(m: "Ours", texts: list[str]) -> tuple[list[str], list[float]]:
    hyps, ms = [], []
    for x in texts:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hyps.append(m.translate([x])[0])
        torch.cuda.synchronize()
        ms.append(round(1000 * (time.perf_counter() - t0), 2))
    return hyps, ms


def stat(ms) -> dict:
    a = np.array(ms)
    return {"n": len(a), "p50_ms": round(float(np.percentile(a, 50)), 1), "p90_ms": round(float(np.percentile(a, 90)), 1),
            "mean_ms": round(float(a.mean()), 1), "total_s": round(float(a.sum()) / 1000, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=("ours", "nano"), default="ours")
    ap.add_argument("--ckpt", type=Path, help="ours: R2 checkpoint; nano: model directory")
    ap.add_argument("--encoder")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "artifacts/ar_mt_20261005/data_v2_ext")
    ap.add_argument("--tasks", default="m109,murasaki,luna,throughput")
    ap.add_argument("--bsz", type=int, default=64)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batched", type=int, default=0,
                    help="1 = accuracy only: every task in length-sorted batches of --bsz, no per-item timing")
    args = ap.parse_args()
    m = Ours(args.ckpt, args.encoder, args.data, "cuda") if args.system == "ours" else Nano(args.ckpt, "cuda")
    print(json.dumps({"ckpt": str(args.ckpt), "step": m.step, "gpu": torch.cuda.get_device_name(),
                      "torch": torch.__version__}), flush=True)
    tasks = args.tasks.split(",")
    recs = load_m109()[:args.limit or None]
    if args.batched:                                   # accuracy only (user 2026-10-06 21:4x): no timing
        run = lambda mm, texts: (mm.batched(texts, args.bsz), [None] * len(texts))  # noqa: E731
        summ = lambda ms: {"n": len(ms), "batched": args.bsz}                       # noqa: E731
    else:
        run, summ = timed, stat
        run(m, [r["clean"] for r in recs[:args.warm]])
    if "m109" in tasks:
        for inp in ("clean", "nar", "mocr"):
            hyps, ms = run(m, [r[inp] for r in recs])
            write_jsonl(OUT / f"hyp/m109.{args.tag}.{inp}.jsonl",
                        [{"id": r["id"], "input": r[inp], "hyp": h, "ms": t} for r, h, t in zip(recs, hyps, ms)])
            print(json.dumps({"task": "m109", "input": inp, **summ(ms),
                              "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "ex": hyps[:2]},
                             ensure_ascii=False), flush=True)
    if "murasaki" in tasks:
        rows = load_murasaki()[:args.limit or None]
        allms = []
        out = []
        for r in rows:
            hyps, ms = run(m, r["segs"])
            allms += ms
            out.append({"id": r["id"], "category": r["category"], "hyp": "".join(hyps), "seg_hyps": hyps,
                        "seg_ms": ms, "ms": None if args.batched else round(sum(ms), 2)})
        write_jsonl(OUT / f"hyp/murasaki.{args.tag}.seg.jsonl", out)
        print(json.dumps({"task": "murasaki_seg", **summ(allms), "ex": out[0]["hyp"][:60]}, ensure_ascii=False),
              flush=True)
    if "luna" in tasks:
        for name in LUNA_SETS:
            rows = load_luna(name)[:args.limit or None]
            hyps, ms = run(m, [r["source"] for r in rows])
            write_jsonl(OUT / f"hyp/luna_{name}.{args.tag}.single.jsonl",
                        [{"id": r["id"], "input": r["source"], "hyp": h, "ms": t} for r, h, t in zip(rows, hyps, ms)])
            print(json.dumps({"task": f"luna_{name}", **summ(ms)}), flush=True)
    if "throughput" in tasks:
        texts = [r["clean"] for r in recs]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hb = m.batched(texts, args.bsz)
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        f = OUT / f"hyp/m109.{args.tag}.clean.jsonl"
        h1 = [json.loads(x)["hyp"] for x in open(f, encoding="utf-8")] if f.exists() else None
        rep = {"tag": args.tag, "task": "m109_clean", "n": len(texts), "bsz": args.bsz, "secs": round(secs, 1),
               "items_per_s": round(len(texts) / secs, 1),
               "same_as_batch1": None if h1 is None else round(sum(a == b for a, b in zip(hb, h1)) / len(hb), 4)}
        (OUT / f"throughput.{args.tag}.json").write_text(json.dumps(rep) + "\n")
        print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
