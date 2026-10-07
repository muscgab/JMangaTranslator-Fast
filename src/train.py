#!/usr/bin/env python3
"""Train the ModernBERT-ja + AR decoder translator (model.py).

Stage A (steps < --stage-a): encoder frozen and run without grad; bridge, fusion, decoder train.
Stage B: encoder unfrozen at lr * --enc-lr-mult, warmed up over --enc-warmup steps.
Schedule: linear warmup over --warmup steps, cosine decay to --final-lr-frac * lr at --steps.
Batches: token budget (padded src + padded tgt per row times rows <= --max-tokens), built from pools of
mixed-source examples sorted by length. Sources are memory-mapped packed files from build_data.py and are
mixed by --mix weights; each source walks a fresh seeded permutation per epoch. The sampler state is saved
with every checkpoint, so --resume continues on exactly the next batch.
Checkpoints: last.pt (model + optimizer + sampler, every --save-minutes and at the end), best.pt (model only,
best dev chrF). Log: train_log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ARMT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DAT_TOK = ROOT / "artifacts/preview200m_training_20260923/tokenizer/joint.model"
DAT_SHARED = ROOT / "artifacts/ar_mt_20261005/dat_shared_fp16.pt"    # {"shared.weight": [40960,768]} from base_fp16.pt
DAT_WEIGHTS = ROOT / "artifacts/dat_context_20260926/pack/base_fp16.pt"
UNK, BOS, EOS, PAD = 0, 1, 2, 3
MB_PAD = 3


def chrf(hyps: list[str], refs: list[str]) -> float:
    try:
        import sacrebleu

        return float(sacrebleu.metrics.CHRF().corpus_score(hyps, [refs]).score)
    except ImportError:                                   # fallback: char 6-gram, beta 2, whitespace removed
        stats = np.zeros((6, 3))
        for h, r in zip(hyps, refs):
            h, r = h.replace(" ", ""), r.replace(" ", "")
            for n in range(1, 7):
                hc = Counter(h[i:i + n] for i in range(len(h) - n + 1))
                rc = Counter(r[i:i + n] for i in range(len(r) - n + 1))
                stats[n - 1] += (sum((hc & rc).values()), sum(hc.values()), sum(rc.values()))
        p = np.mean([m / hh if hh else 0 for m, hh, _ in stats])
        rr = np.mean([m / rf if rf else 0 for m, _, rf in stats])
        return float(100 * 5 * p * rr / (4 * p + rr)) if p + rr else 0.0


class Source:
    def __init__(self, data: Path, name: str, lut: np.ndarray):
        self.name = name
        self.src = np.memmap(data / f"{name}.src.bin", dtype=np.uint32, mode="r")
        self.tgt = np.memmap(data / f"{name}.tgt.bin", dtype=np.uint16, mode="r")
        self.so = np.load(data / f"{name}.src.idx.npy")
        self.to = np.load(data / f"{name}.tgt.idx.npy")
        self.n = len(self.so) - 1
        self.slen, self.tlen = np.diff(self.so), np.diff(self.to)
        self.lut = lut

    def get(self, i: int):
        return np.asarray(self.src[self.so[i]:self.so[i + 1]], dtype=np.int64), \
            self.lut[np.asarray(self.tgt[self.to[i]:self.to[i + 1]], dtype=np.int64)]


class Sampler:
    def __init__(self, sources: list[Source], weights: list[float], seed: int, max_tokens: int, pool: int):
        self.sources, self.seed, self.max_tokens, self.pool = sources, seed, max_tokens, pool
        self.w = np.asarray(weights, dtype=np.float64) / sum(weights)
        self.state = {"rng": np.random.default_rng(seed).bit_generator.state, "epoch": [0] * len(sources),
                      "pos": [0] * len(sources), "consumed": 0}
        self._perm = {}
        self._pool_state = None
        self._batches = []

    def perm(self, k: int, epoch: int):
        """Permutation of source k for this epoch; one cached per source (sources alternate within a pool)."""
        cached = self._perm.get(k)
        if cached is None or cached[0] != epoch:
            cached = (epoch, np.random.default_rng([self.seed, k, epoch]).permutation(self.sources[k].n))
            self._perm[k] = cached
        return cached[1]

    def _build_pool(self):
        self._pool_state = json.loads(json.dumps(self.state))
        rng = np.random.default_rng()
        rng.bit_generator.state = self.state["rng"]
        picks = rng.choice(len(self.sources), size=self.pool, p=self.w)
        items = []
        for k in picks:
            s = self.sources[k]
            if self.state["pos"][k] >= s.n:
                self.state["epoch"][k] += 1
                self.state["pos"][k] = 0
            items.append((k, int(self.perm(k, self.state["epoch"][k])[self.state["pos"][k]])))
            self.state["pos"][k] += 1
        items.sort(key=lambda it: (int(self.sources[it[0]].slen[it[1]]), int(self.sources[it[0]].tlen[it[1]])))
        batches, cur, ms, mt = [], [], 0, 0
        for k, i in items:
            s, t = int(self.sources[k].slen[i]), int(self.sources[k].tlen[i])
            if cur and (len(cur) + 1) * (max(ms, s) + max(mt, t)) > self.max_tokens:
                batches.append(cur)
                cur, ms, mt = [], 0, 0
            cur.append((k, i))
            ms, mt = max(ms, s), max(mt, t)
        if cur:
            batches.append(cur)
        order = rng.permutation(len(batches))
        self._batches = [batches[j] for j in order]
        self.state["rng"] = rng.bit_generator.state
        self.state["consumed"] = 0

    def resume(self, saved: dict):
        """saved = pool-start state + batches consumed from that pool."""
        self.state = json.loads(json.dumps(saved["pool_state"]))
        self._build_pool()
        self._batches = self._batches[saved["consumed"]:]
        self.state["consumed"] = saved["consumed"]

    def checkpoint(self) -> dict:
        return {"pool_state": self._pool_state, "consumed": self.state["consumed"]}

    def next(self):
        if not self._batches:
            self._build_pool()
        b = self._batches.pop(0)
        self.state["consumed"] += 1
        return [self.sources[k].get(i) for k, i in b], Counter(self.sources[k].name for k, _ in b)


def collate(rows, device):
    b = len(rows)
    ls, lt = max(len(s) for s, _ in rows), max(len(t) for _, t in rows)
    src = np.full((b, ls), MB_PAD, dtype=np.int64)
    tgt = np.full((b, lt), PAD, dtype=np.int64)
    for j, (s, t) in enumerate(rows):
        src[j, :len(s)] = s
        tgt[j, :len(t)] = t
    src = torch.from_numpy(src).to(device)
    mask = torch.from_numpy(np.arange(ls)[None] < np.array([len(s) for s, _ in rows])[:, None]).to(device)
    return src, mask, torch.from_numpy(tgt).to(device)


def load_eval(path: Path, sp, lut, n: int | None):
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    rows = rows[:n] if n else rows
    for r in rows:
        r["tgt"] = lut[np.asarray([BOS] + sp.encode(r["reference"]) + [EOS], dtype=np.int64)]
        r["src"] = np.asarray(r["src"], dtype=np.int64)
    return rows


def decode_ids(ids, dat_ids, sp):
    return sp.decode([int(dat_ids[i]) for i in ids if dat_ids[i] >= 4])


@torch.no_grad()
def evaluate(model, rows, sp, dat_ids, device, amp, bsz=64, byte_compact=None):
    model.eval()
    tot, cnt, hyps = 0.0, 0, []
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["src"]))
    for j in range(0, len(order), bsz):
        part = [rows[i] for i in order[j:j + bsz]]
        src, mask, tgt = collate([(r["src"], r["tgt"]) for r in part], device)
        with amp():
            _, nll, n = model.loss(src, mask, tgt, PAD, 0.0)
            out = model.generate(src, mask, BOS, EOS, PAD, max_len=3 * src.shape[1] + 10, byte_compact=byte_compact)
        tot += float(nll) * int(n)
        cnt += int(n)
        hyps += list(zip([order[k] for k in range(j, j + len(part))], out))
    hyps = [decode_ids(o, dat_ids, sp) for _, o in sorted(hyps)]
    model.train()
    return tot / cnt, chrf(hyps, [r["reference"] for r in rows]), hyps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "artifacts/ar_mt_20261005/data")
    ap.add_argument("--mix", default="luna:1")
    ap.add_argument("--encoder", default=str(ROOT / "models/modernbert_ja/modernbert-ja-310m"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dec-layers", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=4096)
    ap.add_argument("--bridge", default="swiglu", choices=["swiglu", "mlp", "linear"])
    ap.add_argument("--bridge-hidden", type=int, default=2048)
    ap.add_argument("--fusion", type=int, default=1)
    ap.add_argument("--null-tokens", type=int, default=2)
    ap.add_argument("--max-tgt", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--init-emb", type=int, default=1)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--stage-a", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc-lr-mult", type=float, default=0.1)
    ap.add_argument("--enc-warmup", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--final-lr-frac", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--pool", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--save-minutes", type=float, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps-this-run", type=int, default=0, help="stop early (smoke tests)")
    ap.add_argument("--stop-at", type=int, default=0, help="stop at this absolute step; the LR schedule still uses --steps")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_bf16 = bool(args.bf16) and device == "cuda"
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else (lambda: torch.autocast(device, enabled=False))
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    vocab = json.loads((args.data / "vocab.json").read_text())
    dat_ids = vocab["dat_ids"]
    lut = np.zeros(vocab["dat_vocab"], dtype=np.int64)               # DAT id -> compact id (unknown -> UNK)
    for i, d in enumerate(dat_ids):
        if d >= 0:
            lut[d] = i
    sp = spm.SentencePieceProcessor(model_file=str(DAT_TOK))
    byte_compact = [int(lut[x]) for x in vocab["byte_piece_ids"]] if "byte_piece_ids" in vocab else None

    cfg = {"encoder": args.encoder, "dec_layers": args.dec_layers, "ffn": args.ffn, "bridge": args.bridge,
           "bridge_hidden": args.bridge_hidden, "fusion": bool(args.fusion), "null_tokens": args.null_tokens,
           "vocab": len(dat_ids), "max_tgt": args.max_tgt, "dropout": args.dropout}
    model = ARMT(cfg)
    if args.init_emb and not args.resume and model.d == 768:
        table = (torch.load(DAT_SHARED, map_location="cpu")["shared.weight"] if DAT_SHARED.exists() else
                 torch.load(DAT_WEIGHTS, map_location="cpu", weights_only=False)["model"]["shared.weight"]).float()
        print(json.dumps({"emb_rows_from_dat": model.init_embeddings(table, dat_ids)}), flush=True)
        del table
    model.to(device).train()

    enc_params = list(model.encoder.parameters())
    dec_named = [(n, p) for n, p in model.named_parameters() if not n.startswith("encoder.")]
    # no weight decay on embeddings, positions, null memory, norms, and the fusion scales/logits (gamma, logits)
    nd = lambda n, p: p.ndim < 2 or n.startswith(("emb.", "pos.", "null", "fusion.gamma", "fusion.logits"))  # noqa: E731
    decay = [p for n, p in dec_named if not nd(n, p)]
    no_decay = [p for n, p in dec_named if nd(n, p)]
    enc_decay = [p for p in enc_params if p.ndim >= 2]
    enc_no = [p for p in enc_params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd, "kind": "dec"},
                             {"params": no_decay, "weight_decay": 0.0, "kind": "dec"},
                             {"params": enc_decay, "weight_decay": args.wd, "kind": "enc"},
                             {"params": enc_no, "weight_decay": 0.0, "kind": "enc"}],
                            lr=args.lr, betas=(0.9, 0.98), eps=1e-8, fused=device == "cuda")

    mix = [(s.split(":")[0], float(s.split(":")[1])) for s in args.mix.split(",")]
    sources = [Source(args.data, name, lut) for name, _ in mix]
    sampler = Sampler(sources, [w for _, w in mix], args.seed, args.max_tokens, args.pool)
    dev = load_eval(args.data / "eval_dev.jsonl", sp, lut, args.eval_n)

    step, best = 0, -1.0
    if args.resume and (args.out / "last.pt").exists():
        ck = torch.load(args.out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sampler.resume(ck["sampler"])
        step, best = ck["step"], ck["best"]
        torch.set_rng_state(ck["torch_rng"])
        print(json.dumps({"resumed_step": step}), flush=True)
        del ck
    (args.out / "config.json").write_text(json.dumps({"cfg": cfg, "args": {k: str(v) for k, v in vars(args).items()},
                                                      "params_total": sum(p.numel() for p in model.parameters()),
                                                      "params_encoder": sum(p.numel() for p in enc_params),
                                                      "sources": {s.name: s.n for s in sources}}, indent=1) + "\n")

    def save(name: str, full: bool):
        tmp = args.out / f"{name}.tmp"
        obj = {"model": model.state_dict(), "cfg": cfg, "vocab": str(args.data / "vocab.json"), "step": step, "best": best}
        if full:
            obj.update(opt=opt.state_dict(), sampler=sampler.checkpoint(), torch_rng=torch.get_rng_state())
        torch.save(obj, tmp)
        os.replace(tmp, args.out / f"{name}.pt")

    log = open(args.out / "train_log.jsonl", "a")
    t0 = last_save = time.time()
    win = {"loss": 0.0, "nll": 0.0, "tok": 0, "rows": 0, "n": 0, "src": Counter()}
    started = step
    end = min(args.steps, args.stop_at) if args.stop_at else args.steps
    while step < end:
        stage_b = step >= args.stage_a
        warm = min(1.0, (step + 1) / args.warmup)
        prog = min(1.0, step / max(1, args.steps))
        lr = args.lr * warm * (args.final_lr_frac + (1 - args.final_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))
        enc_lr = lr * args.enc_lr_mult * min(1.0, (step - args.stage_a + 1) / args.enc_warmup) if stage_b else 0.0
        for g in opt.param_groups:
            g["lr"] = lr if g["kind"] == "dec" else enc_lr
        for _ in range(args.accum):
            rows, names = sampler.next()
            src, mask, tgt = collate(rows, device)
            with amp():
                loss, nll, n = model.loss(src, mask, tgt, PAD, args.smoothing, encoder_grad=stage_b)
            (loss / args.accum).backward()
            win["loss"] += float(loss.detach())
            win["nll"] += float(nll.detach())
            win["tok"] += int(n)
            win["rows"] += len(rows)
            win["n"] += 1
            win["src"].update(names)
        gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0 or step == args.steps:
            dt = time.time() - t0
            rec = {"step": step, "stage": "B" if stage_b else "A", "loss": round(win["loss"] / win["n"], 4),
                   "nll": round(win["nll"] / win["n"], 4), "lr": lr, "enc_lr": enc_lr, "grad_norm": round(float(gn), 3),
                   "tgt_tok_s": round(win["tok"] / dt, 1), "rows_s": round(win["rows"] / dt, 1),
                   "rows_per_step": round(win["rows"] / max(1, win["n"] / args.accum), 1),
                   "mix": dict(win["src"]), "epochs": dict(zip([s.name for s in sources], sampler.state["epoch"]))}
            if model.fusion is not None:
                rec["gamma_abs"] = round(float(model.fusion.gamma.abs().mean()), 5)
                a = model.fusion.logits.softmax(-1)
                rec["depth_entropy"] = [round(float(-(r * r.log()).sum()), 3) for r in a]
            if device == "cuda":
                rec["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            t0, win = time.time(), {"loss": 0.0, "nll": 0.0, "tok": 0, "rows": 0, "n": 0, "src": Counter()}
        if step % args.eval_every == 0 or step == end:
            dev_nll, dev_chrf, hyps = evaluate(model, dev, sp, dat_ids, device, amp, byte_compact=byte_compact)
            rec = {"step": step, "dev_nll": round(dev_nll, 4), "dev_chrf": round(dev_chrf, 3), "examples": hyps[:3]}
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            if dev_chrf > best:
                best = dev_chrf
                save("best", full=False)
            t0 = time.time()
        if time.time() - last_save > 60 * args.save_minutes or step == end:
            save("last", full=True)
            last_save = time.time()
        if args.max_steps_this_run and step - started >= args.max_steps_this_run:
            save("last", full=True)
            break
    (args.out / "done.json").write_text(json.dumps({"step": step, "best_dev_chrf": best}) + "\n")


if __name__ == "__main__":
    main()
