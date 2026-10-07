#!/usr/bin/env python3
"""Zakkuri v1 on CPU through ONNX Runtime, fp32 (user 2026-10-07: "做。你试试看。int8先不做").

export  export_coreml.EncoderExport (token ids -> 4 cross K/V; masks and RoPE from positions, so the length axis is
        dynamic up to lmax) and DecoderExport (one step, self-attention cache T=128, cross length 2+max bucket)
        -> artifacts/onnx_20261007/{encoder,decoder}_fp32.onnx (legacy TorchScript exporter, opset 17).
bench   export_coreml.Host greedy loop (the one validated against ARMT on Core ML) with ONNX Runtime CPU sessions;
        first --n boxes of sample500.jsonl after --warm; agreement with a reference jsonl (MPS eager fp32 hyps).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ar_mt"))
sys.path.insert(0, str(ROOT / "benchmarks/sakura_cmp"))
OUT = ROOT / "artifacts/onnx_20261007"
CKPT = ROOT / "artifacts/r2_20261006/weights/r2_step54000.pt"
ENC = ROOT / "artifacts/ar_mt_20261005/weights/dapt_v1_model"
DATA = ROOT / "artifacts/ar_mt_20261005/data_v2_ext"
BUCKETS, T = (8, 16, 32, 64, 128), 128


def load_model():
    import torch  # noqa: F401
    from run_ours import Ours
    o = Ours(CKPT, str(ENC), DATA, "cpu")
    o.model.float().eval()
    return o


def export(o):
    import torch
    import export_coreml as ex
    OUT.mkdir(parents=True, exist_ok=True)
    m = o.model
    enc = ex.EncoderExport(m, max(BUCKETS), 1.0).eval()
    dec = ex.DecoderExport(m, 1.0).eval()
    H, hd, d, M = m.layers[0].h, m.layers[0].hd, m.d, 2 + max(BUCKETS)
    ids = torch.full((1, 16), 3, dtype=torch.long)
    ids[0, :9] = torch.arange(5, 14)
    with torch.no_grad():
        torch.onnx.export(enc, (ids,), str(OUT / "encoder_fp32.onnx"), input_names=["ids"],
                          output_names=["ck0", "cv0", "ck1", "cv1"], opset_version=17, dynamo=False,
                          dynamic_axes={"ids": {1: "L"}, **{n: {2: "L2"} for n in ("ck0", "cv0", "ck1", "cv1")}})
        z = lambda *s: torch.zeros(*s)  # noqa: E731
        args = (z(1, 1, d), z(1, H, T, hd), z(1, H, T, hd), z(1, H, T, hd), z(1, H, T, hd), torch.full((1, 1, 1, T), ex.NEG),
                z(1, H, M, hd), z(1, H, M, hd), z(1, H, M, hd), z(1, H, M, hd), z(1, 1, 1, M))
        torch.onnx.export(dec, args, str(OUT / "decoder_fp32.onnx"),
                          input_names=["x", "kc0", "vc0", "kc1", "vc1", "smask", "ck0", "cv0", "ck1", "cv1", "cmask"],
                          output_names=["logits", "k0", "v0", "k1", "v1"], opset_version=17, dynamo=False)
    for f in ("encoder_fp32.onnx", "decoder_fp32.onnx"):
        print(f, round((OUT / f).stat().st_size / 2**20, 1), "MiB", flush=True)


def sessions(threads: int, spin: int = 1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.add_session_config_entry("session.intra_op.allow_spinning", str(spin))
    mk = lambda f: ort.InferenceSession(str(OUT / f), so, providers=["CPUExecutionProvider"])  # noqa: E731
    return mk("encoder_fp32.onnx"), mk("decoder_fp32.onnx")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("export", "bench"))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--spin", type=int, default=1, help="0 = worker threads sleep instead of spinning")
    ap.add_argument("--exact", type=int, default=0, help="1 = encode at the true length (every length its own bucket)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--out", type=Path, default=OUT / "bench")
    args = ap.parse_args()
    o = load_model()
    if args.mode == "export":
        export(o)
        return
    import export_coreml as ex
    se, sd = sessions(args.threads, args.spin)
    vocab = json.loads((DATA / "vocab.json").read_text())
    enc_fn = lambda arr: se.run(None, {"ids": arr.astype(np.int64)})  # noqa: E731
    names = ["x", "kc0", "vc0", "kc1", "vc1", "smask", "ck0", "cv0", "ck1", "cv1", "cmask"]

    def dec_fn(x, caches, smask, cr, cmask):
        return sd.run(None, dict(zip(names, [x, *caches, smask, *cr, cmask])))
    host = ex.Host(o.model, vocab, o.tok, o.sp, tuple(range(1, max(BUCKETS) + 1)) if args.exact else BUCKETS, T, enc_fn, dec_fn)
    rows = [json.loads(x) for x in open(Path(__file__).with_name("sample500.jsonl"), encoding="utf-8")][:args.n]
    for r in rows[:args.warm]:
        host(r["clean"])
    res = []
    for r in rows:
        t = time.perf_counter()
        x = host(r["clean"])
        res.append({"id": r["id"], "hyp": x["hyp"], "ms": round(1000 * (time.perf_counter() - t), 2), "enc_ms": x["enc_ms"],
                    "nout": x["nout"]})
    a = np.array([x["ms"] for x in res])
    ref = {json.loads(x)["id"]: json.loads(x)["hyp"] for x in open(args.ref, encoding="utf-8")} if args.ref else None
    rep = {"mode": f"onnx_cpu_fp32_t{args.threads}" + ("_exact" if args.exact else "") + ("" if args.spin else "_nospin"), "n": len(a), "p50_ms": round(float(np.percentile(a, 50)), 1),
           "p90_ms": round(float(np.percentile(a, 90)), 1), "mean_ms": round(float(a.mean()), 1),
           "enc_ms_p50": round(float(np.percentile([x["enc_ms"] for x in res], 50)), 1),
           "dec_ms_per_step": round(float(sum(x["ms"] - x["enc_ms"] for x in res) / max(1, sum(x["nout"] + 1 for x in res))), 2),
           "identical_to_ref": None if ref is None else round(float(np.mean([x["hyp"] == ref[x["id"]] for x in res])), 4)}
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / f"{rep['mode']}.jsonl", "w", encoding="utf-8") as f:
        for x in res:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
