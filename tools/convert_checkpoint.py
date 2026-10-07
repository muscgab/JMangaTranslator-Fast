#!/usr/bin/env python3
"""Training checkpoint (.pt, pickled dict) -> release files: model.safetensors (every tensor, fp32, unchanged) and
config.json (decoder hyperparameters and special ids). Re-reads both and checks every tensor bit for bit.

  python tools/convert_checkpoint.py CKPT.pt OUT_DIR
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    sd = {k: v.contiguous() for k, v in ck["model"].items()}
    c = ck["cfg"]
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    cfg = {"model_type": "jmangatranslator-fast", "version": "v1", "training_step": ck.get("step"),
           "dec_layers": c["dec_layers"], "ffn": c["ffn"], "bridge": c["bridge"], "bridge_hidden": c["bridge_hidden"],
           "fusion": c["fusion"], "null_tokens": c["null_tokens"], "vocab": c["vocab"], "max_tgt": c["max_tgt"],
           "max_src_tokens": 256, "bos": 1, "eos": 2, "pad": 3, "src_pad": 3,
           "source_checkpoint_sha256": sha256(ckpt)}
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    back = load_file(str(out / "model.safetensors"))
    assert back.keys() == sd.keys()
    bad = [k for k in sd if back[k].dtype != sd[k].dtype or not torch.equal(back[k], sd[k])]
    assert not bad, bad[:5]
    print(json.dumps({"tensors": len(sd), "params": sum(v.numel() for v in sd.values()),
                      "model.safetensors": sha256(out / "model.safetensors"),
                      "bytes": (out / "model.safetensors").stat().st_size, **cfg}, indent=1))


if __name__ == "__main__":
    main()
