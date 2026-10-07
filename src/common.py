"""Shared data loading for the GalTransl-v4-4B vs R2 comparison (Titan Xp, user 2026-10-06 20:0x).

Tasks
  m109      Manga109-s real-OCR set (artifacts/m109_ocr_20261006/sample.jsonl, 3,817 boxes); inputs clean / nar / mocr;
            one box = one request (deployment-like, no context).
  murasaki  Murasaki benchmark light-novel paragraphs (100 Short ~190 chars + 100 Long ~790 chars, human references);
            paragraphs are cut into sentences by split_ja(); the hypothesis is the concatenation of segment outputs.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M109 = ROOT / "artifacts/m109_ocr_20261006"
MURASAKI = ROOT / "benchmarks/Murasaki-benchmark/data"
OUT = Path(os.environ.get("SAKURA_OUT", ROOT / "artifacts/sakura_cmp_20261006"))   # smoke tests use another dir

_END = "。！？!?…"
_CLOSE = "」』）)】〉》"


def split_ja(text: str, max_len: int = 120) -> list[str]:
    """Sentence segments, never inside 「」『』（）: outside brackets, cut after sentence-final punctuation and after a
    closing bracket that ends a quote (like one speech balloon / one line); segments longer than max_len are cut again
    after 、 or ，."""
    segs, cur, depth = [], "", 0
    for i, ch in enumerate(text):
        cur += ch
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if ch in "「『（(【":
            depth += 1
        elif ch in _CLOSE and depth:
            depth -= 1
            if depth == 0 and nxt not in _END and nxt not in _CLOSE:
                segs.append(cur); cur = ""
        elif depth == 0 and ch in _END and nxt not in _END and nxt not in _CLOSE:
            segs.append(cur); cur = ""
    if cur:
        segs.append(cur)
    out = []
    for s in segs:
        while len(s) > max_len:
            cut = max(s.rfind("、", 0, max_len), s.rfind("，", 0, max_len))
            cut = cut + 1 if cut > 0 else max_len
            out.append(s[:cut]); s = s[cut:]
        if s:
            out.append(s)
    assert "".join(out) == text
    return out


def load_m109() -> list[dict]:
    return [json.loads(x) for x in open(M109 / "sample.jsonl", encoding="utf-8")]


def load_murasaki() -> list[dict]:
    rows = []
    for cat in ("short", "long"):
        for k, x in enumerate(open(MURASAKI / f"dataset_{cat}.jsonl", encoding="utf-8")):
            r = json.loads(x)
            rows.append({"id": f"{cat}{k:03d}", "category": r["category"], "src": r["src"], "ref": r["ref"],
                         "segs": split_ja(r["src"])})
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def strip_think(s: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", s, flags=re.S).strip()


LUNA_SETS = ("dev", "heldout", "luna1k", "manga200")
LUNA = ROOT / "artifacts/ar_mt_20261005/data_v2_ext"


def load_luna(name: str) -> list[dict]:
    """Our standard sets (Luna manga lines + Manga200); references are in the Luna teacher's style, which our model was
    trained on, so these favour our model."""
    return [json.loads(x) for x in open(LUNA / f"eval_{name}.jsonl", encoding="utf-8")]
