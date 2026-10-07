"""JMangaTranslator-Fast: Japanese -> Simplified Chinese translation of manga speech bubbles, one bubble at a time.

    from jmt_fast import load
    tr = load("path/to/release", backend="auto")
    tr.translate("堪忍袋の緒が切れた！")
"""
from __future__ import annotations

import importlib.util
import platform

BACKENDS = ("torch", "cuda-graphs", "onnx", "coreml", "mlx")


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def pick() -> str:
    """auto: CUDA Graphs on an NVIDIA GPU, MLX on Apple silicon, else ONNX Runtime on CPU, else PyTorch."""
    if _has("torch"):
        import torch
        if torch.cuda.is_available():
            return "cuda-graphs"
    if platform.system() == "Darwin" and platform.machine() == "arm64" and _has("mlx"):
        return "mlx"
    return "onnx" if _has("onnxruntime") else "torch"


def load(root, backend: str = "auto", **kw):
    backend = pick() if backend == "auto" else backend
    if backend == "torch":
        from .backends import TorchEager
        return TorchEager(root, **kw)
    if backend == "cuda-graphs":
        from .backends import CudaGraphs
        return CudaGraphs(root, **kw)
    if backend == "onnx":
        from .backends import OnnxCPU
        return OnnxCPU(root, **kw)
    if backend == "coreml":
        from .backends import CoreML
        return CoreML(root, **kw)
    if backend == "mlx":
        from .mlx_backend import MLX
        return MLX(root, **kw)
    raise ValueError(f"unknown backend {backend!r}; one of auto, {', '.join(BACKENDS)}")
