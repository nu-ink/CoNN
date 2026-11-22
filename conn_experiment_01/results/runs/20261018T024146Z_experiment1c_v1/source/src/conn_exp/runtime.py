"""Recorded runtime flags. Device selection does not fall back silently."""

from __future__ import annotations

import torch


def configure_runtime() -> dict[str, object]:
    torch.set_num_threads(1)
    flags: dict[str, object] = {"torch_num_threads": 1}
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    flags["cudnn_deterministic"] = True
    flags["cudnn_benchmark"] = False
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is not None and hasattr(cuda_backends, "matmul"):
        cuda_backends.matmul.allow_tf32 = False
        flags["cuda_matmul_allow_tf32"] = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
        flags["cudnn_allow_tf32"] = False
    return flags


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Config requested cuda, but CUDA is not available. Refusing to silently use CPU."
            )
        return torch.device("cuda")
    raise ValueError(f"unknown device {name}")
