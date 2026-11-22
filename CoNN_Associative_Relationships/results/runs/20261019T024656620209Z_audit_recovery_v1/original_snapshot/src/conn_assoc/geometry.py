"""Positional geometry copied from the Experiment 2 coordinate convention.

Input feature i and hidden unit j lie on the unit interval.
The coordinate of index i among n positions is i/(n-1) for n>1, and 0.5 for n=1.
Distance is the absolute difference. Gate rows are destinations and columns are sources.
"""

from __future__ import annotations

import torch


def uniform_coordinates(n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return torch.tensor([0.5], dtype=dtype)
    return torch.linspace(0, 1, n, dtype=dtype)


def pairwise_abs_distance(dest: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Return |dest_j - src_i| with shape [n_dest, n_src]."""
    return (dest[:, None] - src[None, :]).abs()


def temporal_input_coordinates(n_samples: int = 128, n_channels: int = 9) -> torch.Tensor:
    """Time coordinate t/(n_samples-1), repeated across colocated channels."""
    if n_samples < 2:
        raise ValueError("a window needs at least two samples")
    time = torch.arange(n_samples, dtype=torch.float32) / (n_samples - 1)
    return time.repeat_interleave(n_channels)


def engineered_input_coordinates(n_features: int = 561) -> torch.Tensor:
    """Official metadata order, coordinate j/(n_features-1). Not a physical distance."""
    return uniform_coordinates(n_features)
