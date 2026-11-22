"""Fixed 1-D coordinates. Labels and splits are not inputs to this module."""

from __future__ import annotations

from dataclasses import dataclass

import torch

COORDINATE_CONVENTION = (
    "Input feature i and hidden unit j lie on the unit interval. "
    "Coordinate of index i among n positions is i/(n-1) for n>1, and 0.5 for n=1. "
    "Distance is the absolute difference. "
    "Gate matrix rows are destination units and columns are source units, matching PyTorch Linear.weight. "
    "Hidden units are fixed on this grid before training. Coordinates are not learned and do not depend on labels or on the evaluation split."
)


def uniform_coordinates(n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return torch.tensor([0.5], dtype=dtype)
    return torch.linspace(0, 1, n, dtype=dtype)


def pairwise_abs_distance(dest: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Return |dest_j - src_i| with shape [n_dest, n_src]."""
    return (dest[:, None] - src[None, :]).abs()


@dataclass
class Geometry:
    input_coords: torch.Tensor
    hidden_coords: tuple[torch.Tensor, torch.Tensor]
    dist_ih: torch.Tensor
    dist_hh: torch.Tensor


def make_task_geometry(n_features: int, hidden_sizes: tuple[int, int]) -> Geometry:
    if len(hidden_sizes) != 2:
        raise ValueError("expected two hidden layers")
    input_coords = uniform_coordinates(n_features)
    hidden_1 = uniform_coordinates(int(hidden_sizes[0]))
    hidden_2 = uniform_coordinates(int(hidden_sizes[1]))
    return Geometry(
        input_coords=input_coords,
        hidden_coords=(hidden_1, hidden_2),
        dist_ih=pairwise_abs_distance(hidden_1, input_coords),
        dist_hh=pairwise_abs_distance(hidden_2, hidden_1),
    )
