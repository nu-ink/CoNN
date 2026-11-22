"""Two-hidden-layer MLP. Gates multiply a copy of W; they do not write W."""

from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from conn_exp.seeding import paired_seed


class GatedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if gate.shape != self.weight.shape:
            raise ValueError(f"gate shape {tuple(gate.shape)} != weight shape {tuple(self.weight.shape)}")
        applied = gate.detach().to(dtype=self.weight.dtype, device=self.weight.device)
        # Out-of-place product. Do not use mul_ on the parameter.
        w_eff = self.weight * applied
        return F.linear(x, w_eff, self.bias), w_eff


@dataclass
class ForwardCache:
    h1: torch.Tensor
    h2: torch.Tensor
    w_eff_ih: torch.Tensor
    w_eff_hh: torch.Tensor
    logits: torch.Tensor


class CoNNMLP(nn.Module):
    """Input→hidden and hidden→hidden are gated. The output head is dense."""

    def __init__(self, n_features: int, hidden_sizes: tuple[int, int]) -> None:
        super().__init__()
        hidden_1, hidden_2 = int(hidden_sizes[0]), int(hidden_sizes[1])
        self.fc1 = GatedLinear(n_features, hidden_1)
        self.fc2 = GatedLinear(hidden_1, hidden_2)
        self.out_weight = nn.Parameter(torch.empty(1, hidden_2))
        self.out_bias = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor, gate_ih: torch.Tensor, gate_hh: torch.Tensor) -> tuple[torch.Tensor, ForwardCache]:
        pre_h1, w_eff_ih = self.fc1(x, gate_ih)
        h1 = F.relu(pre_h1)
        pre_h2, w_eff_hh = self.fc2(h1, gate_hh)
        h2 = F.relu(pre_h2)
        logits = F.linear(h2, self.out_weight, self.out_bias).reshape(-1)
        return logits, ForwardCache(h1=h1, h2=h2, w_eff_ih=w_eff_ih, w_eff_hh=w_eff_hh, logits=logits)


def _uniform_weight(weight: torch.Tensor, generator: torch.Generator) -> None:
    fan_in = weight.shape[1]
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        weight.uniform_(-bound, bound, generator=generator)


def _uniform_bias(bias: torch.Tensor, fan_in: int, generator: torch.Generator) -> None:
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        bias.uniform_(-bound, bound, generator=generator)


def initialize_model(model: CoNNMLP, generator: torch.Generator) -> None:
    _uniform_weight(model.fc1.weight, generator)
    _uniform_bias(model.fc1.bias, model.fc1.weight.shape[1], generator)
    _uniform_weight(model.fc2.weight, generator)
    _uniform_bias(model.fc2.bias, model.fc2.weight.shape[1], generator)
    _uniform_weight(model.out_weight, generator)
    _uniform_bias(model.out_bias, model.out_weight.shape[1], generator)


def initial_state_dict(
    n_features: int, hidden_sizes: tuple[int, int], seed: int
) -> dict[str, torch.Tensor]:
    model = CoNNMLP(n_features, hidden_sizes)
    generator = torch.Generator()
    generator.manual_seed(paired_seed(seed, "init"))
    initialize_model(model, generator)
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def fingerprint_state(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        array = state[key].detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_optimizer(model: nn.Module, lr: float) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    if lr <= 0:
        raise ValueError("learning rate must be positive")
    kwargs: dict[str, object] = {
        "lr": float(lr),
        "momentum": 0.0,
        "dampening": 0.0,
        "weight_decay": 0.0,
        "nesterov": False,
    }
    signature = inspect.signature(torch.optim.SGD)
    if "maximize" in signature.parameters:
        kwargs["maximize"] = False
    if "foreach" in signature.parameters:
        kwargs["foreach"] = False
    if "differentiable" in signature.parameters:
        kwargs["differentiable"] = False
    if "fused" in signature.parameters:
        kwargs["fused"] = False
    optimizer = torch.optim.SGD(model.parameters(), **kwargs)
    recorded = {"name": "sgd", **kwargs}
    return optimizer, recorded


def trainable_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
