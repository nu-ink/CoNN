# 03 — Mathematical audit

Primary implementation: `conn_experiment_01/src/conn_exp/gates.py`, `geometry.py`, `models.py`, `schedules.py`, `schema.py`. The associative package copies the same gate arithmetic in `conn_assoc/gate.py` and `phase23_core.py`.

## M-01. Gaussian gate

**Origin.** Build brief and `gates.py` lines 20–28. Purpose: a fixed, nonnegative multiplier on each weight.

**Definition.** For distance `d >= 0` and bandwidth `tau > 0`,

```
G(d; tau) = exp( - d^2 / (2 tau^2) )
```

The code floors `tau` at `tau_floor` (default `1e-6`) and clamps the result to `[0, 1]`.

**Independent derivation.** This is the standard one-dimensional Gaussian kernel evaluated at absolute distance, normalized so `G(0)=1`. Set `G=1/2`:

```
1/2 = exp( - d50^2 / (2 tau^2) )
log 2 = d50^2 / (2 tau^2)
d50 = tau * sqrt(2 log 2)
```

`sqrt(2 log 2) = 1.1774100225154747`. The brief's recorder constant `1.17741` is that number rounded to five decimals (`schema.py` line 27, `D50_RECORD_FACTOR = 117741/100000`). The gate uses the exact square root (`schedules.py` line 11). `config.py` lines 172–173 set `tau = d50 / sqrt(2 log 2)`.

**Consistency.** `d` and `tau` are both in the coordinate unit of the unit interval, so the ratio is dimensionless and the exponential is defined. Clamp to `[0, 1]` does not change a real finite result: the exponent is nonpositive, so `G` is already in `(0, 1]`. Clamp matters only for nonfinite inputs.

**Assumptions.** Distance is absolute difference on a fixed line. The same `tau` is used for every entry of a layer at a given step. `tau` is not a function of the input or the loss.

**Boundary.** `d=0` gives `G=1`. As `d/tau` grows, `G` falls below any positive threshold. As `tau` grows, `G` increases toward 1 at every fixed `d`. The exact-ones phase is a hard replacement, not the limit `tau → infinity`. At the scheduled end `d50=1`, `G(d=1)=1/2`, so release changes the farthest multipliers from 1/2 to 1.

**Implementation match.** **CONFIRMED** for the displayed formula. The formula `tau(q) = sigma / r(q)` is **NOT TESTED** because it is not in the repository. Nothing in the code defines `q`, `sigma`, or `r(q)`.

## M-02. Distance

**Definition.** `geometry.py` lines 18–28. Index `i` among `n>1` positions has coordinate `i/(n-1)`. Distance is `|coordinate_destination - coordinate_source|`, shape `[n_destination, n_source]`, matching `nn.Linear.weight`.

**What the geometry means.** On the synthetic task the feature index is the physical layout of the motifs, so neighborhood is meaningful by construction. On engineered HAR, `PROTOCOL_v2.md` states that coordinate `j/560` follows metadata file order and is a semantic prior, not a measured sensor distance. On temporal HAR, all nine channels at time `t` share coordinate `t/127` (`geometry.py` in `conn_assoc`, lines 26–31). Channels are colocated. There is no learned or anatomical distance.

**Support for the claimed behavior.** The Gaussian implements a soft local window on whatever line was supplied. It implements task geometry only when that line is the task geometry. File order does not become physical geometry by being placed on `[0,1]`.

## M-03. Effective weight and the gradient identity

**Definition.** `models.py` lines 23–28:

```python
applied = gate.detach()
w_eff = self.weight * applied
```

So `W_eff = W ⊙ G` with `G` removed from the autograd graph.

**Derivation.** If `G` does not depend on `W`,

```
∂L/∂W_ij = (∂L/∂W_eff_ij) * G_ij
```

This is the chain rule. `schema.py` lines 13–17 state it. `diagnostics.py` lines 28–31 request both tensors from `torch.autograd.grad`. Because `w_eff = W * G_detached`, the leaf gradient with respect to `W` is the effective-weight gradient multiplied by `G`. The identity is a mathematical consequence of the product, not evidence that the curriculum is useful.

**Gate-parameter gradients.** `tau` is a Python float chosen by `schedule_at`. It is not an `nn.Parameter`. Coordinates are not parameters. Therefore training does not compute `∂L/∂tau` or `∂L/∂coordinates`.

If `G` were differentiable in `tau`, the unused derivative would be

```
∂G/∂tau = G * (d^2 / tau^3)
∂L/∂tau = sum_ij (∂L/∂W_eff_ij) * W_ij * ∂G/∂tau
```

Detach sets the applied contribution of this path to zero even if `tau` were later made a tensor. The fixed-gate case and a trainable gate are different models. Experiment 1 implements the fixed case. Phase 3 adds a second multiplier `C` that is not detached (`phase3_context.py` lines 45–48).

**SGD consequence.** With learning rate `η` and `G` held constant during a step,

```
ΔW = -η * G ⊙ ∂L/∂W_eff
```

A constant positive gate does not change the set of representable linear maps, because every `G_ij>0` can be absorbed into the scale of `W_ij`. It does change the optimization path: each edge has step size scaled by `G_ij`. For the effective weights the same constant gate gives `ΔW_eff = -η * G⊙G ⊙ ∂L/∂W_eff`. The curriculum is a prespecified trajectory of per-edge step sizes, followed, in conditions D, B′, C, and E, by a jump to `G=1`.

**Numerical profile, computed independently from the formula in float32.** For the 32×100 input layer at `d50=0.08`: mean `G^2` is 0.1135, 15.1% of entries exceed 1/2, 42.3% are below `1e-6`, and 0.19% are exact float32 zeros. At `d50=1`: mean `G^2` is 0.8142, the minimum is 1/2, and 99.9% of entries exceed 1/2. The early gate is a soft sparse mask. The late Gaussian is already close to dense. Exact release finishes that approach with a discontinuity of size `1-G`, which is 1/2 at distance 1.

The same pattern holds for the 64×1152 temporal input layer: at `d50=0.08`, 41.7% of gates are below `1e-6`.

## M-04. Schedule

`schedules.py` lines 14–43 and 121–186. For curriculum conditions, `tau` moves from `tau_start` to `tau_end` on steps `[0, release)` by smoothstep `u^2 (3-2u)`, then the gate is replaced by exact ones. `release_step = floor(0.7 * total_steps + 0.5)`. For 3200 steps that index is 2240.

Condition E (`schedules.py` lines 187–223) is exact ones, then `tau` from `tau_end` down to `tau_start`, then back to `tau_end`, then exact ones. That is a path control. It is not a monotone reversal of D.

Condition S holds `tau_start` through the final step. Its terminal function class is the local mask. D's terminal function class is an ordinary dense layer.

**Collapse versus bandwidth annealing.** Increasing `tau` widens the kernel. That is bandwidth expansion. Calling the exact-ones jump a collapse matches Experiment 1C's later wording (locality versus collapse) and conflicts with the infographic, which returns the network to a smaller graph. The mathematics that is implemented expands connectivity and then removes the mask. It does not delete neurons, merge units, or reduce width.

## M-05. Context factor

`phase3_context.py` lines 32–34, rank frozen at 2:

```
C(c) = 1 - (0.95/2) * sigmoid(U[c]) @ sigmoid(V[c]).T
W_eff(c) = W ⊙ detach(G) ⊙ C(c)
```

`0.95/2 = 0.475`. Entries of `C` lie in `[0.05, 1]` when the sigmoids lie in `(0,1)`, because the rank-2 Gram form of two sigmoid vectors is at most 2 and the coefficient is 0.475, so the subtracted term is at most 0.95. The code comment's bound is consistent with that calculation at saturation. Initialization `U=-6`, `V=0` gives `sigmoid(-6)≈0.002473` and `C≈0.998825`, matching the amendment.

**Rank.** If the two columns of `U[c]` are identical and the two columns of `V[c]` are identical, `sigmoid(U) sigmoid(V)^T` has rank at most 1. The original initialization sets every column to the same constant, so the nominal rank-2 factor begins at rank at most 1. Phase 3B measures that restriction and a small symmetry-breaking perturbation. A second singular component near `0.003` times the first is a small perturbation of a rank-1 attenuation, not two independent learned relations.

**Local derivatives.** Let `A = 1-C` and `s` be the sigmoid. For component `k`,

```
∂A_ij/∂U_ik = 0.475 * s'(U_ik) * s(V_jk)
∂A_ij/∂V_jk = 0.475 * s(U_ik) * s'(V_jk)
```

`s'(z)=s(z)(1-s(z))`. At `z=-6`, `s'≈0.00247`. At `z=-4.5`, `s'≈0.0109`. Moving the center from −6 to −4.5 increases the local factor by about four, which is the direction of the reported gradient sweep. These local factors are not the full loss gradient. The authors report an autograd match to `1.39e-17` in a float64 fixture. This audit did not re-execute that fixture. The algebra of the local factor is **CONFIRMED**. The numerical autograd claim is **UNVERIFIABLE** in this session because the fixture was not rerun. It is consistent with a correct implementation.

`C` multiplies `W` and cannot amplify a Gaussian entry, because `C<=1`. With `C` stuck near 0.999, the forward map differs from the Gaussian-only map by about one part in a thousand before any learning, and the Phase 2/3 report's observed departure from initialization is at most a few times `1e-6`.

## M-06. Is the model a new function class?

At any moment when every `G_ij>0` and `C` is absent, a CoNN layer represents the same maps as a dense layer. Float32 underflow removes a small fraction of edges at `d50=0.08` (about 0.2%). Values below `1e-6` remain representable but receive negligible updates. After exact release, D is a dense network whose weights arrived by a masked optimization path.

The construction is a special case of a prespecified multiplicative mask, in the same family as fixed local connectivity, annealed DropConnect-style masks, and bandwidth schedules on a graph. The repository's own build brief says it is not claiming a new architecture. This audit finds no theorem or experiment that isolates a capability outside that family.

A trainable input-dependent gate would be a different object. The implemented Gaussian is input-independent. The Phase 3 factor depends on an external context code, not on a learned address into a memory.

## M-07. What the equations do not say

A correct product rule does not imply that widening `tau` improves risk. A correct Gaussian does not imply that the supplied coordinates are the right graph. Matching row energy between D and B′ removes one magnitude explanation for an accuracy gap and leaves source assignment and higher-order mask topology entangled, which `DESIGN_DECISIONS.md` already states.

## Findings

| ID | Statement | Class |
| --- | --- | --- |
| MATH-01 | `G`, `d50`, and `W_eff = W⊙G` match the code | CONFIRMED |
| MATH-02 | `∂L/∂W = G⊙∂L/∂W_eff` holds because `G` is independent of `W` and detached | CONFIRMED |
| MATH-03 | No gradient is applied to `tau` or to coordinates | CONFIRMED |
| MATH-04 | `tau(q)=sigma/r(q)` is not a project equation | NOT TESTED |
| MATH-05 | Terminal D is a dense map; the mask is a training schedule plus, for S and the fixed-width policies, an inference mask | CONFIRMED |
| MATH-06 | Early `d50=0.08` leaves about 42% of synthetic input-layer gates below `1e-6` | CONFIRMED by direct evaluation of the formula |
| MATH-07 | The original context factor is initialized at algebraic rank at most 1 and near 0.9988 | CONFIRMED |
