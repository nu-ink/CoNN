# Experiment 04 Mathematics

**PROPOSED — NOT IMPLEMENTED.**

Let input nodes `i` and destination units `j` have coordinates or latent relationship strengths `D_ji`. The proposed gate is:

`G_ji(tau) = exp(-D_ji^2 / (2 tau^2))`.

For a weight matrix `W`:

`W_eff = W * G.detach()`.

The gate remains detached until a future learning-geometry extension is separately preregistered.

## Proposed invariants

- Dense A: `G_ji = 1`.
- Uniform A′: `G_ji = sqrt(mean(G_B^2))`.
- Scrambled C: preserve each row's gate multiset and selected energy while changing source alignment.
- Aligned B: derive gates from ground-truth relationship coordinates.
- All conditions: use the same parameter count, optimizer, LR grid, step budget and initialization stream.

## Proposed analyses

Compare gate geometry, per-row energy, effective-weight magnitude, gradients, ablation loss delta and final prediction. The ablation and gradient analyses are diagnostic, not a replacement for the preregistered outcome.

