# Mathematical Validation

## Gaussian relationship gate

The implementation uses

`G_ji = exp(-D_ji^2 / (2 tau^2))`,

with `D_ji = |coordinate_destination_j - coordinate_source_i|`. It clamps gates to `[0,1]` and applies a floor to tau. Source: [gates.py](../../conn_experiment_01/src/conn_exp/gates.py).

Coordinates are fixed uniformly on `[0,1]`; input and hidden coordinates are not learned and do not depend on labels or evaluation splits. Source: [geometry.py](../../conn_experiment_01/src/conn_exp/geometry.py).

The half-distance relation is `d50 = tau * sqrt(2 ln 2)`. Start and end targets are converted consistently in schedule code. Source: [schedules.py](../../conn_experiment_01/src/conn_exp/schedules.py).

## Effective weights and gradient flow

The gate is detached before multiplication:

`W_eff = W * G.detach()`.

Thus `G` is not a differentiable parameter and no gate gradient is trained. The implementation does not overwrite `W`. Source: [models.py](../../conn_experiment_01/src/conn_exp/models.py).

For an objective `L`, this gives

`dL/dW = dL/dW_eff * G`

only elementwise. The diagnostic module explicitly distinguishes gate-attenuated parameter gradients from effective-weight gradients. Source: [diagnostics.py](../../conn_experiment_01/src/conn_exp/diagnostics.py).

## Controls

- Dense A uses exact ones.
- A′ is constant on each layer and matches mean square gate energy.
- B′ permutes columns independently within each row, preserving row gate multisets and row energies.
- C permutes input coordinates once; hidden gates retain D's geometry.
- D follows the fixed local-to-global schedule and exact dense release.
- E has dense, local, and return phases; it is not a pure reversal.
- S holds the fixed local width for every step.

Sources: [gates.py](../../conn_experiment_01/src/conn_exp/gates.py), [schema.py](../../conn_experiment_01/src/conn_exp/schema.py), [schedules.py](../../conn_experiment_01/src/conn_exp/schedules.py).

## Mathematical discrepancies

- D's geometry is prescribed, not learned.
- Gate detachment means context/relationship parameters cannot update the geometric gate in the base implementation.
- B′ changes source alignment and higher-order mask topology together; it does not isolate “geometry” alone.
- "Per-row energy matched" is materially different from an identical gate matrix or identical source ordering.
- D50 record factors are a reporting convention rather than a second gate equation.
- Release is schedule path manipulation, not persistent physical connection memory.

No formal proof or formal model establishes that the gate improves generalization.

