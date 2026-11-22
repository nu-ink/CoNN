# Limitations and recommendation

- One designed, favorable five-group DGP; no real-world transfer or distribution-shift study.
- Ten independent seeds; limited precision and distributional assumptions for inference.
- Exact sign-flip inference assumes symmetry; it is not valid for every possible mean-zero distribution.
- The primary gain is 1.133 pp. Neither its point estimate nor the lower confidence bound
  reaches the 2 pp practical criterion. The interval still includes effects slightly above 2 pp.
- Uniform-control evidence does not survive Holm adjustment. Scrambling changes topology
  as well as source alignment, so a uniquely geometric causal mechanism is not established.
- Raw initial weights match; effective initial magnitudes and gradient scaling vary with gates.
  Initial metrics are diagnostics, not causal identification of initialization versus dynamics.
- Fixed budget overfits noisy labels; no validation selection or early stopping was used.
  A different shared, preregistered regularization protocol could change the result.
- Hidden-unit coordinates are a prior, not ground-truth learned semantic locations.
- Negative-control permutation can preserve chance local relationships. A null result there
  does not establish equivalence, and datasets are intentionally paired across task variants.
- Immediate-release persistence is uncertain after multiplicity adjustment.
- Runtime baseline retains ones multiplications. Dense gates do not imply sparse acceleration.
  Sampled process RSS does not isolate model memory or a true process peak.
- No Git revision exists; checkout was waived. Source hashes and a local snapshot provide provenance.
- Local preregistration is not an independently timestamped public registration.
- Mermaid syntax and source references were checked; no formal model checking was performed.
  Installed pstack needs unavailable TLC/mise; its bootstrap would install software, which was prohibited.

Recommendation: **do not advance directly to learned associations on this evidence**.
First run a separately preregistered replication of fixed geometry across new interaction
families, with a shared validation-selected budget and an adequately powered uniform
comparison. Do not reuse these test outcomes to claim a new confirmatory success.
Nothing here tests general intelligence, associative memory, consciousness or transformer replacement.
