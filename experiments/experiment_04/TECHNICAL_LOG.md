# Technical log

- First isolated test suite: 28 passed. Existing relevant regressions: 32 passed.
- First smoke attempt `run_20261121T031540_242324Z_45abfa0e` failed its gate audit
  before any fit. A float32 layer mean-square reduction differed beyond 2e-7
  even though the aligned and scrambled row multisets matched exactly.
- Changed audit accumulation only to float64; row energy tolerance 1e-12,
  uniform energy tolerance 2e-7 (uniform scalar is still constructed in float32).
  Added the actual smoke seed/schedule as a regression fixture. No DGP,
  model, gate values, optimizer, thresholds or confirmatory configuration changed.
  Failed run directory retained. Later runs record updated source hashes.
