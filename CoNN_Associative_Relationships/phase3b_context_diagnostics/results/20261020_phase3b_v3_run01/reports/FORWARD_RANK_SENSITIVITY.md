# Forward versus reference attenuation rank

No model was trained for this sensitivity analysis. It reads saved initial/final checkpoints only. The original rank and matrix-difference diagnostics recompute sigmoid factors and C in float64, isolating the mathematical factorization. This supplement also evaluates C in its actual forward dtype, then converts that C to float64 before constructing A=1−C and computing SVD. Both use sigma1×1e−10 numerical-rank tolerance.

Float32 elementwise rounding can introduce tiny extra singular values into the forward matrix. Its numerical rank at this strict tolerance is therefore not synonymous with independent rank-factor capacity. H1 concerns the unquantized parameterization: identical columns restrict its attenuation to rank at most one; distinct columns can supply a second component. Neither a rounding-induced higher numerical rank nor a mathematically nonzero second component establishes learned utility. The frozen primary statistics and Level2 algorithm are unchanged.

| Gaussian cell | Phase | Mean reference sigma2/sigma1 | Mean actual-forward sigma2/sigma1 | Reference rank range | Forward rank range |
| --- | --- | --- | --- | --- | --- |
| C0 | initial | 9.5223586e-15 | 7.0827209e-15 | 1–1 | 1–1 |
| C0 | final | 1.1892259e-14 | 6.1245311e-06 | 1–1 | 2–35 |
| C1 | initial | 0.003300869 | 0.0033009078 | 2–2 | 64–64 |
| C1 | final | 0.0033007604 | 0.003300754 | 2–2 | 64–64 |
| C2 | initial | 9.4033959e-15 | 8.7736563e-15 | 1–1 | 1–1 |
| C2 | final | 2.124834e-15 | 1.9384048e-06 | 1–1 | 26–64 |
| C4 | initial | 0.0032728225 | 0.003272793 | 2–2 | 64–64 |
| C4 | final | 0.0032723429 | 0.0032723396 | 2–2 | 63–64 |

All completed contextual conditions and both retained C3 pilots are included in the CSV, including separate layers/contexts. Forward-level learning claims still use actual float32 live-support changes and output interventions, rather than reference rank alone.
