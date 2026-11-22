# Gradient-only initialization sweep

Fixed seed400, fold0, first historical paired training batch; initial weights and batches identical across candidates. Dense/Gaussian step0 gates, all contexts. No validation scores used. Local derivatives differ from loss gradients. Parameter fractions use the active lookup row; inactive contexts are not counted as rounding failures. Raw CSV contains all42 measurements.

G0 U=-6.0: gradient ratio range 1.07453e-05–3.1087e-05; reaches1e-3 in 0/6 fixtures; live forward ΔC maximum 0.

G1 U=-5.0: gradient ratio range 2.89324e-05–8.39878e-05; reaches1e-3 in 0/6 fixtures; live forward ΔC maximum 5.96046e-08.

G2 U=-4.5: gradient ratio range 4.72489e-05–0.000137563; reaches1e-3 in 0/6 fixtures; live forward ΔC maximum 0.

G3 U=-4.0: gradient ratio range 7.68716e-05–0.000224286; reaches1e-3 in 0/6 fixtures; live forward ΔC maximum 5.96046e-08.

G4 U=-3.0: gradient ratio range 0.000198966–0.000583005; reaches1e-3 in 0/6 fixtures; live forward ΔC maximum 2.38419e-07.

G5 U=-2.0: gradient ratio range 0.000453125–0.00140061; reaches1e-3 in 1/6 fixtures; live forward ΔC maximum 1.43051e-06.

G6 U=-1.0: gradient ratio range 0.000835967–0.00284571; reaches1e-3 in 5/6 fixtures; live forward ΔC maximum 5.126e-06.

C2/C4 remain U=-4.5. No candidate is selected from this sweep. One-step logit effects distinguish gate-only changes from concurrent weight optimization. Stronger initializations, if desired, require a separate amendment.
