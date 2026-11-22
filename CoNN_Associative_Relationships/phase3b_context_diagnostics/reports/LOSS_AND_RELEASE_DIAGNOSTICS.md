# Loss and dense-release diagnostics

All values below are measured checkpoint evaluations, averaged equally across35fits and three contexts. These pooled-window losses/accuracies differ from the primary equal-subject macro-F1 endpoint. Pre/post evaluations use bitwise-identical weights at optimizer step2240; final values additionally include960updates. No missing step or fit is interpolated.

| Condition | Pre loss | Post loss | Final loss | Gate-switch loss change | Subsequent optimization loss change |
| --- | --- | --- | --- | --- | --- |
| dense_C0 | 0.72309180 | 0.72309180 | 0.78966532 | +0.00000000 | +0.06657352 |
| dense_C1 | 0.73160056 | 0.73160056 | 0.80340589 | +0.00000000 | +0.07180533 |
| dense_C2 | 0.72630095 | 0.72630095 | 0.80509296 | +0.00000000 | +0.07879200 |
| dense_C4 | 0.73003114 | 0.73003114 | 0.80281329 | +0.00000000 | +0.07278215 |
| dense_masked | 0.72542359 | 0.72542359 | 0.79702308 | +0.00000000 | +0.07159949 |
| gaussian_C0 | 0.65301847 | 0.68439766 | 0.73875837 | +0.03137919 | +0.05436072 |
| gaussian_C1 | 0.66189288 | 0.69630358 | 0.71016464 | +0.03441070 | +0.01386106 |
| gaussian_C2 | 0.65945292 | 0.69187339 | 0.70755728 | +0.03242047 | +0.01568389 |
| gaussian_C4 | 0.65619365 | 0.68868805 | 0.71836076 | +0.03249440 | +0.02967271 |
| gaussian_constant | 0.65375575 | 0.68580463 | 0.70717029 | +0.03204888 | +0.02136566 |
| gaussian_frozen | 0.65517320 | 0.68704588 | 0.70748127 | +0.03187269 | +0.02043538 |
| gaussian_masked | 0.65240406 | 0.68386828 | 0.70244664 | +0.03146422 | +0.01857836 |
| gaussian_shuffled | 0.65492178 | 0.68633345 | 0.70614719 | +0.03141167 | +0.01981374 |

The loss chart uses observed curve checkpoints only, averaged over completed fits/contexts. It does not replace the subject endpoint or establish that widening is generally helpful or harmful.
