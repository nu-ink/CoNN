# Historical precision estimate

Existing Phase3 predictions only; no new training or validation candidate selection. Independent confusion calculations were used.

dense: mean -0.00009255; paired subject SD 0.00394236; seed SD 0.00176987; descriptive bootstrap 95% CI [-0.00172922, +0.00158275], width 0.00331197. Approximate normal zero-exclusion threshold 0.00168618; rough 80% normal-power effect 0.00241022. These are conditional engineering estimates, not a prospective power guarantee.

gaussian: mean -0.00041791; paired subject SD 0.00520815; seed SD 0.00285288; descriptive bootstrap 95% CI [-0.00263591, +0.00172669], width 0.00436260. Approximate normal zero-exclusion threshold 0.00222756; rough 80% normal-power effect 0.00318407. These are conditional engineering estimates, not a prospective power guarantee.

All 21 subjects have all six ground-truth classes: True. Thus supported-class and historical six-class macro-F1 coincide here. The +.005 endpoint is retained. Five seeds do not create additional independent subjects; overlapping CV training sets, historical selection and unknown future variance limit precision.
