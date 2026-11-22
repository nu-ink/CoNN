# Test and completion report

| Suite | Passed | Failures | Errors | Skipped |
| --- | --- | --- | --- | --- |
| Experiment 04 | 29 | 0 | 0 | 0 |
| Existing relevant regressions | 32 | 0 | 0 | 0 |


New suite command, from repository root:

```powershell
python -m pytest experiments/experiment_04/tests -q -p no:cacheprovider --junitxml=experiments/experiment_04/results/unit_junit.xml
```

Relevant historical suite, from `conn_experiment_01`:

```powershell
python -m pytest tests/test_gates.py tests/test_schedules.py tests/test_models.py tests/test_data.py tests/test_diagnostics.py tests/test_reproducibility.py -q -p no:cacheprovider --junitxml=../experiments/experiment_04/results/regression_junit.xml
```

Reports: [unit XML](../../../experiments/experiment_04/results/unit_junit.xml), [regression XML](../../../experiments/experiment_04/results/regression_junit.xml).
This is a targeted regression suite for reused behavior, not the entire historical project suite.

Coverage includes shape/range/zero/monotonicity, all-step row multisets and energy,
uniform energy, weight multiplication, numerical gradients and detach, exact release,
invalid configs, DGP reproducibility and interaction, independent split examples,
noise, label consistency, absence of label inputs to coordinates/permutation,
matching parameter/init/data/order/optimizer state, finite short fitting, known metrics,
unique outputs, bitwise training replay and injected failure/interruption records.

One real smoke attempt failed before fitting on float32 audit accumulation; it is preserved
as `run_20261121T031540_242324Z_45abfa0e`. Float64 diagnostic accumulation fixed the
technical issue without changing any gate, configuration or success threshold. The
fixture was added to the tests. See [technical log](../../../experiments/experiment_04/TECHNICAL_LOG.md).
Intentional exception fixtures in temporary directories are passing tests, not failed studies.

The subsequent smoke completed eight fits and 24 endpoint metrics. Pilot completed
16 fits and deliberately left its test split unread. Confirmatory completed 80 fits
and 240 endpoint metrics, all finite. Artifact verification recomputes saved prediction
metrics, checks 160 checkpoints, validates pairing and original-file hashes, and writes
[integrity.json](../../../experiments/experiment_04/results/run_20261121T031743_424624Z_7ac51c80/integrity.json).

Eight Mermaid files parsed in pinned Mermaid 11.12.1. All referenced Python symbols
were verified via AST. Seven flowcharts rendered to SVG/PNG; the standard statechart
was parsed only. See [validation](diagrams/validation.json). No formal model checking.
No VS Code Agent notification tool was exposed; the final written response is the completion notification.
