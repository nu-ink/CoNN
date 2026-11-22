# Paste into Cursor Agent

Read `CURSOR_BUILD_BRIEF.md` in full and implement CoNN Experiment 01 according to it. The purpose is a controlled, reproducible PyTorch research experiment—not a product or a demonstration designed to make the proposed method win.

Start with T01–T03: scaffold, synthetic AND task (with optional XOR), coordinate/gate definitions, all invariant unit tests. Then implement T04–T07 in order, running tests after each task. Implement all seven controls and ensure D vs row-wise shuffled B' is the pre-specified primary comparison. Do not replace row-wise shuffle with a global shuffle, redraw shuffles each step, leak test labels into coordinates, change the optimizer without recording it, or use unpaired random seeds. SGD (no momentum, no weight decay) is the initial optimizer. Include exact dense final phase, fixed step-based schedule and explicit validation/test isolation. Keep any proposed design deviation visible with a reason before applying it.

Build functioning code and unit tests, not just a plan. Provide terminal commands and an exact results directory structure. Do a short smoke run first; if the environment supports it, run five pilot seeds after all tests pass. Never invent experimental results. After implementation, summarize changed files, run commands, passed tests, results or blockers, and next tasks.
