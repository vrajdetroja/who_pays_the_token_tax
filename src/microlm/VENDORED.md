# Vendored: microlm

This package is copied unchanged from the author's earlier study,
[micro-lm-ablations](https://github.com/vrajdetroja/micro-lm-ablations), commit
`930069118844b5be7e9253bb7d8e69fbf8221559` (2026-10-17). Its tests are in `tests/microlm/`
(minus `test_analysis.py`, which tests that repo's analysis scripts).

It supplies the model, optimizer, LR schedules, exact-resume checkpoints, run-config
hashing, and the from-scratch byte-level BPE. Everything specific to this study lives in
`src/recount/` and imports from here. Do not edit files in this directory: if the study
needs different behavior, put it in `recount` so the provenance above stays true.
