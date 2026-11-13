# Class failure audit

Run with project Python, from repository root:

```
python experiments/class_failure_20261026/run.py
python experiments/class_failure_20261026/presence.py
```

Requires existing frozen-fold evidence and strength predictions on D:. Writes only to `D:/huawei2026-experiments/class_failure_20261026/`. No neural backbone changes, no target-label hyperparameter selection, no ONNX export. Read `docs/CLASS_FAILURE_20261026.md` for limitations and results. The 64 fits are lightweight logistic regressions, not 64 full model training runs.
