# Joint evidence chain, 2026-10-28

New end-to-end model, 186,554 parameters. No old037/054 learned weights, pretrained teacher, external data, U data, class bias search or hard protection routes.

The existing audited deterministic preprocessing supplies static/dynamic amplitude, phase-pair features, timing and scales. This round does not add missing raw validity masks or recover previously averaged raw detail. Stable frequency shapes are encoded per receiver; local changes and both directions of phase pairs use shared encoders. A context vector from stable state and scale/timing statistics applies bounded feature modulation before temporal aggregation. Thirty local time slices per view retain mean, variability, peaks and learned local-probability disagreement. A joint five-class head reads the combined representation. Local probabilities have no separate activity labels and must not be interpreted as verified person detections or calibrated uncertainty.

Full and two ablations have the same parameter count: `no_context` disables early feature modulation but keeps late context; `no_spread` removes temporal variability, peak excess and local-probability variability but keeps temporal mean and context. Disabled paths have no effective gradient, so equal parameter count does not imply equal effective capacity.

Protocol: 15 epochs each, AdamW 0.0006, decay0.02, batch24 source windows each with forward/reverse view, NLL+.05 reverse consistency. Class→room→record→window weighted sampling. No added feature noise. Whole-record source calibration picks epoch, and last15 weights are also saved. Four reused leave-room folds plus three existing record-group folds; no record overlaps train/cal/test. Full uses seeds42/2026, ablations seed42:28 development fits. Full-data runs use both seeds and save source-selected and last15 checkpoints:2 further fits. Primary full architecture and seed42 are fixed before outcomes, not chosen by target accuracy.

Commands from project root:

```
python experiments/evidence_chain_20261028/train_chain.py
python experiments/evidence_chain_20261028/train_chain.py --full
python experiments/evidence_chain_20261028/export_chain.py
python tools/check_candidate.py D:/huawei2026-experiments/evidence_chain_20261028/export
python experiments/evidence_chain_20261028/export_chain.py --publish
python experiments/evidence_chain_20261028/export_chain.py --last
python tools/check_candidate.py D:/huawei2026-experiments/evidence_chain_20261028/export_last
python experiments/evidence_chain_20261028/export_chain.py --last --publish
python experiments/evidence_chain_20261028/report_chain.py
```

Outputs: `D:/huawei2026-experiments/evidence_chain_20261028/`. Published ONNX is a research model; low scores do not automatically terminate the hypothesis, and neither do they demonstrate eventual success. Review the paired ablations, training curves and per-class failures.
