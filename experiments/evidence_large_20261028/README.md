# Larger relation chain — 2026-10-28

10,874,074 trainable parameters, random seed42, new weights. Uses the same audited preprocessing/cache and whole-record development splits as evidence_chain_20261028. This is an expanded architecture, not a parameter-only control; architecture and capacity effects cannot be separated by this run.

The amplitude path retains three temporal scales (dynamic amplitude, residual from5-sample mean, residual from25-sample mean). Separate shared CNNs retain four frequency bands. Stable amplitude means/spreads and per-receiver scale summaries provide context. Three receiver nodes and six directed phase-pair edges exchange messages twice, retaining endpoint correspondence. Four transformer layers jointly process context and30 local tokens per receiver. A final context query pools all90 local tokens. Five-class joint readout uses mean, spread, peak, updated context and local-probability summaries. Receiver order invariance is maintained by shared operations and symmetric final pooling.

No room/activity/person-position labels, no external/U data, no old weights or output bias. Local probabilities are not validated person detectors. Raw validity masks and raw details lost by previous preprocessing are still absent. Three temporal scales are not guaranteed physical speed categories.

Protocol fixed before training:7 development fits (4 leave-room,3 record-group), one seed42,15epochs each. Same AdamW, sampling, batch24 with2 time views and consistency loss as the small chain. Source calibration chooses epoch; floor of median7 source-selected epochs sets the all48-record checkpoint. Full fit still runs15epochs and retains both checkpoints. Target scores are reported, not used to select model/epoch. All splits are reused development sets, not blind evaluation. More seeds and capacity-matched ablations remain future work.

Initial synthetic checks: receiver permutations max probability error4.47e-8; backward passed; zero input finite. Measured trained/export checks are stored inD:/huawei2026-experiments/evidence_large_20261028/.

Commands:
```
python experiments/evidence_large_20261028/train_chain.py
python experiments/evidence_large_20261028/train_chain.py --full
python experiments/evidence_large_20261028/export_chain.py
python tools/check_candidate.py D:/huawei2026-experiments/evidence_large_20261028/export
python experiments/evidence_large_20261028/export_chain.py --publish
python experiments/evidence_large_20261028/export_chain.py --last
python tools/check_candidate.py D:/huawei2026-experiments/evidence_large_20261028/export_last
python experiments/evidence_large_20261028/export_chain.py --last --publish
python experiments/evidence_large_20261028/report.py
```
