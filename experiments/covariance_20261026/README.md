# Normalized temporal covariance round

`run.py` builds deterministic features from the existing exact preprocessing cache and trains 16 new CNNs: four held-out rooms × two distinct source calibration record sets × two random seeds. Outputs live on `D:/huawei2026-experiments/covariance_20261026/`.

Each view produces eight 30×30 images: amplitude covariance, amplitude-difference covariance, phase-pair covariance and phase-difference covariance, each pooled over receivers/pairs by mean and standard deviation. Time centering and energy normalization suppress static offsets and global gain. No claim is made that these operations remove all environmental effects, or that covariance rank is the number of people. Static occupancy information can also be lost.

The classifier is a three-layer CNN with global mean/std pooling and a five-way head. Training uses 40 epochs, AdamW (0.001, weight decay 0.02), cosine decay, class/room balanced sampling, and forward/reverse probability consistency. Selection uses only source calibration NLL after epoch 5; the unrestricted selected epoch is also recorded. Five epochs is a declared heuristic, not proof of adequate learning. Every epoch logs source fit and calibration confusion matrices; target predictions are evaluated only after source selection.

Partition 0 preserves the previous manifest. Partition 1 uses the second hash-ranked eligible record of each class. The two models overlap in their training records, so their errors are not independent. All target-room records and all unknown-room records remain excluded from proxy training and calibration. These rooms have been reused for development; this is not fresh blind evaluation.

Run `python experiments/covariance_20261026/run.py`, then `python experiments/covariance_20261026/report.py`. No automatic publication or replacement of the platform 0.54 model.
