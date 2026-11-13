# Stable evidence chain research

Research priority clarified2026-10-29: test a physically reasonable evidence chain and verify module roles;60% is a long-term performance goal, not a criterion to score-chase or reject an unfinished mechanism. Target-fold labels are not used for gradient updates or epoch selection. Architecture iteration uses these development results, so they are not a blind final test. Three record folds completed; leave-room and full-data stages are pending, not claimed completed.

Motivation: the large model fitted only539/960 source windows at15epochs. In24 inspected source windows its trained stable-state output mean absolute magnitude was1.134, versus0.269 for dynamic features; this suggests, but does not prove, early state dominance.

Architecture:11,274,732 parameters. Keep shared multiscale CNNs and directed receiver graph. Normalize state, dynamic and relation embeddings before fusion; learn initially small context/clock/message/transformer residual gains. Add local temporal convolution and independent dynamic/state classifier paths. Joint logits add0.25dynamic and0.1state logits. Auxiliary NLL weights0.15dynamic and0.05state. Time-reversal JS remains0.05. No room/activity labels or teacher/U/external data.

Training: seed42; AdamW lr0.0002, decay0.02,3epoch warmup then cosine to10% over30epochs; batch24 windows with two time views; same class/room/record/window weighted sampling and whole-record source calibration as prior rounds. Per-fold epoch chosen by source balanced NLL. Save selected and fixed30 checkpoints. Three record folds run first; leave-room folds follow. All-data epoch selection uses only source calibration. This round changes multiple factors, so causal attribution requires later ablations.

Initial synthetic receiver permutation max probability error4.47e-8; backward passed. Trained checks and resource verification required before delivery. Cache and exact data hashes are recorded inD:/huawei2026-experiments/stable_chain_20261028/protocol.json.

Commands: `python experiments/stable_chain_20261028/train_chain.py --record`, then without`--record` for all declared folds, then`--full` for all48records. Existing finished fits are not overwritten.
