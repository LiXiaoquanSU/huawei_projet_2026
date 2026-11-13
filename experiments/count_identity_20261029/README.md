# Class-conditional room contrast

Separate controlled experiment; does not combine the count-order loss. Same texture inference architecture, parameters, seed42, initialization,20epoch schedule, sampling and data splits as texture_chain_20261028. Only an embedding return and auxiliary training objective are added. Inference still uses the original five-class head and no room IDs/references.

Positive pairs: same count, different explicitly known rooms, fit records only. Negative pairs: different counts, same room. Unknown-room records are excluded from positives; they are not assumed to be different physical rooms. Average forward/reverse representations and normalize; temperature0.2; contrastive loss weight0.05. Anchors require at least one valid positive and one negative. NLL and0.05time reversal consistency unchanged. No test/calibration examples in contrastive pairs.

This tests count-conditional invariance rather than marginal domain adversarial alignment. It may still discard useful context or align incompatible activities; useful learning is a hypothesis, not established. Label/site confounds and only two rooms with2—4people severely limit identifiability. Three reused record folds run first. Source calibration balanced NLL selects checkpoint; fixed20 also saved. Neither ranking nor contrastive accuracy can substitute for count accuracy or platform score.

Outputs:D:/huawei2026-experiments/count_identity_20261029/. All records and failed runs retained.
