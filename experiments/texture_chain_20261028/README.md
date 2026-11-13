# Frequency/time texture experiment (started2026-10-29)

Question: does preserving joint static/dynamic2D texture before pooling provide useful evidence that earlier mean/spread state encoding discarded? The existing deterministic cache is unchanged; this does not recover original raw resolution.5,432,581 parameters. Three shared residual CNN blocks encode static amplitude, dynamic amplitude and local residual jointly. A separate shared CNN encodes both directions of phase pairs. Four final frequency bins and temporal mean/spread are retained before symmetric receiver/pair pooling. Absolute-scale summary joins only at the final classifier. No activity labels, hard presence rules or old learned weights.

Unlike the graph network, this model deliberately does not retain amplitude-to-phase endpoint correspondence. That is a limitation and a useful negative control for the dependency probe, not a complete replacement of the graph hypothesis. Multiple factors differ, so performance alone cannot isolate the benefit of preserving texture.

Three reused whole-record folds, same seed42,20epochs,AdamW0.0003 weight_decay0.02,cosine to0.00003,batch24 with2time views,NLL+.05time consistency,same class/room/record sampler. Epoch selection uses only source calibration balanced NLL. Fixed20 checkpoint also preserved. No external/U data. No random overlapping train/test windows.

Mechanism audit compares unmodified held-record predictions, coherent receiver relabeling, deliberately mismatched phase edges, phase time shift, and static features from another window in the same recording. Coherent relabeling must leave predictions unchanged. Other interventions probe dependence only: they can leave the physical data manifold and do not establish correct causal reasoning. They are not used as score-improving preprocessing.

Data/weights:D:/huawei2026-experiments/texture_chain_20261028/. Audit:D:/huawei2026-experiments/mechanism_20261029/.

Compatibility note: the inherited trainer's`mechanism.mean_local_probability_std`is not meaningful here: this model has no local probability head and returns repeated global probabilities to satisfy the old interface. Likewise`mean_absolute_modulation`is summary-vector magnitude, not a learned physical modulation strength. Do not use these legacy fields as mechanism evidence; use the dedicated audit instead.
