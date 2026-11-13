# Static phase information versus capacity

5,761,925parameters, two versions with identical weights at initialization. Full uses384static relative-phase-curvature descriptors; zero uses identical modules with a zero vector. The new branch receives raw descriptors, normalized per-window descriptors and their mean/spread; projects to128dimensions and joins the existing texture/phase/summary readout. No room ID at inference. Model is invariant to coherent receiver reorder because curvature is pair-symmetric; synthetic probability error2.98e-8, backward passed.

Exact same seed42,splits,sampler,AdamW0.0003,20epochs,batch24doubleviews,NLL+.05reversal consistency. No order or identity auxiliary loss in this comparison. Source calibration balanced NLL selects weights; fixed20 also retained. Three record folds for each condition are trained first. No selection by target performance. Equal parameter count does not imply equal effective capacity: zero-input branch can only learn constants. Compare both versions and previous texture baseline.

New raw feature: unit inter-RX phasor r(f), then r(f+d)r(f-d)conj(r(f))^2 with d8/16. Remove ideal constant/linear phase ambiguity without centering out temporal mean. Mask invalid subcarriers and average600frames/32frequencygroups; retain real,absoluteimag,magnitude, pairmean/std. It may encode static room geometry rather than occupancy. Training and tests use exactly the same extraction; no target labels or target baseline used. Static features are unchanged by time reversal.

Outputs:D:/huawei2026-experiments/static_texture_20261029/; raw feature extraction:D:/huawei2026-experiments/static_phase_20261029/. Room/full fits and any export remain pending until executed; no existing ONNX was overwritten.
