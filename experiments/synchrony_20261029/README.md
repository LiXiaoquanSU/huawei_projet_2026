# Synchrony versus occupancy evidence

Run`python experiments/synchrony_20261029/run.py`and repeat with`--spectral`; summarize with`report.py`. All experiment outputs are inD:/huawei2026-experiments/synchrony_20261029/. The spectral control is saved separately.

Two binary tasks distinguish synchronized cached feature pairs from29-sample temporal offset or another window7steps away in the same recording. Correlation/energy features use equal120sample crops and48dimensions per relation mode. RX amplitude pairs and amplitude-change/phase-change pairs are tested independently. Inputs remain in their original whole-record split; each negative uses the same room/count/record as its positive. Source-only StandardScaler and C selection are used. Four leave-room and three whole-record folds, plus5class count probes of summary/relation/combined features.147 logistic fits per control,294 total.

The frequency-centered control removes each receiver's frequency-common amplitude at each time; it does not average across receivers and is not claimed to isolate hardware. Both controls use the same13dimensional strength/time summary. All results are exploratory, on reused development folds. Empty-room synchrony is reported explicitly. A successful synchrony task does not establish human-specific features or usefulness of a neural synchrony auxiliary loss.

The initial original run predates addition of the spectral flag; its recorded code hash is from that initial version. The current default feature computations and split logic are unchanged. Report and predictions retain both runs without selecting one by target score. No neural model or submission file is changed by these scripts.
