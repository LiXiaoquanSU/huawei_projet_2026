# Brescia data audit and use boundary

Source: https://zenodo.org/records/17225694 (v2), DOI10.5281/zenodo.17225694. Authors: Elena Tonini, Renato Lo Cigno, Francesco Gringoli, Marco Cominelli. License: CC BY-SA4.0, https://creativecommons.org/licenses/by-sa/4.0/ . Preserve citation, configuration and transformation provenance with derived data.

Associated v2 paper: https://opendl.ifip-tc6.org/db/conf/wons/wons2026/1571217424.pdf . The teammate's30Hz description fits older captures, not every capture: tableI includes~33ms,10ms and5ms intervals. Real MAT timestamps, when present, determine observed spacing. Single-Rx3/4-person records have no measured timestamps in their denseMAT; their time is explicitly nominal. No copying antennas or invented high-rate motion samples.

Three/four-person acquisitions are seated laptop use, not full static/slow/fast labels. `moving:false` is not proof of zero body movement. Many configuration notes mention people walking past the office door, unlike the contest's no-immediate-outside-passers condition. Preserve this source-domain difference in the audit. One extra office adds an environment, not sufficient proof of room invariance; label/configuration confounding remains.

Full0--4 three-Rx supervised concatenation is therefore not used. Train a single-real-antenna amplitude encoder, transfer compatible spatial weights to the official static route, compare against random initialization and matched source-only pretraining. External validation is separate and date-heldout;3/4 classes lack independent date support, so do not report an external0--4 validation claim. Never merge external easy samples into the official evaluation denominator.

Derived training input is `single_rx_features_v2.pt`; its amplitude transform is `.5*log1p(abs(CSI)**2)`, matching the official static route. `windows_v2.csv` records every source acquisition, timing provenance, activity flag, split and source SHA256. The initial non-v2 feature file is retained solely as a cleaning-history artifact and is not used for transfer training. Primary cleaning audit v2 retains generation-time field names; the final derived audit report labels configured antenna count separately from the single actual exported core used.

License note: "We do not redistribute raw data" does not by itself determine obligations for released model weights. CC's official FAQ explains that permission-requiring uses must satisfy license conditions and ShareAlike applies to publicly shared adaptations; whether specific weights constitute adapted material is not resolved here. No assertion that weights must, or need not, be CC BY-SA. Local research proceeds with attribution; external-trained weights remain research-only pending competition external-data terms and distribution conditions. Source: https://creativecommons.org/faq/#artificial-intelligence-and-cc-licenses . No communication to organizers has been sent.

Other candidates audited on2026-10-21, not downloaded into this campaign:

- RadioPoints WiFi-CrowdCounting: https://github.com/RadioPoints/Device-free_RF_Human_Sensing_Datasets . Author repository describes0--7 people in3 rooms, Intel5300. Better multi-room/count coverage, different acquisition representation. README says non-commercial research while repository displays GPL3; inspect data-specific terms before contest use. Candidate for transferable normalized temporal features, not direct1024-bin concatenation.
- A CSI Dataset for Wireless Human Sensing on80MHz Wi-Fi Channels: https://arxiv.org/abs/2305.03170 . Authors describe802.11ac,13subjects, multiple environments/days/hardware,23.6GB. Potential environment diversity; original channel/antenna/timestamp/count mapping must be checked before transfer.
- Exposing the CSI: https://github.com/ansresearch/exposing-the-csi . Actual synchronized video ground truth exists;160MHz,4antennas, three collectors, activity tasks. Useful for state/motion representation research but not a ready-made0--4-person dataset.
- Brescia2026 tracking/activity dataset: https://zenodo.org/records/21335109 . About200minutes,51.8GB; useful activity context but not established complete0--4 coverage. Do not download a much larger package merely because it is802.11ax.

These are source-verified candidate descriptions, not evidence that adding them raises this competition's accuracy.
