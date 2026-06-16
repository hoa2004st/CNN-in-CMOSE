**Prediction-level agreement and ensemble headroom of the base configurations (in-domain CMOSE).**

| Statistic | Value |
| --- | --- |
| Configs compared / test clips | 15 / 1221 |
| Best single-config accuracy | 77.3% |
| Majority-vote accuracy (all configs) | 73.5% |
| Oracle accuracy (>=1 config correct) | 97.6% |
| Clips every config gets wrong | 2.4% |
| Mean pairwise Cohen κ — same model, different loss | 0.391 |
| Mean pairwise Cohen κ — same loss, different model | 0.374 |
| Mean pairwise Cohen κ — different model and loss | 0.273 |
| openface_tcn/CE vs i3d_mlp/CE — Cohen κ | 0.474 |
| openface_tcn/CE vs i3d_mlp/CE — only openface_tcn correct | 7.4% |
| openface_tcn/CE vs i3d_mlp/CE — only i3d_mlp correct | 8.6% |
| openface_tcn/CE vs i3d_mlp/CE — both wrong | 15.3% |
| openface_tcn/CE + i3d_mlp/CE — pair oracle accuracy | 84.7% |
