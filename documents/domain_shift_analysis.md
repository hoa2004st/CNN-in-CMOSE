# Domain Shift Analysis

This report applies retained CMOSE-trained models to the private accepted subset.

- Accepted private clips analyzed after excluding `Delete` notes: 368
- Private target metrics use the manual labels CSV and exclude rows whose notes contain `Delete`.
- Source reference uses each run's saved CMOSE test confusion matrix.

## CMOSE Test Reference

| Run | Accuracy | Macro Accuracy | F1 Macro | F1 Weighted | MAE | MSE |
|---|---:|---:|---:|---:|---:|---:|
| i3d_mlp/ce | 0.7682 | 0.5368 | 0.5889 | 0.7534 | 0.2539 | 0.3014 |
| lstm/weighted_ce | 0.5700 | 0.5206 | 0.4502 | 0.5963 | 0.4881 | 0.6143 |
| openface_mlp/weighted_ce | 0.6102 | 0.5376 | 0.4928 | 0.6295 | 0.4259 | 0.5061 |
| openface_tcn_i3d_fusion/ce | 0.7699 | 0.5637 | 0.5972 | 0.7555 | 0.2531 | 0.3022 |
| temporal_cnn/weighted_ce | 0.6945 | 0.6200 | 0.5723 | 0.7085 | 0.3366 | 0.4038 |
| transformer/ce | 0.7486 | 0.4985 | 0.5441 | 0.7289 | 0.2801 | 0.3407 |

## Private Prediction Distribution

| Run | HD | DE | EG | HE | Dominant | Mean Confidence | Mean Entropy |
|---|---:|---:|---:|---:|---|---:|---:|
| i3d_mlp/ce | 0.000 | 0.030 | 0.671 | 0.299 | EG | 0.589 | 0.649 |
| lstm/weighted_ce | 0.227 | 0.220 | 0.400 | 0.154 | EG | 0.536 | 0.722 |
| openface_mlp/weighted_ce | 0.241 | 0.164 | 0.348 | 0.248 | EG | 0.444 | 0.813 |
| openface_tcn_i3d_fusion/ce | 0.491 | 0.453 | 0.028 | 0.028 | HD | 0.692 | 0.535 |
| temporal_cnn/weighted_ce | 0.026 | 0.112 | 0.787 | 0.075 | EG | 0.916 | 0.140 |
| transformer/ce | 0.058 | 0.154 | 0.694 | 0.093 | EG | 0.638 | 0.624 |

## Largest Distribution Shifts

Positive values mean the class is predicted more often on private clips than on the same run's CMOSE test predictions.

| Run | Class | Private - CMOSE Predicted | Private Proportion | CMOSE Predicted Proportion |
|---|---|---:|---:|---:|
| openface_tcn_i3d_fusion/ce | Engage | -0.748 | 0.028 | 0.776 |
| openface_tcn_i3d_fusion/ce | Highly Disengage | +0.470 | 0.491 | 0.020 |
| openface_tcn_i3d_fusion/ce | Disengage | +0.335 | 0.453 | 0.118 |
| i3d_mlp/ce | Highly Engage | +0.236 | 0.299 | 0.063 |
| openface_mlp/weighted_ce | Highly Disengage | +0.211 | 0.241 | 0.029 |
| openface_mlp/weighted_ce | Engage | -0.206 | 0.348 | 0.554 |
| temporal_cnn/weighted_ce | Engage | +0.198 | 0.787 | 0.590 |
| lstm/weighted_ce | Highly Disengage | +0.182 | 0.227 | 0.045 |
| i3d_mlp/ce | Engage | -0.112 | 0.671 | 0.782 |
| i3d_mlp/ce | Disengage | -0.110 | 0.030 | 0.141 |
| lstm/weighted_ce | Engage | -0.103 | 0.400 | 0.502 |
| transformer/ce | Engage | -0.102 | 0.694 | 0.796 |

## Cross-Model Agreement

| Metric | Value |
|---|---:|
| Mean agreement rate | 0.3305 |
| Median agreement rate | 0.2667 |
| Mean prediction entropy | 0.6775 |
| Median prediction entropy | 0.7296 |
| Mean confidence | 0.6359 |
| Fleiss' kappa | 0.0063 |
| Mean pairwise Cohen's kappa | 0.0423 |
| Mean pairwise raw agreement | 0.3305 |

Lowest pairwise Cohen's kappa values indicate the model pairs that disagree most often across private clips.

| Model A | Model B | Cohen's Kappa | Raw Agreement |
|---|---|---:|---:|
| temporal_cnn/weighted_ce | i3d_mlp/ce | -0.0472 | 0.5327 |
| transformer/ce | i3d_mlp/ce | -0.0425 | 0.4766 |
| openface_mlp/weighted_ce | i3d_mlp/ce | -0.0365 | 0.2874 |
| lstm/weighted_ce | i3d_mlp/ce | -0.0250 | 0.3037 |
| openface_mlp/weighted_ce | openface_tcn_i3d_fusion/ce | -0.0160 | 0.1963 |
| i3d_mlp/ce | openface_tcn_i3d_fusion/ce | -0.0013 | 0.0397 |
| temporal_cnn/weighted_ce | openface_tcn_i3d_fusion/ce | 0.0141 | 0.1005 |
| transformer/ce | openface_tcn_i3d_fusion/ce | 0.0196 | 0.1379 |

## Interpretation

- Manual-label metrics quantify private target-domain performance after removing clips marked `Delete`.
- Distribution shifts diagnose how each source-trained run changes behavior on private clips relative to CMOSE test predictions.
- A class is treated as unstable when it shows a large private-vs-source prediction proportion shift and/or low private confidence.

## Output Files

- `outputs\domain_shift_analysis\prediction_distribution.csv`
- `outputs\domain_shift_analysis\domain_shift_by_class.csv`
- `outputs\domain_shift_analysis\domain_shift_summary.json`
- `outputs\domain_shift_analysis\private_vs_source_predicted_shift.png`
- `outputs\dataset_analysis\private`
- `outputs\dataset_analysis\cmose`
