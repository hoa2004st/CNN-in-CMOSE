# Domain Shift Analysis

This report applies retained CMOSE-trained models to the private accepted subset.

- Accepted private clips analyzed: 428
- Target labels are unavailable, so findings diagnose prediction behavior, confidence, and distribution shift rather than target accuracy.
- Source reference uses each run's saved CMOSE test confusion matrix.

## CMOSE Test Reference

| Run | Accuracy | Macro Accuracy | F1 Macro |
|---|---:|---:|---:|
| i3d_mlp/ce | 0.7682 | 0.5368 | 0.5889 |
| lstm/weighted_ce | 0.5700 | 0.5206 | 0.4502 |
| openface_mlp/weighted_ce | 0.6102 | 0.5376 | 0.4928 |
| openface_tcn_i3d_fusion/ce | 0.7699 | 0.5637 | 0.5972 |
| temporal_cnn/weighted_ce | 0.6945 | 0.6200 | 0.5723 |
| transformer/ce | 0.7486 | 0.4985 | 0.5441 |

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

## Interpretation

- The analysis should be read as domain-shift diagnosis, not target-domain performance evaluation.
- A class is treated as unstable when it shows a large private-vs-source prediction proportion shift and/or low private confidence.
- Minority CMOSE classes remain especially hard to interpret without private labels; large HD/HE swings should be discussed as hypothesis-generating evidence.

## Output Files

- `outputs/domain_shift_analysis/private_predictions.csv`
- `outputs/domain_shift_analysis/prediction_distribution.csv`
- `outputs/domain_shift_analysis/domain_shift_by_class.csv`
- `outputs/domain_shift_analysis/domain_shift_summary.json`
- `outputs/domain_shift_analysis/private_prediction_distribution.png`
- `outputs/domain_shift_analysis/private_vs_source_predicted_shift.png`
