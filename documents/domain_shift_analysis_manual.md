# Domain Shift Analysis

This report applies retained CMOSE-trained models to the private accepted subset.

- Accepted private clips analyzed after excluding `Delete` notes: 368
- Private target metrics use the manual labels CSV and exclude rows whose notes contain `Delete`.
- Source reference uses each run's saved CMOSE test confusion matrix.

## CMOSE Test Reference

| Run | Accuracy | Macro Accuracy | F1 Macro | F1 Weighted | MAE | MSE |
|---|---:|---:|---:|---:|---:|---:|
| i3d_mlp/ce | 0.7682 | 0.5368 | 0.5889 | 0.7534 | 0.2539 | 0.3014 |
| i3d_mlp/ordinal | 0.6757 | 0.6227 | 0.5576 | 0.6928 | 0.3669 | 0.4619 |
| i3d_mlp/weighted_ce | 0.6732 | 0.6255 | 0.5567 | 0.6924 | 0.3718 | 0.4685 |
| lstm/ce | 0.7052 | 0.4139 | 0.4429 | 0.6754 | 0.3243 | 0.3849 |
| lstm/ordinal | 0.4808 | 0.5120 | 0.3964 | 0.5149 | 0.5872 | 0.7297 |
| lstm/weighted_ce | 0.5700 | 0.5206 | 0.4502 | 0.5963 | 0.4881 | 0.6143 |
| openface_mlp/ce | 0.7346 | 0.4163 | 0.4608 | 0.6995 | 0.2858 | 0.3284 |
| openface_mlp/ordinal | 0.5283 | 0.3696 | 0.3395 | 0.5473 | 0.4988 | 0.5528 |
| openface_mlp/weighted_ce | 0.6102 | 0.5376 | 0.4928 | 0.6295 | 0.4259 | 0.5061 |
| openface_tcn_i3d_fusion/ce | 0.7699 | 0.5637 | 0.5972 | 0.7555 | 0.2531 | 0.3022 |
| openface_tcn_i3d_fusion/ordinal | 0.5627 | 0.5933 | 0.4591 | 0.5919 | 0.4922 | 0.6069 |
| openface_tcn_i3d_fusion/weighted_ce | 0.6593 | 0.6337 | 0.5542 | 0.6790 | 0.3792 | 0.4676 |
| temporal_cnn/ce | 0.7649 | 0.5012 | 0.5478 | 0.7423 | 0.2555 | 0.2965 |
| temporal_cnn/ordinal | 0.6249 | 0.6260 | 0.5192 | 0.6498 | 0.4193 | 0.5111 |
| temporal_cnn/weighted_ce | 0.6945 | 0.6200 | 0.5723 | 0.7085 | 0.3366 | 0.4038 |
| transformer/ce | 0.7486 | 0.4985 | 0.5441 | 0.7289 | 0.2801 | 0.3407 |
| transformer/ordinal | 0.5332 | 0.5714 | 0.4441 | 0.5665 | 0.5324 | 0.6765 |
| transformer/weighted_ce | 0.5799 | 0.5685 | 0.4712 | 0.6099 | 0.4816 | 0.6208 |

## Private Manual-Label Metrics

| Run | Accuracy | Macro Accuracy | F1 Macro | F1 Weighted | MAE | MSE |
|---|---:|---:|---:|---:|---:|---:|
| i3d_mlp/ce | 0.5489 | 0.2871 | 0.2489 | 0.5456 | 0.5516 | 0.7853 |
| i3d_mlp/ordinal | 0.2283 | 0.2647 | 0.1272 | 0.2257 | 0.9565 | 1.3859 |
| i3d_mlp/weighted_ce | 0.2935 | 0.3068 | 0.1996 | 0.3177 | 0.8696 | 1.2500 |
| lstm/ce | 0.3886 | 0.2816 | 0.2513 | 0.4500 | 0.8424 | 1.3533 |
| lstm/ordinal | 0.1467 | 0.2181 | 0.1397 | 0.1890 | 1.3859 | 2.6141 |
| lstm/weighted_ce | 0.4022 | 0.2838 | 0.2594 | 0.4777 | 0.9185 | 1.6902 |
| openface_mlp/ce | 0.4946 | 0.2590 | 0.2221 | 0.4980 | 0.5734 | 0.7092 |
| openface_mlp/ordinal | 0.3397 | 0.2545 | 0.1759 | 0.3643 | 0.7636 | 0.9701 |
| openface_mlp/weighted_ce | 0.3424 | 0.2812 | 0.2335 | 0.4108 | 0.9511 | 1.6250 |
| openface_tcn_i3d_fusion/ce | 0.1359 | 0.3935 | 0.1218 | 0.0861 | 1.4076 | 2.6848 |
| openface_tcn_i3d_fusion/ordinal | 0.3913 | 0.2214 | 0.2073 | 0.4433 | 0.7446 | 1.0380 |
| openface_tcn_i3d_fusion/weighted_ce | 0.1440 | 0.3243 | 0.1366 | 0.1109 | 1.3016 | 2.3505 |
| temporal_cnn/ce | 0.6603 | 0.3320 | 0.3431 | 0.6263 | 0.3723 | 0.4375 |
| temporal_cnn/ordinal | 0.4484 | 0.3246 | 0.2520 | 0.4801 | 0.6739 | 0.9728 |
| temporal_cnn/weighted_ce | 0.6685 | 0.3179 | 0.3241 | 0.6389 | 0.3967 | 0.5543 |
| transformer/ce | 0.5870 | 0.3395 | 0.3387 | 0.5815 | 0.4728 | 0.6033 |
| transformer/ordinal | 0.3098 | 0.3442 | 0.2557 | 0.3336 | 0.8071 | 1.0571 |
| transformer/weighted_ce | 0.3424 | 0.3290 | 0.2441 | 0.3820 | 0.8261 | 1.2011 |

## Private Prediction Distribution

| Run | HD | DE | EG | HE | Dominant | Mean Confidence | Mean Entropy |
|---|---:|---:|---:|---:|---|---:|---:|
| i3d_mlp/ce | 0.000 | 0.033 | 0.660 | 0.307 | EG | 0.585 | 0.654 |
| i3d_mlp/ordinal | 0.024 | 0.005 | 0.166 | 0.804 | HE | 0.793 | 0.374 |
| i3d_mlp/weighted_ce | 0.049 | 0.133 | 0.250 | 0.568 | HE | 0.604 | 0.672 |
| lstm/ce | 0.155 | 0.188 | 0.427 | 0.231 | EG | 0.527 | 0.766 |
| lstm/ordinal | 0.503 | 0.217 | 0.106 | 0.174 | HD | 0.596 | 0.678 |
| lstm/weighted_ce | 0.228 | 0.212 | 0.402 | 0.158 | EG | 0.535 | 0.725 |
| openface_mlp/ce | 0.000 | 0.372 | 0.617 | 0.011 | EG | 0.714 | 0.437 |
| openface_mlp/ordinal | 0.000 | 0.660 | 0.340 | 0.000 | DE | 1.000 | 0.000 |
| openface_mlp/weighted_ce | 0.231 | 0.155 | 0.367 | 0.247 | EG | 0.441 | 0.815 |
| openface_tcn_i3d_fusion/ce | 0.503 | 0.446 | 0.024 | 0.027 | HD | 0.690 | 0.539 |
| openface_tcn_i3d_fusion/ordinal | 0.035 | 0.440 | 0.448 | 0.076 | EG | 0.534 | 0.724 |
| openface_tcn_i3d_fusion/weighted_ce | 0.356 | 0.508 | 0.033 | 0.103 | DE | 0.756 | 0.434 |
| temporal_cnn/ce | 0.014 | 0.122 | 0.821 | 0.043 | EG | 0.883 | 0.204 |
| temporal_cnn/ordinal | 0.019 | 0.052 | 0.448 | 0.481 | HE | 0.841 | 0.266 |
| temporal_cnn/weighted_ce | 0.016 | 0.092 | 0.810 | 0.082 | EG | 0.921 | 0.133 |
| transformer/ce | 0.030 | 0.147 | 0.726 | 0.098 | EG | 0.649 | 0.610 |
| transformer/ordinal | 0.062 | 0.557 | 0.255 | 0.125 | DE | 0.600 | 0.645 |
| transformer/weighted_ce | 0.106 | 0.465 | 0.323 | 0.106 | DE | 0.577 | 0.688 |

## Largest Distribution Shifts

Positive values mean the class is predicted more often on private clips than on the same run's CMOSE test predictions.

| Run | Class | Private - CMOSE Predicted | Private Proportion | CMOSE Predicted Proportion |
|---|---|---:|---:|---:|
| openface_tcn_i3d_fusion/ce | Engage | -0.752 | 0.024 | 0.776 |
| i3d_mlp/ordinal | Highly Engage | +0.674 | 0.804 | 0.130 |
| openface_tcn_i3d_fusion/weighted_ce | Engage | -0.495 | 0.033 | 0.527 |
| openface_tcn_i3d_fusion/ce | Highly Disengage | +0.482 | 0.503 | 0.020 |
| i3d_mlp/weighted_ce | Highly Engage | +0.430 | 0.568 | 0.138 |
| lstm/ordinal | Highly Disengage | +0.426 | 0.503 | 0.077 |
| i3d_mlp/ordinal | Engage | -0.399 | 0.166 | 0.565 |
| lstm/ce | Engage | -0.393 | 0.427 | 0.820 |
| openface_tcn_i3d_fusion/ce | Disengage | +0.328 | 0.446 | 0.118 |
| lstm/ordinal | Engage | -0.322 | 0.106 | 0.428 |
| openface_tcn_i3d_fusion/weighted_ce | Highly Disengage | +0.313 | 0.356 | 0.043 |
| temporal_cnn/ordinal | Highly Engage | +0.297 | 0.481 | 0.184 |

## Cross-Model Agreement

| Metric | Value |
|---|---:|
| Mean agreement rate | 0.3151 |
| Median agreement rate | 0.3007 |
| Mean prediction entropy | 0.8360 |
| Median prediction entropy | 0.8611 |
| Mean confidence | 0.6803 |
| Fleiss' kappa | 0.0354 |
| Mean pairwise Cohen's kappa | 0.0540 |
| Mean pairwise raw agreement | 0.3151 |

Lowest pairwise Cohen's kappa values indicate the model pairs that disagree most often across private clips.

| Model A | Model B | Cohen's Kappa | Raw Agreement |
|---|---|---:|---:|
| openface_mlp/ordinal | temporal_cnn/weighted_ce | -0.1337 | 0.2473 |
| openface_mlp/ce | temporal_cnn/weighted_ce | -0.0865 | 0.4946 |
| transformer/weighted_ce | i3d_mlp/ce | -0.0777 | 0.2038 |
| transformer/ordinal | i3d_mlp/ce | -0.0592 | 0.1793 |
| lstm/ce | i3d_mlp/weighted_ce | -0.0577 | 0.2283 |
| temporal_cnn/ordinal | i3d_mlp/ordinal | -0.0555 | 0.4321 |
| transformer/weighted_ce | i3d_mlp/weighted_ce | -0.0500 | 0.1685 |
| temporal_cnn/weighted_ce | i3d_mlp/ce | -0.0441 | 0.5435 |

## Interpretation

- Manual-label metrics quantify private target-domain performance after removing clips marked `Delete`.
- Distribution shifts diagnose how each source-trained run changes behavior on private clips relative to CMOSE test predictions.
- A class is treated as unstable when it shows a large private-vs-source prediction proportion shift and/or low private confidence.

## Output Files

- `outputs\domain_shift_analysis_manual\prediction_distribution.csv`
- `outputs\domain_shift_analysis_manual\domain_shift_by_class.csv`
- `outputs\domain_shift_analysis_manual\domain_shift_summary.json`
- `outputs\domain_shift_analysis_manual\private_vs_source_predicted_shift.png`
- `outputs\dataset_analysis_manual\private`
- `outputs\dataset_analysis_manual\cmose`
