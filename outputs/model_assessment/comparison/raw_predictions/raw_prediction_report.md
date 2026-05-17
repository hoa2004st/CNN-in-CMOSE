# Raw Prediction Outputs

This report records the direct predictions made by each configured CMOSE-trained run.

- Accepted private clips predicted: 428
- Runs predicted: 18

## Raw CSV Files

| Dataset | Long format | One row per clip | Distribution | Metrics |
|---|---|---|---|---|
| Private | `outputs/model_assessment/comparison/raw_predictions/private_predictions.csv` | `outputs/model_assessment/comparison/raw_predictions/private_predictions_by_clip.csv` | `outputs/model_assessment/private/prediction_distribution.csv` | `outputs/model_assessment/private/supervised_metrics.csv` |
| CMOSE test | `outputs/model_assessment/comparison/raw_predictions/cmose_test_predictions.csv` | `outputs/model_assessment/comparison/raw_predictions/cmose_test_predictions_by_clip.csv` | `outputs/model_assessment/cmose_testset/prediction_distribution.csv` | `outputs/model_assessment/cmose_testset/supervised_metrics.csv` |

## Prediction Columns

- `predictions.csv` / `*_predictions.csv`: one row per `run` and `clip_id`, including predicted class, confidence, margin, normalized entropy, and class probabilities.
- `predictions_by_clip.csv` / `*_predictions_by_clip.csv`: one row per clip, with each run's prediction in separate columns.

## CMOSE Test Metrics

| Run | Accuracy | Macro Accuracy | F1 Macro | F1 Weighted | MAE | MSE |
|---|---:|---:|---:|---:|---:|---:|
| i3d_mlp/ce | 0.7723 | 0.5392 | 0.5960 | 0.7549 | 0.2473 | 0.2883 |
| i3d_mlp/ordinal | 0.6077 | 0.6173 | 0.5044 | 0.6355 | 0.4472 | 0.5684 |
| i3d_mlp/weighted_ce | 0.6880 | 0.6176 | 0.5686 | 0.7043 | 0.3505 | 0.4357 |
| lstm/ce | 0.7011 | 0.3788 | 0.4092 | 0.6611 | 0.3227 | 0.3718 |
| lstm/ordinal | 0.4373 | 0.5247 | 0.3805 | 0.4675 | 0.6413 | 0.8116 |
| lstm/weighted_ce | 0.5971 | 0.5507 | 0.4727 | 0.6228 | 0.4578 | 0.5741 |
| openface_mlp/ce | 0.7248 | 0.3851 | 0.4241 | 0.6760 | 0.2973 | 0.3415 |
| openface_mlp/ordinal | 0.5471 | 0.3388 | 0.3389 | 0.5556 | 0.4840 | 0.5479 |
| openface_mlp/weighted_ce | 0.4996 | 0.5113 | 0.4100 | 0.5320 | 0.5708 | 0.7183 |
| openface_tcn_i3d_fusion/ce | 0.7453 | 0.4318 | 0.4396 | 0.7200 | 0.2981 | 0.3980 |
| openface_tcn_i3d_fusion/ordinal | 0.5586 | 0.5707 | 0.4776 | 0.5873 | 0.4816 | 0.5651 |
| openface_tcn_i3d_fusion/weighted_ce | 0.6486 | 0.6139 | 0.5268 | 0.6732 | 0.4021 | 0.5119 |
| tcn/ce | 0.7576 | 0.4990 | 0.5428 | 0.7368 | 0.2604 | 0.2981 |
| tcn/ordinal | 0.6167 | 0.6214 | 0.5063 | 0.6451 | 0.4373 | 0.5569 |
| tcn/weighted_ce | 0.6282 | 0.6124 | 0.4989 | 0.6521 | 0.4333 | 0.5659 |
| transformer/ce | 0.7518 | 0.4931 | 0.5411 | 0.7325 | 0.2744 | 0.3301 |
| transformer/ordinal | 0.5266 | 0.5879 | 0.4456 | 0.5588 | 0.5405 | 0.6863 |
| transformer/weighted_ce | 0.5823 | 0.5831 | 0.4781 | 0.6118 | 0.4767 | 0.6110 |

## Private Manual-Label Metrics

| Run | Accuracy | Macro Accuracy | F1 Macro | F1 Weighted | MAE | MSE |
|---|---:|---:|---:|---:|---:|---:|
| i3d_mlp/ce | 0.4728 | 0.2972 | 0.2432 | 0.4962 | 0.6658 | 0.9973 |
| i3d_mlp/ordinal | 0.1304 | 0.2491 | 0.0628 | 0.0521 | 1.0761 | 1.5435 |
| i3d_mlp/weighted_ce | 0.2120 | 0.2761 | 0.1310 | 0.1993 | 0.9701 | 1.3995 |
| lstm/ce | 0.6712 | 0.2490 | 0.2259 | 0.5849 | 0.3641 | 0.4348 |
| lstm/ordinal | 0.2310 | 0.2740 | 0.1813 | 0.2368 | 0.9049 | 1.1984 |
| lstm/weighted_ce | 0.3342 | 0.2403 | 0.2055 | 0.3851 | 0.7880 | 1.0543 |
| openface_mlp/ce | 0.7065 | 0.2632 | 0.2360 | 0.6002 | 0.3234 | 0.3832 |
| openface_mlp/ordinal | 0.3967 | 0.2245 | 0.1864 | 0.4248 | 0.7065 | 0.9130 |
| openface_mlp/weighted_ce | 0.2147 | 0.3079 | 0.1510 | 0.1783 | 0.9293 | 1.2283 |
| openface_tcn_i3d_fusion/ce | 0.1712 | 0.2383 | 0.0973 | 0.1225 | 0.9511 | 1.1957 |
| openface_tcn_i3d_fusion/ordinal | 0.3859 | 0.2592 | 0.1961 | 0.4186 | 0.7391 | 0.9891 |
| openface_tcn_i3d_fusion/weighted_ce | 0.3424 | 0.2808 | 0.2156 | 0.3832 | 0.7826 | 1.0543 |
| tcn/ce | 0.5842 | 0.3099 | 0.3219 | 0.5766 | 0.4484 | 0.5136 |
| tcn/ordinal | 0.6522 | 0.2812 | 0.2721 | 0.5983 | 0.4212 | 0.5679 |
| tcn/weighted_ce | 0.6332 | 0.3067 | 0.3055 | 0.6065 | 0.4565 | 0.6522 |
| transformer/ce | 0.6495 | 0.3166 | 0.3150 | 0.6001 | 0.3967 | 0.4891 |
| transformer/ordinal | 0.1712 | 0.3466 | 0.1890 | 0.1225 | 0.9701 | 1.2582 |
| transformer/weighted_ce | 0.3587 | 0.3886 | 0.2811 | 0.4020 | 0.8098 | 1.1630 |

## Prediction Distribution

| Dataset | Run | Highly Disengage | Disengage | Engage | Highly Engage |
|---|---|---:|---:|---:|---:|
| Private | i3d_mlp/ce | 0.033 | 0.026 | 0.542 | 0.400 |
| Private | i3d_mlp/ordinal | 0.005 | 0.005 | 0.014 | 0.977 |
| Private | i3d_mlp/weighted_ce | 0.007 | 0.033 | 0.114 | 0.846 |
| Private | lstm/ce | 0.002 | 0.051 | 0.932 | 0.014 |
| Private | lstm/ordinal | 0.049 | 0.692 | 0.175 | 0.084 |
| Private | lstm/weighted_ce | 0.016 | 0.407 | 0.343 | 0.234 |
| Private | openface_mlp/ce | 0.000 | 0.000 | 0.984 | 0.016 |
| Private | openface_mlp/ordinal | 0.000 | 0.514 | 0.486 | 0.000 |
| Private | openface_mlp/weighted_ce | 0.047 | 0.857 | 0.075 | 0.021 |
| Private | openface_tcn_i3d_fusion/ce | 0.000 | 0.876 | 0.072 | 0.051 |
| Private | openface_tcn_i3d_fusion/ordinal | 0.002 | 0.565 | 0.397 | 0.035 |
| Private | openface_tcn_i3d_fusion/weighted_ce | 0.005 | 0.526 | 0.285 | 0.185 |
| Private | tcn/ce | 0.012 | 0.231 | 0.724 | 0.033 |
| Private | tcn/ordinal | 0.068 | 0.044 | 0.864 | 0.023 |
| Private | tcn/weighted_ce | 0.056 | 0.065 | 0.783 | 0.096 |
| Private | transformer/ce | 0.035 | 0.100 | 0.843 | 0.021 |
| Private | transformer/ordinal | 0.121 | 0.727 | 0.037 | 0.114 |
| Private | transformer/weighted_ce | 0.168 | 0.442 | 0.304 | 0.086 |
| CMOSE test | i3d_mlp/ce | 0.016 | 0.129 | 0.799 | 0.057 |
| CMOSE test | i3d_mlp/ordinal | 0.062 | 0.289 | 0.462 | 0.187 |
| CMOSE test | i3d_mlp/weighted_ce | 0.038 | 0.270 | 0.571 | 0.120 |
| CMOSE test | lstm/ce | 0.011 | 0.115 | 0.844 | 0.030 |
| CMOSE test | lstm/ordinal | 0.082 | 0.319 | 0.337 | 0.262 |
| CMOSE test | lstm/weighted_ce | 0.052 | 0.229 | 0.524 | 0.196 |
| CMOSE test | openface_mlp/ce | 0.011 | 0.080 | 0.883 | 0.026 |
| CMOSE test | openface_mlp/ordinal | 0.007 | 0.245 | 0.644 | 0.104 |
| CMOSE test | openface_mlp/weighted_ce | 0.068 | 0.320 | 0.418 | 0.194 |
| CMOSE test | openface_tcn_i3d_fusion/ce | 0.000 | 0.125 | 0.786 | 0.088 |
| CMOSE test | openface_tcn_i3d_fusion/ordinal | 0.055 | 0.373 | 0.428 | 0.143 |
| CMOSE test | openface_tcn_i3d_fusion/weighted_ce | 0.061 | 0.282 | 0.514 | 0.143 |
| CMOSE test | tcn/ce | 0.018 | 0.132 | 0.801 | 0.049 |
| CMOSE test | tcn/ordinal | 0.066 | 0.275 | 0.468 | 0.191 |
| CMOSE test | tcn/weighted_ce | 0.079 | 0.167 | 0.545 | 0.208 |
| CMOSE test | transformer/ce | 0.013 | 0.133 | 0.795 | 0.058 |
| CMOSE test | transformer/ordinal | 0.081 | 0.276 | 0.412 | 0.231 |
| CMOSE test | transformer/weighted_ce | 0.058 | 0.261 | 0.471 | 0.210 |

## Run Metrics Files

- `i3d_mlp/ce`: `outputs/training_log/i3d_mlp/ce/metrics.json`
- `i3d_mlp/ordinal`: `outputs/training_log/i3d_mlp/ordinal/metrics.json`
- `i3d_mlp/weighted_ce`: `outputs/training_log/i3d_mlp/weighted_ce/metrics.json`
- `lstm/ce`: `outputs/training_log/lstm/ce/metrics.json`
- `lstm/ordinal`: `outputs/training_log/lstm/ordinal/metrics.json`
- `lstm/weighted_ce`: `outputs/training_log/lstm/weighted_ce/metrics.json`
- `openface_mlp/ce`: `outputs/training_log/openface_mlp/ce/metrics.json`
- `openface_mlp/ordinal`: `outputs/training_log/openface_mlp/ordinal/metrics.json`
- `openface_mlp/weighted_ce`: `outputs/training_log/openface_mlp/weighted_ce/metrics.json`
- `openface_tcn_i3d_fusion/ce`: `outputs/training_log/openface_tcn_i3d_fusion/ce/metrics.json`
- `openface_tcn_i3d_fusion/ordinal`: `outputs/training_log/openface_tcn_i3d_fusion/ordinal/metrics.json`
- `openface_tcn_i3d_fusion/weighted_ce`: `outputs/training_log/openface_tcn_i3d_fusion/weighted_ce/metrics.json`
- `tcn/ce`: `outputs/training_log/tcn/ce/metrics.json`
- `tcn/ordinal`: `outputs/training_log/tcn/ordinal/metrics.json`
- `tcn/weighted_ce`: `outputs/training_log/tcn/weighted_ce/metrics.json`
- `transformer/ce`: `outputs/training_log/transformer/ce/metrics.json`
- `transformer/ordinal`: `outputs/training_log/transformer/ordinal/metrics.json`
- `transformer/weighted_ce`: `outputs/training_log/transformer/weighted_ce/metrics.json`
