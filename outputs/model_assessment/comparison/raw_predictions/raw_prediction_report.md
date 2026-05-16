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
| i3d_mlp/ce | 0.7666 | 0.5242 | 0.5774 | 0.7492 | 0.2580 | 0.3137 |
| i3d_mlp/ordinal | 0.6577 | 0.6115 | 0.5400 | 0.6774 | 0.3866 | 0.4832 |
| i3d_mlp/weighted_ce | 0.6650 | 0.6302 | 0.5632 | 0.6851 | 0.3702 | 0.4439 |
| lstm/ce | 0.7027 | 0.4015 | 0.4289 | 0.6679 | 0.3268 | 0.3890 |
| lstm/ordinal | 0.4988 | 0.5092 | 0.4108 | 0.5310 | 0.5602 | 0.6830 |
| lstm/weighted_ce | 0.5561 | 0.4992 | 0.4274 | 0.5860 | 0.5168 | 0.6790 |
| openface_mlp/ce | 0.7437 | 0.4626 | 0.5060 | 0.7195 | 0.2801 | 0.3292 |
| openface_mlp/ordinal | 0.5356 | 0.2665 | 0.2660 | 0.5345 | 0.5201 | 0.6364 |
| openface_mlp/weighted_ce | 0.5741 | 0.5790 | 0.4771 | 0.6005 | 0.4791 | 0.5938 |
| openface_tcn_i3d_fusion/ce | 0.7469 | 0.4478 | 0.4575 | 0.7265 | 0.2875 | 0.3645 |
| openface_tcn_i3d_fusion/ordinal | 0.5405 | 0.5774 | 0.4115 | 0.5756 | 0.5430 | 0.7166 |
| openface_tcn_i3d_fusion/weighted_ce | 0.6577 | 0.6383 | 0.5436 | 0.6795 | 0.3849 | 0.4767 |
| tcn/ce | 0.7649 | 0.5862 | 0.6089 | 0.7594 | 0.2498 | 0.2793 |
| tcn/ordinal | 0.6847 | 0.6264 | 0.5696 | 0.7010 | 0.3473 | 0.4144 |
| tcn/weighted_ce | 0.6183 | 0.6103 | 0.4951 | 0.6457 | 0.4496 | 0.5987 |
| transformer/ce | 0.7379 | 0.5202 | 0.5460 | 0.7263 | 0.2924 | 0.3579 |
| transformer/ordinal | 0.6470 | 0.6028 | 0.5359 | 0.6678 | 0.3989 | 0.4971 |
| transformer/weighted_ce | 0.5831 | 0.5884 | 0.4853 | 0.6112 | 0.4758 | 0.6102 |

## Private Manual-Label Metrics

| Run | Accuracy | Macro Accuracy | F1 Macro | F1 Weighted | MAE | MSE |
|---|---:|---:|---:|---:|---:|---:|
| i3d_mlp/ce | 0.4864 | 0.2993 | 0.2674 | 0.5127 | 0.6413 | 0.9402 |
| i3d_mlp/ordinal | 0.3587 | 0.2720 | 0.1926 | 0.3952 | 0.7935 | 1.1522 |
| i3d_mlp/weighted_ce | 0.1793 | 0.2646 | 0.1119 | 0.1472 | 1.0054 | 1.4348 |
| lstm/ce | 0.6005 | 0.2460 | 0.2416 | 0.5598 | 0.4484 | 0.5516 |
| lstm/ordinal | 0.2011 | 0.2695 | 0.1542 | 0.1743 | 0.9348 | 1.2120 |
| lstm/weighted_ce | 0.1223 | 0.2186 | 0.1148 | 0.1015 | 1.1685 | 1.7935 |
| openface_mlp/ce | 0.3696 | 0.2522 | 0.2089 | 0.4010 | 0.7038 | 0.8505 |
| openface_mlp/ordinal | 0.5082 | 0.2216 | 0.2006 | 0.4958 | 0.5543 | 0.6848 |
| openface_mlp/weighted_ce | 0.1848 | 0.3096 | 0.1671 | 0.1520 | 1.0761 | 1.6685 |
| openface_tcn_i3d_fusion/ce | 0.1630 | 0.2500 | 0.0797 | 0.0865 | 0.9538 | 1.1875 |
| openface_tcn_i3d_fusion/ordinal | 0.1060 | 0.1591 | 0.0812 | 0.1406 | 1.5245 | 2.9647 |
| openface_tcn_i3d_fusion/weighted_ce | 0.2908 | 0.2617 | 0.1905 | 0.3275 | 0.8668 | 1.2147 |
| tcn/ce | 0.6087 | 0.3185 | 0.3040 | 0.5911 | 0.4755 | 0.6549 |
| tcn/ordinal | 0.6440 | 0.2643 | 0.2579 | 0.5924 | 0.3995 | 0.4864 |
| tcn/weighted_ce | 0.1793 | 0.2423 | 0.1495 | 0.2431 | 1.3886 | 2.6875 |
| transformer/ce | 0.4185 | 0.3985 | 0.3060 | 0.4610 | 0.7337 | 1.0815 |
| transformer/ordinal | 0.2717 | 0.2836 | 0.2009 | 0.2903 | 0.8478 | 1.1087 |
| transformer/weighted_ce | 0.4103 | 0.3192 | 0.2751 | 0.4586 | 0.7609 | 1.1413 |

## Prediction Distribution

| Dataset | Run | Highly Disengage | Disengage | Engage | Highly Engage |
|---|---|---:|---:|---:|---:|
| Private | i3d_mlp/ce | 0.023 | 0.042 | 0.572 | 0.362 |
| Private | i3d_mlp/ordinal | 0.012 | 0.002 | 0.369 | 0.617 |
| Private | i3d_mlp/weighted_ce | 0.016 | 0.023 | 0.089 | 0.871 |
| Private | lstm/ce | 0.014 | 0.028 | 0.836 | 0.121 |
| Private | lstm/ordinal | 0.026 | 0.787 | 0.117 | 0.070 |
| Private | lstm/weighted_ce | 0.178 | 0.526 | 0.058 | 0.238 |
| Private | openface_mlp/ce | 0.007 | 0.537 | 0.430 | 0.026 |
| Private | openface_mlp/ordinal | 0.000 | 0.009 | 0.717 | 0.273 |
| Private | openface_mlp/weighted_ce | 0.178 | 0.512 | 0.072 | 0.238 |
| Private | openface_tcn_i3d_fusion/ce | 0.000 | 0.951 | 0.049 | 0.000 |
| Private | openface_tcn_i3d_fusion/ordinal | 0.610 | 0.276 | 0.105 | 0.009 |
| Private | openface_tcn_i3d_fusion/weighted_ce | 0.026 | 0.610 | 0.208 | 0.157 |
| Private | tcn/ce | 0.068 | 0.159 | 0.750 | 0.023 |
| Private | tcn/ordinal | 0.019 | 0.098 | 0.846 | 0.037 |
| Private | tcn/weighted_ce | 0.551 | 0.019 | 0.187 | 0.243 |
| Private | transformer/ce | 0.117 | 0.439 | 0.353 | 0.091 |
| Private | transformer/ordinal | 0.056 | 0.652 | 0.222 | 0.070 |
| Private | transformer/weighted_ce | 0.133 | 0.325 | 0.402 | 0.140 |
| CMOSE test | i3d_mlp/ce | 0.016 | 0.134 | 0.796 | 0.054 |
| CMOSE test | i3d_mlp/ordinal | 0.044 | 0.256 | 0.544 | 0.156 |
| CMOSE test | i3d_mlp/weighted_ce | 0.037 | 0.308 | 0.517 | 0.138 |
| CMOSE test | lstm/ce | 0.020 | 0.102 | 0.835 | 0.044 |
| CMOSE test | lstm/ordinal | 0.059 | 0.290 | 0.429 | 0.222 |
| CMOSE test | lstm/weighted_ce | 0.055 | 0.286 | 0.482 | 0.178 |
| CMOSE test | openface_mlp/ce | 0.014 | 0.117 | 0.813 | 0.056 |
| CMOSE test | openface_mlp/ordinal | 0.007 | 0.262 | 0.679 | 0.052 |
| CMOSE test | openface_mlp/weighted_ce | 0.060 | 0.280 | 0.475 | 0.185 |
| CMOSE test | openface_tcn_i3d_fusion/ce | 0.000 | 0.156 | 0.767 | 0.077 |
| CMOSE test | openface_tcn_i3d_fusion/ordinal | 0.141 | 0.132 | 0.473 | 0.255 |
| CMOSE test | openface_tcn_i3d_fusion/weighted_ce | 0.059 | 0.263 | 0.523 | 0.156 |
| CMOSE test | tcn/ce | 0.023 | 0.165 | 0.730 | 0.082 |
| CMOSE test | tcn/ordinal | 0.031 | 0.260 | 0.552 | 0.156 |
| CMOSE test | tcn/weighted_ce | 0.088 | 0.219 | 0.517 | 0.176 |
| CMOSE test | transformer/ce | 0.022 | 0.137 | 0.760 | 0.081 |
| CMOSE test | transformer/ordinal | 0.044 | 0.261 | 0.548 | 0.147 |
| CMOSE test | transformer/weighted_ce | 0.055 | 0.260 | 0.477 | 0.208 |

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
