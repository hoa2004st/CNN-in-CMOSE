# Raw Prediction Outputs

This report records the direct predictions made by each configured CMOSE-trained run.

- Accepted private clips predicted: 428
- Runs predicted: 18

## Raw CSV Files

| Dataset | Long format | One row per clip | Distribution | Metrics |
|---|---|---|---|---|
| Private | `outputs\model_assessment\comparison\raw_predictions\private_predictions.csv` | `outputs\model_assessment\comparison\raw_predictions\private_predictions_by_clip.csv` | `outputs\model_assessment\private\prediction_distribution.csv` | `outputs\model_assessment\private\supervised_metrics.csv` |
| CMOSE test | `outputs\model_assessment\comparison\raw_predictions\cmose_test_predictions.csv` | `outputs\model_assessment\comparison\raw_predictions\cmose_test_predictions_by_clip.csv` | `outputs\model_assessment\cmose_testset\prediction_distribution.csv` | `outputs\model_assessment\cmose_testset\supervised_metrics.csv` |

## Prediction Columns

- `predictions.csv` / `*_predictions.csv`: one row per `run` and `clip_id`, including predicted class, confidence, margin, normalized entropy, and class probabilities.
- `predictions_by_clip.csv` / `*_predictions_by_clip.csv`: one row per clip, with each run's prediction in separate columns.

## CMOSE Test Metrics

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
| tcn/ce | 0.7649 | 0.5012 | 0.5478 | 0.7423 | 0.2555 | 0.2965 |
| tcn/ordinal | 0.6249 | 0.6260 | 0.5192 | 0.6498 | 0.4193 | 0.5111 |
| tcn/weighted_ce | 0.6945 | 0.6200 | 0.5723 | 0.7085 | 0.3366 | 0.4038 |
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
| tcn/ce | 0.6603 | 0.3320 | 0.3431 | 0.6263 | 0.3723 | 0.4375 |
| tcn/ordinal | 0.4484 | 0.3246 | 0.2520 | 0.4801 | 0.6739 | 0.9728 |
| tcn/weighted_ce | 0.6685 | 0.3179 | 0.3241 | 0.6389 | 0.3967 | 0.5543 |
| transformer/ce | 0.5870 | 0.3395 | 0.3387 | 0.5815 | 0.4728 | 0.6033 |
| transformer/ordinal | 0.3098 | 0.3442 | 0.2557 | 0.3336 | 0.8071 | 1.0571 |
| transformer/weighted_ce | 0.3424 | 0.3290 | 0.2441 | 0.3820 | 0.8261 | 1.2011 |

## Prediction Distribution

| Dataset | Run | Highly Disengage | Disengage | Engage | Highly Engage |
|---|---|---:|---:|---:|---:|
| Private | i3d_mlp/ce | 0.000 | 0.030 | 0.671 | 0.299 |
| Private | i3d_mlp/ordinal | 0.026 | 0.005 | 0.171 | 0.799 |
| Private | i3d_mlp/weighted_ce | 0.047 | 0.119 | 0.252 | 0.582 |
| Private | lstm/ce | 0.152 | 0.194 | 0.414 | 0.241 |
| Private | lstm/ordinal | 0.488 | 0.236 | 0.105 | 0.171 |
| Private | lstm/weighted_ce | 0.227 | 0.220 | 0.400 | 0.154 |
| Private | openface_mlp/ce | 0.000 | 0.369 | 0.619 | 0.012 |
| Private | openface_mlp/ordinal | 0.000 | 0.666 | 0.334 | 0.000 |
| Private | openface_mlp/weighted_ce | 0.241 | 0.164 | 0.348 | 0.248 |
| Private | openface_tcn_i3d_fusion/ce | 0.491 | 0.453 | 0.028 | 0.028 |
| Private | openface_tcn_i3d_fusion/ordinal | 0.030 | 0.446 | 0.439 | 0.084 |
| Private | openface_tcn_i3d_fusion/weighted_ce | 0.357 | 0.505 | 0.033 | 0.105 |
| Private | tcn/ce | 0.014 | 0.131 | 0.815 | 0.040 |
| Private | tcn/ordinal | 0.023 | 0.061 | 0.472 | 0.444 |
| Private | tcn/weighted_ce | 0.026 | 0.112 | 0.787 | 0.075 |
| Private | transformer/ce | 0.058 | 0.154 | 0.694 | 0.093 |
| Private | transformer/ordinal | 0.098 | 0.526 | 0.236 | 0.140 |
| Private | transformer/weighted_ce | 0.124 | 0.444 | 0.301 | 0.131 |
| CMOSE test | i3d_mlp/ce | 0.014 | 0.141 | 0.782 | 0.063 |
| CMOSE test | i3d_mlp/ordinal | 0.048 | 0.257 | 0.565 | 0.130 |
| CMOSE test | i3d_mlp/weighted_ce | 0.045 | 0.273 | 0.545 | 0.138 |
| CMOSE test | lstm/ce | 0.019 | 0.106 | 0.820 | 0.056 |
| CMOSE test | lstm/ordinal | 0.077 | 0.265 | 0.428 | 0.230 |
| CMOSE test | lstm/weighted_ce | 0.045 | 0.257 | 0.502 | 0.196 |
| CMOSE test | openface_mlp/ce | 0.006 | 0.102 | 0.845 | 0.048 |
| CMOSE test | openface_mlp/ordinal | 0.000 | 0.382 | 0.522 | 0.096 |
| CMOSE test | openface_mlp/weighted_ce | 0.029 | 0.253 | 0.554 | 0.164 |
| CMOSE test | openface_tcn_i3d_fusion/ce | 0.020 | 0.118 | 0.776 | 0.085 |
| CMOSE test | openface_tcn_i3d_fusion/ordinal | 0.086 | 0.224 | 0.475 | 0.215 |
| CMOSE test | openface_tcn_i3d_fusion/weighted_ce | 0.043 | 0.269 | 0.527 | 0.161 |
| CMOSE test | tcn/ce | 0.020 | 0.120 | 0.814 | 0.046 |
| CMOSE test | tcn/ordinal | 0.060 | 0.257 | 0.499 | 0.184 |
| CMOSE test | tcn/weighted_ce | 0.030 | 0.212 | 0.590 | 0.168 |
| CMOSE test | transformer/ce | 0.016 | 0.131 | 0.796 | 0.057 |
| CMOSE test | transformer/ordinal | 0.075 | 0.283 | 0.424 | 0.217 |
| CMOSE test | transformer/weighted_ce | 0.052 | 0.259 | 0.470 | 0.219 |

## Run Metrics Files

- `i3d_mlp/ce`: `outputs\training_log\i3d_mlp\ce\metrics.json`
- `i3d_mlp/ordinal`: `outputs\training_log\i3d_mlp\ordinal\metrics.json`
- `i3d_mlp/weighted_ce`: `outputs\training_log\i3d_mlp\weighted_ce\metrics.json`
- `lstm/ce`: `outputs\training_log\lstm\ce\metrics.json`
- `lstm/ordinal`: `outputs\training_log\lstm\ordinal\metrics.json`
- `lstm/weighted_ce`: `outputs\training_log\lstm\weighted_ce\metrics.json`
- `openface_mlp/ce`: `outputs\training_log\openface_mlp\ce\metrics.json`
- `openface_mlp/ordinal`: `outputs\training_log\openface_mlp\ordinal\metrics.json`
- `openface_mlp/weighted_ce`: `outputs\training_log\openface_mlp\weighted_ce\metrics.json`
- `openface_tcn_i3d_fusion/ce`: `outputs\training_log\openface_tcn_i3d_fusion\ce\metrics.json`
- `openface_tcn_i3d_fusion/ordinal`: `outputs\training_log\openface_tcn_i3d_fusion\ordinal\metrics.json`
- `openface_tcn_i3d_fusion/weighted_ce`: `outputs\training_log\openface_tcn_i3d_fusion\weighted_ce\metrics.json`
- `tcn/ce`: `outputs\training_log\tcn\ce\metrics.json`
- `tcn/ordinal`: `outputs\training_log\tcn\ordinal\metrics.json`
- `tcn/weighted_ce`: `outputs\training_log\tcn\weighted_ce\metrics.json`
- `transformer/ce`: `outputs\training_log\transformer\ce\metrics.json`
- `transformer/ordinal`: `outputs\training_log\transformer\ordinal\metrics.json`
- `transformer/weighted_ce`: `outputs\training_log\transformer\weighted_ce\metrics.json`
