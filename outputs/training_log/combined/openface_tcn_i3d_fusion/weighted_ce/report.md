# fusion [Weighted CE]

- Run folder: `openface_tcn_i3d_fusion/weighted_ce`
- Metrics file: `outputs\training_log\combined\openface_tcn_i3d_fusion\weighted_ce\metrics.json`
- Accuracy: 0.5770
- Macro Accuracy: 0.5499
- Macro F1: 0.4812
- Weighted F1: 0.5833
- MAE: 0.4672
- MSE: 0.5644
- Best epoch: 8

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.21      0.51      0.30        39
       Disengage       0.39      0.45      0.42       305
          Engage       0.73      0.53      0.62      1729
   Highly Engage       0.50      0.70      0.59       932

        accuracy                           0.58      3005
       macro avg       0.46      0.55      0.48      3005
    weighted avg       0.62      0.58      0.58      3005
```