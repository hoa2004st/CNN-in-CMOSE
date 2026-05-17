# fusion [Weighted CE]

- Run folder: `openface_tcn_i3d_fusion/weighted_ce`
- Metrics file: `outputs/training_log/openface_tcn_i3d_fusion/weighted_ce/metrics.json`
- Accuracy: 0.6486
- Macro Accuracy: 0.6139
- Macro F1: 0.5268
- Weighted F1: 0.6732
- MAE: 0.4021
- MSE: 0.5119
- Best epoch: 6

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.23      0.49      0.31        35
       Disengage       0.41      0.64      0.50       221
          Engage       0.88      0.65      0.75       847
   Highly Engage       0.46      0.68      0.55       118

        accuracy                           0.65      1221
       macro avg       0.49      0.61      0.53      1221
    weighted avg       0.74      0.65      0.67      1221
```