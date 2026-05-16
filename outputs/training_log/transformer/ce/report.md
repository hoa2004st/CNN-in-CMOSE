# transformer [CE]

- Run folder: `transformer/ce`
- Metrics file: `outputs/training_log/transformer/ce/metrics.json`
- Accuracy: 0.7379
- Macro Accuracy: 0.5202
- Macro F1: 0.5460
- Weighted F1: 0.7263
- MAE: 0.2924
- MSE: 0.3579
- Best epoch: 31

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.44      0.34      0.39        35
       Disengage       0.59      0.45      0.51       221
          Engage       0.80      0.87      0.83       847
   Highly Engage       0.49      0.42      0.45       118

        accuracy                           0.74      1221
       macro avg       0.58      0.52      0.55      1221
    weighted avg       0.72      0.74      0.73      1221
```