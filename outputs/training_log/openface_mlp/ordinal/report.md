# openface_mlp [Ordinal]

- Run folder: `openface_mlp/ordinal`
- Metrics file: `outputs\training_log\openface_mlp\ordinal\metrics.json`
- Accuracy: 0.5471
- Macro Accuracy: 0.3388
- Macro F1: 0.3389
- Weighted F1: 0.5556
- MAE: 0.4840
- MSE: 0.5479
- Best epoch: 35

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.22      0.06      0.09        35
       Disengage       0.23      0.31      0.26       221
          Engage       0.71      0.66      0.68       847
   Highly Engage       0.31      0.33      0.32       118

        accuracy                           0.55      1221
       macro avg       0.37      0.34      0.34      1221
    weighted avg       0.57      0.55      0.56      1221
```