# lstm [CE]

- Run folder: `lstm/ce`
- Metrics file: `outputs/training_log/lstm/ce/metrics.json`
- Accuracy: 0.7027
- Macro Accuracy: 0.4015
- Macro F1: 0.4289
- Weighted F1: 0.6679
- MAE: 0.3268
- MSE: 0.3890
- Best epoch: 12

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.33      0.23      0.27        35
       Disengage       0.53      0.30      0.38       221
          Engage       0.75      0.90      0.82       847
   Highly Engage       0.39      0.18      0.24       118

        accuracy                           0.70      1221
       macro avg       0.50      0.40      0.43      1221
    weighted avg       0.66      0.70      0.67      1221
```