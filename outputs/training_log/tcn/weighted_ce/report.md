# tcn [Weighted CE]

- Run folder: `tcn/weighted_ce`
- Metrics file: `outputs/training_log/tcn/weighted_ce/metrics.json`
- Accuracy: 0.6282
- Macro Accuracy: 0.6124
- Macro F1: 0.4989
- Weighted F1: 0.6521
- MAE: 0.4333
- MSE: 0.5659
- Best epoch: 22

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.22      0.60      0.32        35
       Disengage       0.49      0.45      0.47       221
          Engage       0.84      0.66      0.74       847
   Highly Engage       0.34      0.74      0.47       118

        accuracy                           0.63      1221
       macro avg       0.47      0.61      0.50      1221
    weighted avg       0.71      0.63      0.65      1221
```