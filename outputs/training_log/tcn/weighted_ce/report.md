# tcn [Weighted CE]

- Run folder: `tcn/weighted_ce`
- Metrics file: `outputs/training_log/tcn/weighted_ce/metrics.json`
- Accuracy: 0.6183
- Macro Accuracy: 0.6103
- Macro F1: 0.4951
- Weighted F1: 0.6457
- MAE: 0.4496
- MSE: 0.5987
- Best epoch: 13

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.19      0.60      0.29        35
       Disengage       0.44      0.53      0.48       221
          Engage       0.85      0.63      0.73       847
   Highly Engage       0.37      0.68      0.48       118

        accuracy                           0.62      1221
       macro avg       0.46      0.61      0.50      1221
    weighted avg       0.71      0.62      0.65      1221
```