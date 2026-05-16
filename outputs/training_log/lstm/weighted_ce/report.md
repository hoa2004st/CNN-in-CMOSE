# lstm [Weighted CE]

- Run folder: `lstm/weighted_ce`
- Metrics file: `outputs/training_log/lstm/weighted_ce/metrics.json`
- Accuracy: 0.5569
- Macro Accuracy: 0.5004
- Macro F1: 0.4284
- Weighted F1: 0.5867
- MAE: 0.5152
- MSE: 0.6757
- Best epoch: 14

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.16      0.31      0.22        35
       Disengage       0.34      0.54      0.42       221
          Engage       0.82      0.57      0.67       847
   Highly Engage       0.31      0.58      0.41       118

        accuracy                           0.56      1221
       macro avg       0.41      0.50      0.43      1221
    weighted avg       0.66      0.56      0.59      1221
```