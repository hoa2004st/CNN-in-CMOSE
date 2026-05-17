# transformer [Weighted CE]

- Run folder: `transformer/weighted_ce`
- Metrics file: `outputs/training_log/transformer/weighted_ce/metrics.json`
- Accuracy: 0.5823
- Macro Accuracy: 0.5831
- Macro F1: 0.4781
- Weighted F1: 0.6118
- MAE: 0.4767
- MSE: 0.6110
- Best epoch: 37

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.25      0.51      0.34        35
       Disengage       0.37      0.53      0.44       221
          Engage       0.86      0.58      0.69       847
   Highly Engage       0.32      0.70      0.44       118

        accuracy                           0.58      1221
       macro avg       0.45      0.58      0.48      1221
    weighted avg       0.70      0.58      0.61      1221
```