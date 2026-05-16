# transformer [Weighted CE]

- Run folder: `transformer/weighted_ce`
- Metrics file: `outputs/training_log/transformer/weighted_ce/metrics.json`
- Accuracy: 0.5831
- Macro Accuracy: 0.5884
- Macro F1: 0.4853
- Weighted F1: 0.6112
- MAE: 0.4758
- MSE: 0.6102
- Best epoch: 37

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.28      0.54      0.37        35
       Disengage       0.37      0.53      0.44       221
          Engage       0.85      0.58      0.69       847
   Highly Engage       0.32      0.69      0.44       118

        accuracy                           0.58      1221
       macro avg       0.46      0.59      0.49      1221
    weighted avg       0.69      0.58      0.61      1221
```