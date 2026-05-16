# i3d_mlp [CE]

- Run folder: `i3d_mlp/ce`
- Metrics file: `outputs/training_log/i3d_mlp/ce/metrics.json`
- Accuracy: 0.7666
- Macro Accuracy: 0.5242
- Macro F1: 0.5774
- Weighted F1: 0.7492
- MAE: 0.2580
- MSE: 0.3137
- Best epoch: 6

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.58      0.31      0.41        35
       Disengage       0.63      0.47      0.54       221
          Engage       0.80      0.91      0.85       847
   Highly Engage       0.71      0.40      0.51       118

        accuracy                           0.77      1221
       macro avg       0.68      0.52      0.58      1221
    weighted avg       0.75      0.77      0.75      1221
```