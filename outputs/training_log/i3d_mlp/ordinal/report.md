# i3d_mlp [Ordinal]

- Run folder: `i3d_mlp/ordinal`
- Metrics file: `outputs/training_log/i3d_mlp/ordinal/metrics.json`
- Accuracy: 0.6577
- Macro Accuracy: 0.6115
- Macro F1: 0.5400
- Weighted F1: 0.6774
- MAE: 0.3866
- MSE: 0.4832
- Best epoch: 22

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.31      0.49      0.38        35
       Disengage       0.42      0.60      0.50       221
          Engage       0.86      0.68      0.76       847
   Highly Engage       0.42      0.69      0.52       118

        accuracy                           0.66      1221
       macro avg       0.51      0.61      0.54      1221
    weighted avg       0.73      0.66      0.68      1221
```