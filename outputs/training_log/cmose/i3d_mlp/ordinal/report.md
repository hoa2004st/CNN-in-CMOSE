# i3d_mlp [Ordinal]

- Run folder: `cmose/i3d_mlp/ordinal`
- Metrics file: `outputs\training_log\cmose\i3d_mlp\ordinal\metrics.json`
- Accuracy: 0.6077
- Macro Accuracy: 0.6173
- Macro F1: 0.5044
- Weighted F1: 0.6355
- MAE: 0.4472
- MSE: 0.5684
- Best epoch: 11

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.24      0.51      0.32        35
       Disengage       0.39      0.62      0.48       221
          Engage       0.88      0.59      0.71       847
   Highly Engage       0.39      0.75      0.51       118

        accuracy                           0.61      1221
       macro avg       0.47      0.62      0.50      1221
    weighted avg       0.73      0.61      0.64      1221
```