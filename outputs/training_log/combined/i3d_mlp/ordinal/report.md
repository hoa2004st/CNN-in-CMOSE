# i3d_mlp [Ordinal]

- Run folder: `combined/i3d_mlp/ordinal`
- Metrics file: `outputs\training_log\combined\i3d_mlp\ordinal\metrics.json`
- Accuracy: 0.5221
- Macro Accuracy: 0.5299
- Macro F1: 0.4339
- Weighted F1: 0.5354
- MAE: 0.5268
- MSE: 0.6313
- Best epoch: 3

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.17      0.54      0.26        39
       Disengage       0.29      0.53      0.37       305
          Engage       0.65      0.51      0.57      1729
   Highly Engage       0.53      0.53      0.53       932

        accuracy                           0.52      3005
       macro avg       0.41      0.53      0.43      3005
    weighted avg       0.57      0.52      0.54      3005
```