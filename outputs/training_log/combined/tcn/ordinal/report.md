# openface_tcn [Ordinal]

- Run folder: `combined/tcn/ordinal`
- Metrics file: `outputs\training_log\combined\tcn\ordinal\metrics.json`
- Accuracy: 0.4982
- Macro Accuracy: 0.5668
- Macro F1: 0.4223
- Weighted F1: 0.4835
- MAE: 0.5757
- MSE: 0.7434
- Best epoch: 34

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.16      0.54      0.24        39
       Disengage       0.31      0.60      0.41       305
          Engage       0.83      0.30      0.44      1729
   Highly Engage       0.47      0.83      0.60       932

        accuracy                           0.50      3005
       macro avg       0.44      0.57      0.42      3005
    weighted avg       0.65      0.50      0.48      3005
```