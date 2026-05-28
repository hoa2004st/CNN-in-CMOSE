# openface_tcn [CE]

- Run folder: `tcn/ce`
- Metrics file: `outputs\training_log\combined\tcn\ce\metrics.json`
- Accuracy: 0.6200
- Macro Accuracy: 0.4447
- Macro F1: 0.4927
- Weighted F1: 0.5972
- MAE: 0.3960
- MSE: 0.4286
- Best epoch: 14

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.71      0.31      0.43        39
       Disengage       0.58      0.27      0.37       305
          Engage       0.64      0.81      0.72      1729
   Highly Engage       0.56      0.39      0.46       932

        accuracy                           0.62      3005
       macro avg       0.62      0.44      0.49      3005
    weighted avg       0.61      0.62      0.60      3005
```