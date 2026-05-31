# openface_mlp [CE]

- Run folder: `combined/openface_mlp/ce`
- Metrics file: `outputs\training_log\combined\openface_mlp\ce\metrics.json`
- Accuracy: 0.6190
- Macro Accuracy: 0.3986
- Macro F1: 0.4391
- Weighted F1: 0.5585
- MAE: 0.3933
- MSE: 0.4180
- Best epoch: 38

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.90      0.23      0.37        39
       Disengage       0.61      0.25      0.35       305
          Engage       0.62      0.92      0.74      1729
   Highly Engage       0.63      0.20      0.30       932

        accuracy                           0.62      3005
       macro avg       0.69      0.40      0.44      3005
    weighted avg       0.62      0.62      0.56      3005
```