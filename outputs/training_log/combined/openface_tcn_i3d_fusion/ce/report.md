# fusion [CE]

- Run folder: `openface_tcn_i3d_fusion/ce`
- Metrics file: `outputs\training_log\combined\openface_tcn_i3d_fusion\ce\metrics.json`
- Accuracy: 0.6363
- Macro Accuracy: 0.4131
- Macro F1: 0.4249
- Weighted F1: 0.6269
- MAE: 0.3817
- MSE: 0.4210
- Best epoch: 3

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.00      0.00      0.00        39
       Disengage       0.56      0.34      0.43       305
          Engage       0.69      0.74      0.71      1729
   Highly Engage       0.55      0.57      0.56       932

        accuracy                           0.64      3005
       macro avg       0.45      0.41      0.42      3005
    weighted avg       0.62      0.64      0.63      3005
```