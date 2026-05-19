# i3d_mlp [CE]

- Run folder: `i3d_mlp/ce`
- Metrics file: `outputs\training_log\i3d_mlp\ce\metrics.json`
- Accuracy: 0.7723
- Macro Accuracy: 0.5392
- Macro F1: 0.5960
- Weighted F1: 0.7549
- MAE: 0.2473
- MSE: 0.2883
- Best epoch: 11

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.63      0.34      0.44        35
       Disengage       0.61      0.43      0.51       221
          Engage       0.80      0.92      0.86       847
   Highly Engage       0.77      0.46      0.57       118

        accuracy                           0.77      1221
       macro avg       0.70      0.54      0.60      1221
    weighted avg       0.76      0.77      0.75      1221
```