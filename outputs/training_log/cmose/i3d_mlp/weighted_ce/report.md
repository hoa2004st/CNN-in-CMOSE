# i3d_mlp [Weighted CE]

- Run folder: `i3d_mlp/weighted_ce`
- Metrics file: `outputs\training_log\cmose\i3d_mlp\weighted_ce\metrics.json`
- Accuracy: 0.6904
- Macro Accuracy: 0.6087
- Macro F1: nan
- Weighted F1: nan
- MAE: 0.3563
- MSE: nan
- Best epoch: 21

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.30      0.46      0.36        35
       Disengage       0.46      0.62      0.53       221
          Engage       0.86      0.73      0.79       847
   Highly Engage       0.48      0.63      0.54       118

        accuracy                           0.69      1221
       macro avg       0.53      0.61      0.56      1221
    weighted avg       0.74      0.69      0.71      1221
```