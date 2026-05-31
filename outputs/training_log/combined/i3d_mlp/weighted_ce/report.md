# i3d_mlp [Weighted CE]

- Run folder: `combined/i3d_mlp/weighted_ce`
- Metrics file: `outputs\training_log\combined\i3d_mlp\weighted_ce\metrics.json`
- Accuracy: 0.5285
- Macro Accuracy: 0.5491
- Macro F1: 0.4573
- Weighted F1: 0.5238
- MAE: 0.5394
- MSE: 0.6819
- Best epoch: 11

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.27      0.51      0.35        39
       Disengage       0.29      0.48      0.36       305
          Engage       0.82      0.37      0.51      1729
   Highly Engage       0.47      0.84      0.61       932

        accuracy                           0.53      3005
       macro avg       0.46      0.55      0.46      3005
    weighted avg       0.65      0.53      0.52      3005
```