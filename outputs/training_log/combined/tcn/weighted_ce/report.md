# openface_tcn [Weighted CE]

- Run folder: `tcn/weighted_ce`
- Metrics file: `outputs\training_log\combined\tcn\weighted_ce\metrics.json`
- Accuracy: 0.5298
- Macro Accuracy: 0.5476
- Macro F1: nan
- Weighted F1: nan
- MAE: 0.5647
- MSE: nan
- Best epoch: 26

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.10      0.54      0.17        39
       Disengage       0.44      0.43      0.43       305
          Engage       0.81      0.38      0.52      1729
   Highly Engage       0.47      0.85      0.60       932

        accuracy                           0.53      3005
       macro avg       0.45      0.55      0.43      3005
    weighted avg       0.66      0.53      0.53      3005
```