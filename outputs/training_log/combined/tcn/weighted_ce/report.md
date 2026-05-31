# openface_tcn [Weighted CE]

- Run folder: `combined/tcn/weighted_ce`
- Metrics file: `outputs\training_log\combined\tcn\weighted_ce\metrics.json`
- Accuracy: 0.5291
- Macro Accuracy: 0.5563
- Macro F1: 0.4318
- Weighted F1: 0.5262
- MAE: 0.5521
- MSE: 0.7411
- Best epoch: 14

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.14      0.62      0.23        39
       Disengage       0.37      0.39      0.38       305
          Engage       0.79      0.39      0.52      1729
   Highly Engage       0.47      0.84      0.60       932

        accuracy                           0.53      3005
       macro avg       0.44      0.56      0.43      3005
    weighted avg       0.64      0.53      0.53      3005
```