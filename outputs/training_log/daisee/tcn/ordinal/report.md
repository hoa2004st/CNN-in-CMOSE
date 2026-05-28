# openface_tcn [Ordinal]

- Run folder: `tcn/ordinal`
- Metrics file: `outputs\training_log\daisee\tcn\ordinal\metrics.json`
- Accuracy: 0.2321
- Macro Accuracy: 0.3759
- Macro F1: 0.1642
- Weighted F1: 0.2713
- MAE: 1.1878
- MSE: 2.1900
- Best epoch: 6

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.01      0.75      0.02         4
       Disengage       0.03      0.29      0.06        84
          Engage       0.56      0.09      0.16       882
   Highly Engage       0.47      0.38      0.42       814

        accuracy                           0.23      1784
       macro avg       0.27      0.38      0.16      1784
    weighted avg       0.49      0.23      0.27      1784
```