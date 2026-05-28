# openface_tcn [Weighted CE]

- Run folder: `tcn/weighted_ce`
- Metrics file: `outputs\training_log\daisee\tcn\weighted_ce\metrics.json`
- Accuracy: 0.2775
- Macro Accuracy: 0.2394
- Macro F1: 0.1898
- Weighted F1: 0.3233
- MAE: 0.9893
- MSE: 1.5566
- Best epoch: 8

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.00      0.00      0.00         4
       Disengage       0.04      0.40      0.08        84
          Engage       0.52      0.16      0.25       882
   Highly Engage       0.48      0.39      0.43       814

        accuracy                           0.28      1784
       macro avg       0.26      0.24      0.19      1784
    weighted avg       0.48      0.28      0.32      1784
```