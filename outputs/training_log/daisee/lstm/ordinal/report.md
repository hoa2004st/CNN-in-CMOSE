# openface_lstm [Ordinal]

- Run folder: `daisee/lstm/ordinal`
- Metrics file: `outputs\training_log\daisee\lstm\ordinal\metrics.json`
- Accuracy: 0.3212
- Macro Accuracy: 0.3108
- Macro F1: 0.1609
- Weighted F1: 0.2660
- MAE: 1.1155
- MSE: 2.2522
- Best epoch: 4

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.00      0.50      0.01         4
       Disengage       0.09      0.05      0.06        84
          Engage       0.61      0.01      0.02       882
   Highly Engage       0.46      0.68      0.55       814

        accuracy                           0.32      1784
       macro avg       0.29      0.31      0.16      1784
    weighted avg       0.52      0.32      0.27      1784
```