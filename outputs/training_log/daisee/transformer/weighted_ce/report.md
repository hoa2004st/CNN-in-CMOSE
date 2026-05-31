# openface_transformer [Weighted CE]

- Run folder: `daisee/transformer/weighted_ce`
- Metrics file: `outputs\training_log\daisee\transformer\weighted_ce\metrics.json`
- Accuracy: 0.3952
- Macro Accuracy: 0.3014
- Macro F1: 0.2343
- Weighted F1: 0.3751
- MAE: 0.7377
- MSE: 1.0168
- Best epoch: 6

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.00      0.00      0.00         4
       Disengage       0.09      0.39      0.15        84
          Engage       0.55      0.15      0.24       882
   Highly Engage       0.47      0.66      0.55       814

        accuracy                           0.40      1784
       macro avg       0.28      0.30      0.23      1784
    weighted avg       0.49      0.40      0.38      1784
```