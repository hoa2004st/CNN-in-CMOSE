# openface_lstm [Weighted CE]

- Run folder: `lstm/weighted_ce`
- Metrics file: `outputs\training_log\combined\lstm\weighted_ce\metrics.json`
- Accuracy: 0.5068
- Macro Accuracy: 0.5221
- Macro F1: nan
- Weighted F1: nan
- MAE: 0.5544
- MSE: nan
- Best epoch: 13

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.15      0.49      0.23        39
       Disengage       0.33      0.41      0.36       305
          Engage       0.76      0.34      0.47      1729
   Highly Engage       0.46      0.85      0.60       932

        accuracy                           0.51      3005
       macro avg       0.42      0.52      0.41      3005
    weighted avg       0.62      0.51      0.49      3005
```