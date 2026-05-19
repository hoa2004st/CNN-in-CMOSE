# openface_lstm [CE]

- Run folder: `lstm/ce`
- Metrics file: `outputs\training_log\lstm\ce\metrics.json`
- Accuracy: 0.7011
- Macro Accuracy: 0.3788
- Macro F1: 0.4092
- Weighted F1: 0.6611
- MAE: 0.3227
- MSE: 0.3718
- Best epoch: 10

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.43      0.17      0.24        35
       Disengage       0.49      0.31      0.38       221
          Engage       0.74      0.90      0.82       847
   Highly Engage       0.41      0.13      0.19       118

        accuracy                           0.70      1221
       macro avg       0.52      0.38      0.41      1221
    weighted avg       0.66      0.70      0.66      1221
```