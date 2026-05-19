# openface_lstm [Weighted CE]

- Run folder: `lstm/weighted_ce`
- Metrics file: `outputs\training_log\lstm\weighted_ce\metrics.json`
- Accuracy: 0.5971
- Macro Accuracy: 0.5507
- Macro F1: 0.4727
- Weighted F1: 0.6228
- MAE: 0.4578
- MSE: 0.5741
- Best epoch: 32

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.25      0.46      0.33        35
       Disengage       0.40      0.50      0.44       221
          Engage       0.83      0.62      0.71       847
   Highly Engage       0.31      0.62      0.41       118

        accuracy                           0.60      1221
       macro avg       0.45      0.55      0.47      1221
    weighted avg       0.68      0.60      0.62      1221
```