# openface_mlp [Weighted CE]

- Run folder: `openface_mlp/weighted_ce`
- Metrics file: `outputs\training_log\openface_mlp\weighted_ce\metrics.json`
- Accuracy: 0.4996
- Macro Accuracy: 0.5113
- Macro F1: 0.4100
- Weighted F1: 0.5320
- MAE: 0.5708
- MSE: 0.7183
- Best epoch: 29

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.18      0.43      0.25        35
       Disengage       0.29      0.51      0.37       221
          Engage       0.80      0.48      0.60       847
   Highly Engage       0.31      0.63      0.42       118

        accuracy                           0.50      1221
       macro avg       0.40      0.51      0.41      1221
    weighted avg       0.64      0.50      0.53      1221
```