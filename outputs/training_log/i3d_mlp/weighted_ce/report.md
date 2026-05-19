# i3d_mlp [Weighted CE]

- Run folder: `i3d_mlp/weighted_ce`
- Metrics file: `outputs\training_log\i3d_mlp\weighted_ce\metrics.json`
- Accuracy: 0.6880
- Macro Accuracy: 0.6176
- Macro F1: 0.5686
- Weighted F1: 0.7043
- MAE: 0.3505
- MSE: 0.4357
- Best epoch: 16

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.34      0.46      0.39        35
       Disengage       0.43      0.64      0.51       221
          Engage       0.87      0.71      0.78       847
   Highly Engage       0.53      0.66      0.59       118

        accuracy                           0.69      1221
       macro avg       0.54      0.62      0.57      1221
    weighted avg       0.74      0.69      0.70      1221
```