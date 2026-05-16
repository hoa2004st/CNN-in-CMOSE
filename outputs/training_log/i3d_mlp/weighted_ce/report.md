# i3d_mlp [Weighted CE]

- Run folder: `i3d_mlp/weighted_ce`
- Metrics file: `outputs/training_log/i3d_mlp/weighted_ce/metrics.json`
- Accuracy: 0.6650
- Macro Accuracy: 0.6302
- Macro F1: 0.5632
- Weighted F1: 0.6851
- MAE: 0.3702
- MSE: 0.4439
- Best epoch: 13

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.36      0.46      0.40        35
       Disengage       0.41      0.70      0.52       221
          Engage       0.88      0.66      0.76       847
   Highly Engage       0.49      0.70      0.58       118

        accuracy                           0.67      1221
       macro avg       0.54      0.63      0.56      1221
    weighted avg       0.75      0.67      0.69      1221
```