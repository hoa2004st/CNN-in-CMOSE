# fusion [Weighted CE]

- Run folder: `openface_tcn_i3d_fusion/weighted_ce`
- Metrics file: `outputs/training_log/openface_tcn_i3d_fusion/weighted_ce/metrics.json`
- Accuracy: 0.6577
- Macro Accuracy: 0.6383
- Macro F1: 0.5436
- Weighted F1: 0.6795
- MAE: 0.3849
- MSE: 0.4767
- Best epoch: 6

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.26      0.54      0.36        35
       Disengage       0.43      0.63      0.51       221
          Engage       0.88      0.66      0.75       847
   Highly Engage       0.45      0.72      0.55       118

        accuracy                           0.66      1221
       macro avg       0.51      0.64      0.54      1221
    weighted avg       0.74      0.66      0.68      1221
```