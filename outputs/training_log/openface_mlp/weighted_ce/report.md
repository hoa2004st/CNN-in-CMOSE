# openface_mlp [Weighted CE]

- Run folder: `openface_mlp/weighted_ce`
- Metrics file: `outputs/training_log/openface_mlp/weighted_ce/metrics.json`
- Accuracy: 0.5717
- Macro Accuracy: 0.5781
- Macro F1: 0.4758
- Weighted F1: 0.5982
- MAE: 0.4816
- MSE: 0.5962
- Best epoch: 61

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.25      0.51      0.33        35
       Disengage       0.35      0.55      0.43       221
          Engage       0.83      0.56      0.67       847
   Highly Engage       0.36      0.69      0.47       118

        accuracy                           0.57      1221
       macro avg       0.45      0.58      0.48      1221
    weighted avg       0.68      0.57      0.60      1221
```