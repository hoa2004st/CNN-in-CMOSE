# openface_transformer [Weighted CE]

- Run folder: `combined/transformer/weighted_ce`
- Metrics file: `outputs\training_log\combined\transformer\weighted_ce\metrics.json`
- Accuracy: 0.5245
- Macro Accuracy: 0.5094
- Macro F1: 0.4415
- Weighted F1: 0.5066
- MAE: 0.5268
- MSE: 0.6346
- Best epoch: 6

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.26      0.41      0.32        39
       Disengage       0.35      0.37      0.36       305
          Engage       0.78      0.35      0.48      1729
   Highly Engage       0.46      0.91      0.61       932

        accuracy                           0.52      3005
       macro avg       0.46      0.51      0.44      3005
    weighted avg       0.63      0.52      0.51      3005
```