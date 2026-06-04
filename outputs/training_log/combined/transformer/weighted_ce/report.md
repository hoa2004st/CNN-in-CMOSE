# openface_transformer [Weighted CE]

- Run folder: `transformer/weighted_ce`
- Metrics file: `outputs\training_log\combined\transformer\weighted_ce\metrics.json`
- Accuracy: 0.5255
- Macro Accuracy: 0.5068
- Macro F1: nan
- Weighted F1: nan
- MAE: 0.5241
- MSE: nan
- Best epoch: 6

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.25      0.41      0.31        39
       Disengage       0.35      0.36      0.36       305
          Engage       0.78      0.35      0.48      1729
   Highly Engage       0.46      0.91      0.61       932

        accuracy                           0.53      3005
       macro avg       0.46      0.51      0.44      3005
    weighted avg       0.63      0.53      0.51      3005
```