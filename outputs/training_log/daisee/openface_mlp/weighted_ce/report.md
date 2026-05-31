# openface_mlp [Weighted CE]

- Run folder: `daisee/openface_mlp/weighted_ce`
- Metrics file: `outputs\training_log\daisee\openface_mlp\weighted_ce\metrics.json`
- Accuracy: 0.2887
- Macro Accuracy: 0.2859
- Macro F1: 0.2029
- Weighted F1: 0.3486
- MAE: 0.9905
- MSE: 1.6149
- Best epoch: 12

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.01      0.25      0.02         4
       Disengage       0.04      0.32      0.07        84
          Engage       0.52      0.31      0.39       882
   Highly Engage       0.47      0.26      0.33       814

        accuracy                           0.29      1784
       macro avg       0.26      0.29      0.20      1784
    weighted avg       0.47      0.29      0.35      1784
```