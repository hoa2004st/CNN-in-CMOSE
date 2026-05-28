# i3d_mlp [Weighted CE]

- Run folder: `i3d_mlp/weighted_ce`
- Metrics file: `outputs\training_log\daisee\i3d_mlp\weighted_ce\metrics.json`
- Accuracy: 0.3845
- Macro Accuracy: 0.2861
- Macro F1: 0.2564
- Weighted F1: 0.4331
- MAE: 0.8380
- MSE: 1.3772
- Best epoch: 2

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.00      0.00      0.00         4
       Disengage       0.08      0.37      0.12        84
          Engage       0.53      0.35      0.42       882
   Highly Engage       0.55      0.42      0.48       814

        accuracy                           0.38      1784
       macro avg       0.29      0.29      0.26      1784
    weighted avg       0.52      0.38      0.43      1784
```