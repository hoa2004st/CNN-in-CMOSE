# openface_transformer [CE]

- Run folder: `transformer/ce`
- Metrics file: `outputs\training_log\combined\transformer\ce\metrics.json`
- Accuracy: 0.6067
- Macro Accuracy: 0.4394
- Macro F1: 0.4783
- Weighted F1: 0.5875
- MAE: 0.4097
- MSE: 0.4429
- Best epoch: 20

## Classification Report

```text
                  precision    recall  f1-score   support

Highly Disengage       0.58      0.28      0.38        39
       Disengage       0.51      0.30      0.38       305
          Engage       0.64      0.79      0.70      1729
   Highly Engage       0.54      0.38      0.45       932

        accuracy                           0.61      3005
       macro avg       0.57      0.44      0.48      3005
    weighted avg       0.59      0.61      0.59      3005
```