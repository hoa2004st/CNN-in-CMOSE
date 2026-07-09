**Baseline models by loss, in-domain (CMOSE), sorted by QWK.**

| Model | Loss | QWK | Macro-Accuracy | Macro-MAE | Accuracy | MAE | Cohen κ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| openface_tcn | CE | 0.537 | 0.535 | 0.547 | 0.761 | 0.256 | 0.432 |
| i3d_mlp | CE | 0.519 | 0.526 | 0.585 | 0.773 | 0.244 | 0.436 |
| i3d_mlp | Weighted CE | 0.5 | 0.611 | 0.481 | 0.685 | 0.354 | 0.412 |
| i3d_mlp | Ordinal | 0.487 | 0.635 | 0.458 | 0.64 | 0.408 | 0.383 |
| openface_tcn | Weighted CE | 0.464 | 0.597 | 0.54 | 0.626 | 0.428 | 0.352 |
| openface_transformer | Ordinal | 0.443 | 0.588 | 0.504 | 0.553 | 0.502 | 0.281 |
| openface_tcn | Ordinal | 0.436 | 0.6 | 0.522 | 0.551 | 0.512 | 0.291 |
| openface_transformer | CE | 0.434 | 0.493 | 0.665 | 0.75 | 0.279 | 0.394 |
| openface_transformer | Weighted CE | 0.4 | 0.569 | 0.575 | 0.577 | 0.486 | 0.293 |
| openface_mlp | Weighted CE | 0.399 | 0.545 | 0.583 | 0.557 | 0.499 | 0.252 |
| openface_lstm | CE | 0.387 | 0.42 | 0.719 | 0.712 | 0.314 | 0.286 |
| openface_mlp | CE | 0.369 | 0.405 | 0.744 | 0.723 | 0.299 | 0.28 |
| openface_lstm | Weighted CE | 0.329 | 0.534 | 0.673 | 0.527 | 0.563 | 0.238 |
| openface_lstm | Ordinal | 0.318 | 0.491 | 0.694 | 0.415 | 0.673 | 0.154 |
| openface_mlp | Ordinal | 0.253 | 0.348 | 0.798 | 0.493 | 0.542 | 0.118 |
