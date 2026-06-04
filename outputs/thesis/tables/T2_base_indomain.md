**T2. Base models x losses, in-domain (CMOSE -> CMOSE), sorted by QWK.**

| Model | Loss | Accuracy | Macro-Accuracy | QWK | Cohen κ | MAE | Macro-MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| openface_tcn | CE | 0.761 | 0.535 | 0.537 | 0.432 | 0.256 | 0.547 |
| i3d_mlp | Ordinal | 0.686 | 0.613 | 0.5 | 0.419 | 0.355 | 0.488 |
| i3d_mlp | CE | 0.764 | 0.508 | 0.493 | 0.418 | 0.256 | 0.625 |
| i3d_mlp | Weighted CE | 0.69 | 0.609 | 0.482 | 0.424 | 0.356 | 0.511 |
| openface_tcn | Weighted CE | 0.626 | 0.597 | 0.464 | 0.352 | 0.428 | 0.54 |
| openface_transformer | Ordinal | 0.553 | 0.588 | 0.443 | 0.281 | 0.502 | 0.504 |
| openface_tcn | Ordinal | 0.551 | 0.6 | 0.436 | 0.291 | 0.512 | 0.522 |
| openface_transformer | CE | 0.75 | 0.493 | 0.434 | 0.394 | 0.279 | 0.665 |
| openface_transformer | Weighted CE | 0.577 | 0.569 | 0.4 | 0.293 | 0.486 | 0.575 |
| openface_mlp | Weighted CE | 0.557 | 0.545 | 0.399 | 0.252 | 0.499 | 0.583 |
| openface_lstm | CE | 0.712 | 0.42 | 0.387 | 0.286 | 0.314 | 0.719 |
| openface_mlp | CE | 0.723 | 0.405 | 0.369 | 0.28 | 0.299 | 0.744 |
| openface_lstm | Weighted CE | 0.527 | 0.534 | 0.329 | 0.238 | 0.563 | 0.673 |
| openface_lstm | Ordinal | 0.415 | 0.491 | 0.318 | 0.154 | 0.673 | 0.694 |
| openface_mlp | Ordinal | 0.493 | 0.348 | 0.253 | 0.118 | 0.542 | 0.798 |
