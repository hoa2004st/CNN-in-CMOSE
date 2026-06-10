**T6. Private set (test-only): best base vs best hybrid by training source (macro-MAE lower is better).**

| Train source | Best base (model/loss) | Base QWK | Base macro-MAE | Best hybrid (arch) | Hybrid QWK | Hybrid macro-acc | Hybrid macro-MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CMOSE | openface_transformer/ce | 0.19 | 0.75 | Hybrid (OpenFace only) TCN_T_LSTM_TCN_T | 0.309 | 0.385 | 0.914 |
| DaiSEE | i3d_mlp/ce | 0.256 | 1.031 | Hybrid + I3D LSTM_LSTM_TCN_LSTM_LSTM | 0.306 | 0.35 | 0.966 |
| Combined | openface_transformer/ce | 0.285 | 0.997 | Hybrid + I3D T_T_T_LSTM_T | 0.379 | 0.385 | 0.877 |
