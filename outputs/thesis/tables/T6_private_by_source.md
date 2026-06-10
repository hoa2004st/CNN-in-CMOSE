**T6. Private set (test-only): best base vs best hybrid by training source.**

| Train source | Best base (model/loss) | Base QWK | Best hybrid (arch) | Hybrid QWK | Hybrid acc | Hybrid macro-acc |
| --- | --- | --- | --- | --- | --- | --- |
| CMOSE | openface_transformer/ce | 0.19 | Hybrid (OpenFace only) TCN_T_LSTM_TCN_T | 0.309 | 0.555 | 0.385 |
| DaiSEE | i3d_mlp/ce | 0.256 | Hybrid + I3D LSTM_LSTM_TCN_LSTM_LSTM | 0.306 | 0.658 | 0.35 |
| Combined | openface_transformer/ce | 0.285 | Hybrid + I3D T_T_T_LSTM_T | 0.379 | 0.607 | 0.385 |
