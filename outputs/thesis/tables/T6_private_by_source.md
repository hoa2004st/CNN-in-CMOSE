**T6. Private set (test-only): best base vs best hybrid by training source.**

| Train source | Best base (model/loss) | Base QWK | Best hybrid (arch) | Hybrid QWK | Hybrid acc | Hybrid macro-acc |
| --- | --- | --- | --- | --- | --- | --- |
| CMOSE | openface_transformer/ce | 0.19 | Hybrid (OpenFace only) TCN_T_LSTM_TCN_T | 0.309 | 0.555 | 0.385 |
| DaiSEE | openface_lstm/ce | 0.11 | Hybrid + I3D T_T_LSTM_LSTM_TCN | 0.337 | 0.672 | 0.346 |
| Combined | openface_transformer/ce | 0.285 | Hybrid + I3D T_TCN_LSTM_TCN_TCN | 0.403 | 0.634 | 0.39 |
