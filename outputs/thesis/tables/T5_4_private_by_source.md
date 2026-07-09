**Private set (test-only): best baseline vs best proposed hybrid model by training source.**

| Train source | Best baseline (model/loss) | Baseline QWK | Baseline macro-MAE | Best proposed model (arch) | Proposed QWK | Proposed macro-acc | Proposed macro-MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CMOSE | openface_transformer/ce | 0.19 | 0.75 | I3D stream disabled, TCN_T_LSTM_TCN_T | 0.309 | 0.385 | 0.914 |
| DAiSEE | i3d_mlp/ce | 0.256 | 1.031 | I3D stream enabled, LSTM_LSTM_TCN_LSTM_LSTM | 0.306 | 0.35 | 0.966 |
| Combined | openface_transformer/ce | 0.285 | 0.997 | I3D stream enabled, T_T_T_LSTM_T | 0.379 | 0.385 | 0.877 |
