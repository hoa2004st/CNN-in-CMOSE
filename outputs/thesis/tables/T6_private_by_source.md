**T6. Private set (test-only): best base vs best hybrid by training source.**

| Train source | Best base (model/loss) | Base QWK | Best hybrid (arch) | Hybrid QWK | Hybrid acc | Hybrid macro-acc |
| --- | --- | --- | --- | --- | --- | --- |
| CMOSE | openface_transformer/ce | 0.19 | Hybrid (OpenFace only) T_T_TCN_TCN_T | 0.236 | 0.399 | 0.358 |
| DaiSEE | openface_lstm/ce | 0.11 | Hybrid + I3D T_T_TCN_TCN_T | 0.256 | 0.615 | 0.382 |
| Combined | openface_transformer/ce | 0.285 | Hybrid + I3D T_T_T_T_TCN | 0.365 | 0.596 | 0.377 |
