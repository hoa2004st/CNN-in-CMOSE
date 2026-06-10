**T3. Best base model per cross-domain cell (QWK and Accuracy).**

| Metric | Train | Test | Best model | Loss | Score |
| --- | --- | --- | --- | --- | --- |
| QWK | CMOSE | CMOSE | openface_tcn | CE | 0.537 |
| QWK | CMOSE | DaiSEE | openface_lstm | Ordinal | 0.024 |
| QWK | CMOSE | Private | openface_transformer | CE | 0.19 |
| QWK | Combined | CMOSE | i3d_mlp | Ordinal | 0.477 |
| QWK | Combined | DaiSEE | openface_transformer | CE | 0.136 |
| QWK | Combined | Private | openface_transformer | CE | 0.285 |
| QWK | DaiSEE | CMOSE | openface_mlp | CE | 0.045 |
| QWK | DaiSEE | DaiSEE | i3d_mlp | CE | 0.166 |
| QWK | DaiSEE | Private | i3d_mlp | CE | 0.256 |
| Accuracy | CMOSE | CMOSE | i3d_mlp | CE | 0.773 |
| Accuracy | CMOSE | DaiSEE | openface_lstm | CE | 0.489 |
| Accuracy | CMOSE | Private | openface_mlp | CE | 0.59 |
| Accuracy | Combined | CMOSE | i3d_mlp | CE | 0.763 |
| Accuracy | Combined | DaiSEE | openface_mlp | CE | 0.538 |
| Accuracy | Combined | Private | i3d_mlp | CE | 0.62 |
| Accuracy | DaiSEE | CMOSE | openface_mlp | Ordinal | 0.691 |
| Accuracy | DaiSEE | DaiSEE | i3d_mlp | CE | 0.548 |
| Accuracy | DaiSEE | Private | i3d_mlp | CE | 0.615 |
