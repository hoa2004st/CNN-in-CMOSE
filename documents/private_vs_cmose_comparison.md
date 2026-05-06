# Private vs CMOSE Comparison

## OpenFace
- Private files: 808
- CMOSE files: 12197
- Common feature columns: 709

### OpenFace Column Names
- Private sample file: first_quarter_0001_10s.csv
- CMOSE sample file: video10_100_person0.csv
- Private columns: 714
- CMOSE columns: 714
- Missing from private: 0
- Extra in private: 0
- Column order identical: True
- Private sampled rows: 50000
- CMOSE sampled rows: 50000
- Mean Wasserstein distance (common features): 50966.709600
- Group Wasserstein distances:
  - au: 237.087528
  - eye_landmark_2d: 128757.507500
  - face_landmark_2d: 194.272859
  - gaze: 158.853730
  - landmark_3d: 184.869150
  - other: 177.572520
  - pose: 201.648703

## I3D
- Private vectors: 428
- CMOSE vectors (from final_data_1.json): 11902
- Feature dimension: 1024
- Centroid cosine distance: 0.203903
- Composite domain gap score: 18559.034187
