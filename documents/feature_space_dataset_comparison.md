# CMOSE vs Accepted Private Feature-Space Comparison

This comparison uses extracted feature files only: OpenFace CSV files and I3D `.npy` embeddings. The private side is filtered by `data/private/accepted.csv`; no labels or downstream model outputs are used.

## Overall Answer

- OpenFace distributions are strongly shifted when using all extractor rows: mean standardized Wasserstein 0.955, mean KS 0.495.
- I3D distributions show a comparable marginal shift and clear centroid movement: centroid cosine distance 0.204, centroid RMS shift 1.248, mean standardized Wasserstein 0.971.
- The private set is not just a smaller sample of the same feature distribution; the largest OpenFace gaps are geometric/pose related and the I3D embedding centroid also moves.

## Dataset Coverage

- Private accepted clips from manifest: 428 of 808.
- Private accepted OpenFace CSV files: 428; sampled files: 428; sampled rows: 54784.
- CMOSE OpenFace CSV files: 12197; sampled files: 472; sampled rows: 60000.
- Common OpenFace feature columns: 709; column order identical: True.
- Private I3D vectors: 428; CMOSE I3D vectors: 12197; dimension: 1024.

## OpenFace Group Distances

| group | features | mean Wasserstein z | median Wasserstein z | mean KS | mean abs mean shift z |
|---|---:|---:|---:|---:|---:|
| face_landmark_2d | 136 | 1.678 | 1.706 | 0.882 | 1.664 |
| eye_landmark_2d | 112 | 1.663 | 1.699 | 0.869 | 1.651 |
| pdm | 40 | 0.771 | 0.759 | 0.381 | 0.361 |
| eye_landmark_3d | 168 | 0.615 | 0.496 | 0.308 | 0.478 |
| face_landmark_3d | 204 | 0.571 | 0.489 | 0.299 | 0.391 |
| gaze | 8 | 0.422 | 0.425 | 0.248 | 0.342 |
| head_pose | 6 | 0.238 | 0.191 | 0.206 | 0.114 |
| au_presence | 18 | 0.231 | 0.231 | 0.083 | 0.231 |
| au_intensity | 17 | 0.169 | 0.138 | 0.071 | 0.157 |

## OpenFace Extraction Quality

| dataset | mean confidence | median confidence | confidence >= 0.80 | confidence >= 0.95 | success rate |
|---|---:|---:|---:|---:|---:|
| private | 0.834 | 0.980 | 0.856 | 0.640 | 0.863 |
| CMOSE | 0.923 | 0.980 | 0.960 | 0.703 | 0.963 |

## High-Confidence OpenFace Sensitivity

This repeats the OpenFace comparison after filtering to `success == 1` and `confidence >= 0.80` before sampling.

- Private valid sampled rows: 54136; CMOSE valid sampled rows: 60000.
- Valid-only mean standardized Wasserstein: 0.979; mean KS: 0.497.

| group | features | mean Wasserstein z | median Wasserstein z | mean KS | mean abs mean shift z |
|---|---:|---:|---:|---:|---:|
| face_landmark_2d | 136 | 1.741 | 1.765 | 0.898 | 1.740 |
| eye_landmark_2d | 112 | 1.722 | 1.763 | 0.885 | 1.722 |
| pdm | 40 | 0.852 | 0.829 | 0.395 | 0.429 |
| eye_landmark_3d | 168 | 0.604 | 0.459 | 0.300 | 0.503 |
| face_landmark_3d | 204 | 0.560 | 0.425 | 0.288 | 0.419 |
| gaze | 8 | 0.507 | 0.510 | 0.246 | 0.392 |
| head_pose | 6 | 0.414 | 0.350 | 0.196 | 0.264 |
| au_presence | 18 | 0.241 | 0.224 | 0.089 | 0.241 |
| au_intensity | 17 | 0.208 | 0.176 | 0.083 | 0.204 |

## I3D Distances

- Raw centroid cosine distance: 0.204.
- Raw centroid L2 distance: 2.925.
- Pooled-z centroid L2 distance: 39.951.
- Pooled-z centroid RMS shift per dimension: 1.248.
- Mean / median standardized Wasserstein: 0.971 / 0.712.
- Mean / median KS statistic: 0.402 / 0.377.

## Interpretation Notes

- Standardized Wasserstein is computed per feature after pooled z-scoring; values are in standard-deviation units.
- KS is the maximum empirical CDF gap per feature; 0 means identical marginal distributions and 1 means fully separated.
- OpenFace pixel-coordinate groups can reflect camera framing/resolution and face scale, not only behavior.
- The report intentionally avoids classifier/domain-prediction accuracy because that would introduce another model.
