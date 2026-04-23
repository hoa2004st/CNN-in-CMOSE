# Comparison Summary

## Best Run Per Model

- `openface_i3d_temporal_fusion`: run `openface_i3d_temporal_fusion_cross_entropy` (baseline)
  Macro F1=0.5845, Accuracy=0.7527, Best epoch=7
- `temporal_cnn`: run `temporal_cnn` (baseline)
  Macro F1=0.5647, Accuracy=0.7682, Best epoch=38
- `rectangular_cnn`: run `rectangular_cnn` (baseline)
  Macro F1=0.5633, Accuracy=0.7699, Best epoch=357
- `transformer`: run `transformer` (baseline)
  Macro F1=0.5441, Accuracy=0.7445, Best epoch=30
- `lstm`: run `lstm` (baseline)
  Macro F1=0.4357, Accuracy=0.6994, Best epoch=11
- `spectral_cnn`: run `spectral_cnn` (baseline)
  Macro F1=0.4122, Accuracy=0.7404, Best epoch=10
- `paper_cnn`: run `paper_cnn_svd` (SVD)
  Macro F1=0.2189, Accuracy=0.6945, Best epoch=23

## Variant Comparison By Model

### temporal_cnn
- `baseline`: run `temporal_cnn`, Macro F1=0.5647, Accuracy=0.7682
- `weighted_cross_entropy`: run `temporal_cnn_weighted_ce`, Macro F1=0.5592, Accuracy=0.6708
- `ordinal`: run `temporal_cnn_ordinal`, Macro F1=0.5305, Accuracy=0.6257
- `focal(g=2.0)`: run `temporal_cnn_focal`, Macro F1=0.1881, Accuracy=0.1687
