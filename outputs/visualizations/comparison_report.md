# Comparison Summary

## Best Run Per Model

- `openface_tcn_i3d_fusion`: run `openface_tcn_i3d_fusion/ce` (CE)
  Macro F1=0.5972, Macro Accuracy=0.5637, Accuracy=0.7699, Weighted F1=0.7555, Best epoch=6
- `i3d_mlp`: run `i3d_mlp/ce` (CE)
  Macro F1=0.5889, Macro Accuracy=0.5368, Accuracy=0.7682, Weighted F1=0.7534, Best epoch=6
- `temporal_cnn`: run `temporal_cnn/weighted_ce` (Weighted CE)
  Macro F1=0.5723, Macro Accuracy=0.6200, Accuracy=0.6945, Weighted F1=0.7085, Best epoch=61
- `transformer`: run `transformer/ce` (CE)
  Macro F1=0.5441, Macro Accuracy=0.4985, Accuracy=0.7486, Weighted F1=0.7289, Best epoch=25
- `openface_mlp`: run `openface_mlp/weighted_ce` (Weighted CE)
  Macro F1=0.4928, Macro Accuracy=0.5376, Accuracy=0.6102, Weighted F1=0.6295, Best epoch=95
- `lstm`: run `lstm/weighted_ce` (Weighted CE)
  Macro F1=0.4502, Macro Accuracy=0.5206, Accuracy=0.5700, Weighted F1=0.5963, Best epoch=31

## Best Run Per Loss

- `CE`: `openface_tcn_i3d_fusion` (`openface_tcn_i3d_fusion/ce`)
  Macro F1=0.5972, Macro Accuracy=0.5637, Accuracy=0.7699, Weighted F1=0.7555
- `Ordinal`: `i3d_mlp` (`i3d_mlp/ordinal`)
  Macro F1=0.5576, Macro Accuracy=0.6227, Accuracy=0.6757, Weighted F1=0.6928
- `Weighted CE`: `temporal_cnn` (`temporal_cnn/weighted_ce`)
  Macro F1=0.5723, Macro Accuracy=0.6200, Accuracy=0.6945, Weighted F1=0.7085

## Loss Comparison By Model

### i3d_mlp
- `CE`: run `i3d_mlp/ce`, Macro F1=0.5889, Macro Accuracy=0.5368, Accuracy=0.7682, Weighted F1=0.7534
- `Ordinal`: run `i3d_mlp/ordinal`, Macro F1=0.5576, Macro Accuracy=0.6227, Accuracy=0.6757, Weighted F1=0.6928
- `Weighted CE`: run `i3d_mlp/weighted_ce`, Macro F1=0.5567, Macro Accuracy=0.6255, Accuracy=0.6732, Weighted F1=0.6924

### lstm
- `Weighted CE`: run `lstm/weighted_ce`, Macro F1=0.4502, Macro Accuracy=0.5206, Accuracy=0.5700, Weighted F1=0.5963
- `CE`: run `lstm/ce`, Macro F1=0.4429, Macro Accuracy=0.4139, Accuracy=0.7052, Weighted F1=0.6754
- `Ordinal`: run `lstm/ordinal`, Macro F1=0.3964, Macro Accuracy=0.5120, Accuracy=0.4808, Weighted F1=0.5149

### openface_mlp
- `Weighted CE`: run `openface_mlp/weighted_ce`, Macro F1=0.4928, Macro Accuracy=0.5376, Accuracy=0.6102, Weighted F1=0.6295
- `CE`: run `openface_mlp/ce`, Macro F1=0.4608, Macro Accuracy=0.4163, Accuracy=0.7346, Weighted F1=0.6995
- `Ordinal`: run `openface_mlp/ordinal`, Macro F1=0.3395, Macro Accuracy=0.3696, Accuracy=0.5283, Weighted F1=0.5473

### openface_tcn_i3d_fusion
- `CE`: run `openface_tcn_i3d_fusion/ce`, Macro F1=0.5972, Macro Accuracy=0.5637, Accuracy=0.7699, Weighted F1=0.7555
- `Weighted CE`: run `openface_tcn_i3d_fusion/weighted_ce`, Macro F1=0.5542, Macro Accuracy=0.6337, Accuracy=0.6593, Weighted F1=0.6790
- `Ordinal`: run `openface_tcn_i3d_fusion/ordinal`, Macro F1=0.4591, Macro Accuracy=0.5933, Accuracy=0.5627, Weighted F1=0.5919

### temporal_cnn
- `Weighted CE`: run `temporal_cnn/weighted_ce`, Macro F1=0.5723, Macro Accuracy=0.6200, Accuracy=0.6945, Weighted F1=0.7085
- `CE`: run `temporal_cnn/ce`, Macro F1=0.5478, Macro Accuracy=0.5012, Accuracy=0.7649, Weighted F1=0.7423
- `Ordinal`: run `temporal_cnn/ordinal`, Macro F1=0.5192, Macro Accuracy=0.6260, Accuracy=0.6249, Weighted F1=0.6498

### transformer
- `CE`: run `transformer/ce`, Macro F1=0.5441, Macro Accuracy=0.4985, Accuracy=0.7486, Weighted F1=0.7289
- `Weighted CE`: run `transformer/weighted_ce`, Macro F1=0.4712, Macro Accuracy=0.5685, Accuracy=0.5799, Weighted F1=0.6099
- `Ordinal`: run `transformer/ordinal`, Macro F1=0.4441, Macro Accuracy=0.5714, Accuracy=0.5332, Weighted F1=0.5665

## Model Comparison By Loss

### CE
- `openface_tcn_i3d_fusion`: run `openface_tcn_i3d_fusion/ce`, Macro F1=0.5972, Macro Accuracy=0.5637, Accuracy=0.7699, Weighted F1=0.7555
- `i3d_mlp`: run `i3d_mlp/ce`, Macro F1=0.5889, Macro Accuracy=0.5368, Accuracy=0.7682, Weighted F1=0.7534
- `temporal_cnn`: run `temporal_cnn/ce`, Macro F1=0.5478, Macro Accuracy=0.5012, Accuracy=0.7649, Weighted F1=0.7423
- `transformer`: run `transformer/ce`, Macro F1=0.5441, Macro Accuracy=0.4985, Accuracy=0.7486, Weighted F1=0.7289
- `openface_mlp`: run `openface_mlp/ce`, Macro F1=0.4608, Macro Accuracy=0.4163, Accuracy=0.7346, Weighted F1=0.6995
- `lstm`: run `lstm/ce`, Macro F1=0.4429, Macro Accuracy=0.4139, Accuracy=0.7052, Weighted F1=0.6754

### Ordinal
- `i3d_mlp`: run `i3d_mlp/ordinal`, Macro F1=0.5576, Macro Accuracy=0.6227, Accuracy=0.6757, Weighted F1=0.6928
- `temporal_cnn`: run `temporal_cnn/ordinal`, Macro F1=0.5192, Macro Accuracy=0.6260, Accuracy=0.6249, Weighted F1=0.6498
- `openface_tcn_i3d_fusion`: run `openface_tcn_i3d_fusion/ordinal`, Macro F1=0.4591, Macro Accuracy=0.5933, Accuracy=0.5627, Weighted F1=0.5919
- `transformer`: run `transformer/ordinal`, Macro F1=0.4441, Macro Accuracy=0.5714, Accuracy=0.5332, Weighted F1=0.5665
- `lstm`: run `lstm/ordinal`, Macro F1=0.3964, Macro Accuracy=0.5120, Accuracy=0.4808, Weighted F1=0.5149
- `openface_mlp`: run `openface_mlp/ordinal`, Macro F1=0.3395, Macro Accuracy=0.3696, Accuracy=0.5283, Weighted F1=0.5473

### Weighted CE
- `temporal_cnn`: run `temporal_cnn/weighted_ce`, Macro F1=0.5723, Macro Accuracy=0.6200, Accuracy=0.6945, Weighted F1=0.7085
- `i3d_mlp`: run `i3d_mlp/weighted_ce`, Macro F1=0.5567, Macro Accuracy=0.6255, Accuracy=0.6732, Weighted F1=0.6924
- `openface_tcn_i3d_fusion`: run `openface_tcn_i3d_fusion/weighted_ce`, Macro F1=0.5542, Macro Accuracy=0.6337, Accuracy=0.6593, Weighted F1=0.6790
- `openface_mlp`: run `openface_mlp/weighted_ce`, Macro F1=0.4928, Macro Accuracy=0.5376, Accuracy=0.6102, Weighted F1=0.6295
- `transformer`: run `transformer/weighted_ce`, Macro F1=0.4712, Macro Accuracy=0.5685, Accuracy=0.5799, Weighted F1=0.6099
- `lstm`: run `lstm/weighted_ce`, Macro F1=0.4502, Macro Accuracy=0.5206, Accuracy=0.5700, Weighted F1=0.5963
