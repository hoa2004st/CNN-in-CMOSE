# Baseline vs Best Naive

- Primary ranking metric: `f1_macro`

## Best baseline run
- Run: `cmose_baseline_paper`
- File: `outputs\cmose_baseline_paper\metrics.json`
- accuracy: `0.7223587223587223`
- macro_accuracy: `0.429120032509863`
- f1_macro: `0.445572327540369`
- f1_weighted: `0.6902973911163721`
- mae: `0.29565929565929566`

## Best naive run
- Run: `openface_tcn_i3d_fusion/ce`
- Model: `openface_tcn_i3d_fusion`
- Loss: `cross_entropy`
- File: `outputs\openface_tcn_i3d_fusion\ce\metrics.json`
- accuracy: `0.7698607698607699`
- macro_accuracy: `0.5637376174084232`
- f1_macro: `0.5972155073759607`
- f1_weighted: `0.7555015643930407`
- mae: `nan`

## Delta (baseline - naive)
- accuracy: `-0.04750204750204756`
- macro_accuracy: `-0.13461758489856018`
- f1_macro: `-0.15164317983559172`
- f1_weighted: `-0.06520417327666861`
- mae: `nan`
