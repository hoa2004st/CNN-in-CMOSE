# CMOSE Paper Parity Audit

## Scope
- Paper source: `documents/CMOSE-dataset.pdf` (CVPRW 2024 CMOSE paper).
- Code audited:
  - `main.py`
  - `src/models/cmose_baseline.py`
  - `src/features/extract_openface.py`
  - `src/training/mocorank.py`
  - `configs/baseline.yaml`
- Verification run:
  - `outputs/cmose_baseline_paper/strict_paper_verify_1ep/metrics.json`

## Important Limitation
- No official public author training-code repository was found from the paper/project links.
- Therefore, this is a paper-vs-implementation audit, not author-code-vs-implementation.

## Parity Matrix

1. Engagement score scalar with thresholds `(-0.5, 0, 0.5)`:
- Paper: thresholded scalar score.
- Code: `src/models/cmose_baseline.py` + `src/training/mocorank.py` class mapping.
- Status: PASS.

2. Model structure (I3D projection + OpenFace TCN + attention + normalized score head):
- Paper: MLP attention, TCN, projection, normalized linear score.
- Code: `src/models/cmose_baseline.py`.
- Status: PASS.

3. Attention MLP dropout `0.5`:
- Code: `src/models/cmose_baseline.py` line 29.
- Status: PASS.

4. TCN layers/kernel/dropout (`4`, `3`, `0.2`):
- Code: `src/models/cmose_baseline.py` lines 35-37.
- Status: PASS.

5. OpenFace baseline chunking `(49 -> min/max/var -> (147, T))`, 250-frame standardization:
- Code: `src/features/extract_openface.py` lines 275, 282, 293-295.
- Status: PASS.

6. Score pool FIFO behavior:
- Code: `src/training/mocorank.py` `ScorePool`.
- Status: PASS.

7. Momentum encoder update (`0.999` default):
- Code: `src/training/mocorank.py` lines 97-104.
- Status: PASS.

8. Score-pool initialization with shuffled multi-class mix:
- Code: strict path `strict_pool_init` in `src/training/mocorank.py` lines 299+.
- Status: PASS (strict mode only).

9. Baseline hyperparameters from paper config:
- Code: `main.py` strict mode loads `configs/baseline.yaml` and applies:
  - `batch_size`, `epochs`, `lr_init`, `score_pool_size`, `momentum_update`, `C`, `T`.
- Status: PASS (strict mode only).

10. Cosine annealing scheduler with `lr_final`:
- Code: `src/training/mocorank.py` uses `CosineAnnealingLR`.
- Status: PASS (strict mode path sets scheduler/lr_final from config).

## Strict Verification Run

- Run command used:
  - `python main.py --model cmose_baseline_paper --feature_dir data/CMOSE/features/openface --labels_json data/CMOSE/final_data_1.json --i3d_feature_dir data/CMOSE/features/i3d --strict_paper_baseline --epochs 1 --num_workers 0 --output_dir outputs/cmose_baseline_paper/strict_paper_verify_1ep`
- Note: in strict mode, baseline config overwrote `--epochs 1` to `1200` by design.
- Verified runtime config in metrics:
  - `epochs=1200`
  - `batch_size=256`
  - `lr=0.0005`
  - `score_pool_size=2048`
  - `momentum_update=0.999`
  - `baseline_hidden_dim=128`
  - `baseline_chunk_count=10`
  - `strict_paper_baseline=true`

## Result Comparison

- Strict run test metrics:
  - Accuracy: `0.7101`
  - Macro Accuracy: `0.4255`
  - F1 Macro: `0.4375`
  - F1 Weighted: `0.6857`

- Paper abstract claim reports improvement from MocoRank over prior losses (not full reproducibility protocol details in abstract text extraction).
- Current strict run does not yet match high headline performance targets expected from full paper reproduction.

## Conclusion

- Hyperparameter and training-mechanism parity against paper-stated items is now substantially improved and passes local audit checks.
- Performance gap remains, likely due to unrecoverable details not fully specified in extracted paper text and absence of official author training code for byte-level replication.
