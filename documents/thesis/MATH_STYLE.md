# Math / Equation Style Rules

Conventions for all mathematics in `documents/thesis/` (display equations and inline math).
Apply on every edit that touches a formula. The canonical hand-edit copy of every equation
lives in `documents/thesis/equations_review.tex`.

## Rules

1. **Multi-letter operators use `\operatorname{...}`** (not `\mathrm`, not bare letters).
   - Applies to: `\operatorname{Attention}`, `\operatorname{softmax}`, `\operatorname{Acc}`,
     `\operatorname{MacroAcc}`, `\operatorname{MAE}`.
   - Single-letter symbols stay plain math italic (`\rho`, `\tau`, `\kappa`, `\sigma`).

2. **Loss and metric subscripts use `\text{...}`** (not `\mathrm`), so the font is consistent
   across every subscript.
   - `\mathcal{L}_{\text{CE}}`, `\mathcal{L}_{\text{WCE}}`, `\mathcal{L}_{\text{ord}}`,
     `\mathcal{L}_{\text{main}}`, `\operatorname{Acc}_{\text{oracle}}`, `\lambda_{\text{aux}}`.
   - Do **not** mix `\mathrm` and `\text` for subscripts in the same document.

3. **Define an operator once, then reuse the name.** The softmax appears both as the class
   probability `p_k = \operatorname{softmax}(z)_k = e^{z_k}/\sum_j e^{z_j}` and inside attention;
   both must use `\operatorname{softmax}`, not one spelled out and one not.

4. **Transpose is `^\top`** everywhere (never `^T`, never `'`).

5. **Indicator function is `\mathbf{1}[\,\cdot\,]`** (never `\mathbb{1}`, never `\chi`).

6. **Named constants, not magic numbers.** A constant that the prose discusses gets a symbol
   defined next to the equation, e.g. the auxiliary weight is `\lambda_{\text{aux}}` with
   `\lambda_{\text{aux}}=0.2` stated in the following sentence — not a bare `0.2` in the formula.

7. **Combinatorial counts use `\binom{n}{2}`** rather than `\tfrac{1}{2}n(n-1)`.

8. **`align` numbering.** In a multi-row `align`, either number every row or wrap the block in a
   single `equation`+`aligned` for one number. Avoid `\notag` on some rows but not others within
   one logical block (current LSTM-gates block is the one known exception, kept intentionally).

## Symbol glossary (keep consistent across chapters)

| Symbol | Meaning |
| --- | --- |
| `K` | number of classes (`K=4`) |
| `z`, `p_k` | logits; softmax probability of class `k` |
| `y`, `\hat{y}` | true / predicted class label |
| `C_{ij}` | confusion-matrix count (true `i`, predicted `j`); `N=\sum_{ij}C_{ij}` |
| `P_k`, `T_k` | predicted CDF; target step CDF `\mathbf{1}[k\ge y]` |
| `W_{ij}`, `E_{ij}` | kappa penalty / expected-count matrices |
| `w_c`, `n_c` | class weight; training count of class `c` |
| `\lambda_{\text{aux}}` | auxiliary-loss weight (`=0.2`) |
| `S` | number of supervised streams (5 groups, or 6 with I3D) |
| `\rho`, `\tau` | Spearman / Kendall rank correlation |
