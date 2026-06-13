"""Prediction-level agreement and metric-reliability statistics for the results chapters.

Per-config metrics say how *well* each model scores; this module asks questions only the
per-clip prediction log can answer:

* Do configs with similar scores make the same per-clip decisions? (pairwise Cohen's kappa
  between every in-domain base config)
* How much headroom is left if the models err on different clips? (oracle / majority-vote
  accuracy over the config pool)
* Do the OpenFace and I3D feature families fail on the same clips? (pairwise error overlap)
* Do the six evaluation metrics rank models the same way? (Kendall-tau rank correlation)

Everything reads the naive per-clip prediction log and the assessment matrices through
:mod:`src.analysis.aggregate`; the hybrid 1-GB prediction log is never touched.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score

from src.analysis import aggregate as ag
from src.visualization.style import (
    COMPARISON_LOSS_SLUGS,
    MODEL_ORDER,
    display_model_name,
    loss_display_name,
)

QWK = "quadratic_weighted_kappa"


def config_label(model: str, loss: str) -> str:
    return f"{display_model_name(model)}/{loss_display_name(loss)}"


# --------------------------------------------------------------------------------------
# Per-clip prediction table (in-domain CMOSE)
# --------------------------------------------------------------------------------------
def indomain_prediction_table() -> tuple[pd.DataFrame, pd.Series]:
    """Pivot the in-domain CMOSE cell to a clip x config matrix of predicted class ids.

    Returns ``(wide, true)``: ``wide`` has one column per (model, loss) config in
    model-major order, ``true`` the ground-truth class id per clip. Clips missing any
    config's prediction are dropped so every pairwise comparison uses the same clips.
    """
    preds = ag.load_naive_predictions()
    cell = preds[(preds["train_group"] == "cmose") & (preds["test_set"] == "cmose_test")]
    wide = cell.pivot_table(index="clip_id", columns=["model", "loss"],
                            values="predicted_id", aggfunc="first")
    ordered = [(m, l) for m in MODEL_ORDER for l in COMPARISON_LOSS_SLUGS
               if (m, l) in wide.columns]
    wide = wide[ordered].dropna(axis=0).astype(int)
    true = (cell.drop_duplicates("clip_id").set_index("clip_id")["true_id"]
            .reindex(wide.index).astype(int))
    return wide, true


def pairwise_kappa(wide: pd.DataFrame) -> pd.DataFrame:
    """Symmetric config x config matrix of Cohen's kappa between predicted labels."""
    cols = list(wide.columns)
    out = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for a, b in itertools.combinations(cols, 2):
        # If either side predicts a single class everywhere, kappa is undefined (0/0);
        # sklearn returns nan there, which we keep — the heatmap shows it as a gap.
        k = cohen_kappa_score(wide[a], wide[b])
        out.loc[[a], [b]] = k
        out.loc[[b], [a]] = k
    return out


def kappa_block_means(kappa: pd.DataFrame) -> dict[str, float]:
    """Mean off-diagonal kappa split by what the pair shares: model, loss, or neither."""
    sums = {"same model": [], "same loss": [], "different model & loss": []}
    for a, b in itertools.combinations(list(kappa.columns), 2):
        value = float(kappa.loc[[a], [b]].iloc[0, 0])
        if a[0] == b[0]:
            sums["same model"].append(value)
        elif a[1] == b[1]:
            sums["same loss"].append(value)
        else:
            sums["different model & loss"].append(value)
    return {key: float(np.nanmean(vals)) for key, vals in sums.items() if vals}


# --------------------------------------------------------------------------------------
# Ensemble headroom & feature-family overlap
# --------------------------------------------------------------------------------------
def oracle_stats(wide: pd.DataFrame, true: pd.Series) -> dict[str, float]:
    """Best-single / majority-vote / oracle accuracy over the config pool."""
    correct = wide.eq(true, axis=0)
    majority = wide.mode(axis=1).iloc[:, 0]  # ties resolved toward the lower class id
    return {
        "n_clips": int(len(wide)),
        "n_configs": int(wide.shape[1]),
        "best_single_accuracy": float(correct.mean().max()),
        "majority_vote_accuracy": float((majority == true).mean()),
        "oracle_accuracy": float(correct.any(axis=1).mean()),
        "all_wrong_share": float((~correct).all(axis=1).mean()),
    }


def pair_overlap(wide: pd.DataFrame, true: pd.Series,
                 a: tuple[str, str], b: tuple[str, str]) -> dict[str, float]:
    """Error overlap between two configs: who is exclusively right where."""
    ok_a = wide[a].eq(true)
    ok_b = wide[b].eq(true)
    return {
        "kappa": float(cohen_kappa_score(wide[a], wide[b])),
        "both_correct": float((ok_a & ok_b).mean()),
        "only_a": float((ok_a & ~ok_b).mean()),
        "only_b": float((~ok_a & ok_b).mean()),
        "both_wrong": float((~ok_a & ~ok_b).mean()),
        "pair_oracle_accuracy": float((ok_a | ok_b).mean()),
    }


# --------------------------------------------------------------------------------------
# Metric reliability (rank agreement between the six metrics)
# --------------------------------------------------------------------------------------
def metric_rank_correlation(frame: pd.DataFrame,
                            metrics: list[str] | None = None) -> pd.DataFrame:
    """Kendall-tau rank correlation between metrics over the configs in ``frame``.

    Lower-is-better metrics (MAE family) are sign-flipped first, so tau > 0 always means
    "the two metrics agree on which configs are better".
    """
    metrics = metrics or ag.METRIC_COLUMNS
    oriented = frame[metrics].copy()
    for metric in metrics:
        if metric in ag.LOWER_BETTER_METRICS:
            oriented[metric] = -oriented[metric]
    out = pd.DataFrame(np.eye(len(metrics)), index=metrics, columns=metrics)
    for a, b in itertools.combinations(metrics, 2):
        tau = kendalltau(oriented[a], oriented[b]).statistic
        out.loc[a, b] = out.loc[b, a] = float(tau)
    return out
