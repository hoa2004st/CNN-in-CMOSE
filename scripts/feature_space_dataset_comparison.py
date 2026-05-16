"""Feature-space comparison between CMOSE and private datasets.

This script compares extracted feature distributions only. It does not train or
evaluate any downstream model and does not use labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp, wasserstein_distance


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.output_paths import DOMAIN_DIFFERENCE_DIR

META_COLS = {"frame", "face_id", "timestamp", "confidence", "success"}


@dataclass(frozen=True)
class SampledOpenFace:
    features: np.ndarray
    feature_columns: list[str]
    quality: pd.DataFrame
    sampled_files: int
    total_files: int


def _json_safe_float(value: float | np.floating) -> float | None:
    value = float(value)
    if np.isfinite(value):
        return value
    return None


def _list_csv_files(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*.csv") if p.is_file())


def _resolve_cmose_openface_dir() -> Path:
    candidates = [
        REPO_ROOT / "data" / "CMOSE" / "features" / "openface",
        REPO_ROOT / "data" / "CMOSE" / "secondFeature",
        REPO_ROOT / "data" / "CMOSE" / "secondFeature" / "secondFeature",
        REPO_ROOT / "data" / "CMOSE" / "openface-features" / "secondFeature",
    ]
    for candidate in candidates:
        if candidate.exists() and _list_csv_files(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find CMOSE OpenFace CSV files. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _manifest_path(value: object) -> Path:
    return REPO_ROOT / str(value).replace("\\", "/")


def _load_accepted_private_entries(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Accepted private manifest not found: {path}")
    df = pd.read_csv(path)
    required = {"clip_id", "openface_csv", "is_accepted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    accepted = df[pd.to_numeric(df["is_accepted"], errors="coerce").eq(1)].copy()
    if accepted.empty:
        raise ValueError(f"No accepted clips found in {path}")
    return accepted


def _accepted_openface_files(accepted: pd.DataFrame) -> list[Path]:
    files: list[Path] = []
    for rel in accepted["openface_csv"].astype(str):
        path = _manifest_path(rel)
        if path.exists():
            files.append(path)
    return sorted(files)


def _accepted_i3d_stems(accepted: pd.DataFrame) -> set[str]:
    return set(accepted["clip_id"].astype(str))


def _feature_columns_from_header(path: Path) -> list[str]:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    return [c for c in cols if c not in META_COLS]


def _common_openface_columns(private_files: list[Path], cmose_files: list[Path]) -> tuple[list[str], dict]:
    if not private_files or not cmose_files:
        raise FileNotFoundError("OpenFace CSV files were not found in both datasets.")

    private_cols = pd.read_csv(private_files[0], nrows=0).columns.tolist()
    cmose_cols = pd.read_csv(cmose_files[0], nrows=0).columns.tolist()
    private_features = [c for c in private_cols if c not in META_COLS]
    cmose_features = [c for c in cmose_cols if c not in META_COLS]
    common = [c for c in private_features if c in set(cmose_features)]

    return common, {
        "private_sample_file": private_files[0].name,
        "cmose_sample_file": cmose_files[0].name,
        "private_column_count": len(private_cols),
        "cmose_column_count": len(cmose_cols),
        "private_feature_count": len(private_features),
        "cmose_feature_count": len(cmose_features),
        "common_feature_count": len(common),
        "column_order_identical": private_cols == cmose_cols,
        "missing_from_private": sorted(set(cmose_cols) - set(private_cols)),
        "extra_in_private": sorted(set(private_cols) - set(cmose_cols)),
    }


def _sample_openface(
    files: list[Path],
    feature_columns: list[str],
    *,
    max_rows: int,
    per_file_cap: int,
    seed: int,
    require_success: bool = False,
    min_confidence: float | None = None,
) -> SampledOpenFace:
    rng = np.random.default_rng(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)

    usecols = list(META_COLS.intersection({"confidence", "success"})) + feature_columns
    rows: list[np.ndarray] = []
    quality_rows: list[pd.DataFrame] = []
    sampled_files = 0
    remaining = max_rows

    for path in shuffled:
        if remaining <= 0:
            break

        df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
        if df.empty:
            continue

        numeric = df[feature_columns].apply(pd.to_numeric, errors="coerce")
        finite_mask = np.isfinite(numeric.to_numpy(dtype=np.float32, copy=False)).all(axis=1)
        if require_success and "success" in df.columns:
            finite_mask &= pd.to_numeric(df["success"], errors="coerce").eq(1).to_numpy()
        if min_confidence is not None and "confidence" in df.columns:
            finite_mask &= pd.to_numeric(df["confidence"], errors="coerce").ge(min_confidence).to_numpy()
        if not finite_mask.any():
            continue

        valid_df = df.loc[finite_mask]
        numeric = numeric.loc[finite_mask]
        take = min(len(numeric), per_file_cap, remaining)
        idx = rng.choice(len(numeric), size=take, replace=False)

        rows.append(numeric.iloc[idx].to_numpy(dtype=np.float32, copy=False))

        q = valid_df.iloc[idx][[c for c in ("confidence", "success") if c in valid_df.columns]].copy()
        q["file"] = path.name
        quality_rows.append(q)

        sampled_files += 1
        remaining -= take

    features = np.vstack(rows).astype(np.float32, copy=False) if rows else np.empty((0, len(feature_columns)), dtype=np.float32)
    quality = pd.concat(quality_rows, ignore_index=True) if quality_rows else pd.DataFrame()
    return SampledOpenFace(features, feature_columns, quality, sampled_files, len(files))


def _openface_group(column: str) -> str:
    if column.startswith("gaze_"):
        return "gaze"
    if column.startswith("pose_"):
        return "head_pose"
    if column.startswith("eye_lmk_x_") or column.startswith("eye_lmk_y_"):
        return "eye_landmark_2d"
    if column.startswith("eye_lmk_X_") or column.startswith("eye_lmk_Y_") or column.startswith("eye_lmk_Z_"):
        return "eye_landmark_3d"
    if column.startswith("x_") or column.startswith("y_"):
        return "face_landmark_2d"
    if column.startswith("X_") or column.startswith("Y_") or column.startswith("Z_"):
        return "face_landmark_3d"
    if column.startswith("p_") or column == "p_scale":
        return "pdm"
    if column.startswith("AU") and column.endswith("_r"):
        return "au_intensity"
    if column.startswith("AU") and column.endswith("_c"):
        return "au_presence"
    return "other"


def _pooled_standardize(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.vstack([a, b]).astype(np.float64, copy=False)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    std[std < eps] = 1.0
    return ((a - mean) / std).astype(np.float32), ((b - mean) / std).astype(np.float32)


def _featurewise_distances(
    private: np.ndarray,
    cmose: np.ndarray,
    columns: list[str],
) -> pd.DataFrame:
    private_z, cmose_z = _pooled_standardize(private, cmose)
    rows = []
    for i, col in enumerate(columns):
        p = private_z[:, i]
        c = cmose_z[:, i]
        mean_shift = float(p.mean() - c.mean())
        rows.append(
            {
                "feature": col,
                "group": _openface_group(col),
                "private_mean_z": float(p.mean()),
                "cmose_mean_z": float(c.mean()),
                "abs_standardized_mean_diff": abs(mean_shift),
                "standardized_wasserstein": float(wasserstein_distance(p, c)),
                "ks_statistic": float(ks_2samp(p, c, alternative="two-sided", mode="asymp").statistic),
            }
        )
    return pd.DataFrame(rows)


def _group_summary(featurewise: pd.DataFrame) -> pd.DataFrame:
    return (
        featurewise.groupby("group", as_index=False)
        .agg(
            feature_count=("feature", "count"),
            mean_abs_standardized_mean_diff=("abs_standardized_mean_diff", "mean"),
            median_abs_standardized_mean_diff=("abs_standardized_mean_diff", "median"),
            mean_standardized_wasserstein=("standardized_wasserstein", "mean"),
            median_standardized_wasserstein=("standardized_wasserstein", "median"),
            mean_ks_statistic=("ks_statistic", "mean"),
            median_ks_statistic=("ks_statistic", "median"),
        )
        .sort_values("mean_standardized_wasserstein", ascending=False)
        .reset_index(drop=True)
    )


def _quality_summary(quality: pd.DataFrame) -> dict:
    if quality.empty:
        return {}
    out: dict[str, float | int | None] = {"sampled_rows": int(len(quality))}
    if "confidence" in quality:
        confidence = pd.to_numeric(quality["confidence"], errors="coerce")
        out.update(
            {
                "confidence_mean": _json_safe_float(confidence.mean()),
                "confidence_median": _json_safe_float(confidence.median()),
                "confidence_p05": _json_safe_float(confidence.quantile(0.05)),
                "confidence_p95": _json_safe_float(confidence.quantile(0.95)),
                "confidence_ge_0_80_rate": _json_safe_float((confidence >= 0.80).mean()),
                "confidence_ge_0_95_rate": _json_safe_float((confidence >= 0.95).mean()),
            }
        )
    if "success" in quality:
        success = pd.to_numeric(quality["success"], errors="coerce")
        out["success_rate"] = _json_safe_float((success == 1).mean())
    return out


def _load_i3d(root: Path, allowed_stems: set[str] | None = None) -> tuple[np.ndarray, list[str]]:
    files = sorted(root.glob("*.npy"))
    if allowed_stems is not None:
        files = [p for p in files if p.stem in allowed_stems]
    rows = [np.load(path).astype(np.float32).reshape(-1) for path in files]
    if not rows:
        return np.empty((0, 0), dtype=np.float32), []
    dims = {row.shape[0] for row in rows}
    if len(dims) != 1:
        raise ValueError(f"I3D feature dimensions are inconsistent in {root}: {sorted(dims)}")
    return np.stack(rows, axis=0), [p.name for p in files]


def _i3d_summary(private: np.ndarray, cmose: np.ndarray) -> tuple[dict, pd.DataFrame]:
    dim = private.shape[1] if private.size else cmose.shape[1]
    columns = [f"i3d_{i:04d}" for i in range(dim)]
    featurewise = _featurewise_distances(private, cmose, columns)
    private_z, cmose_z = _pooled_standardize(private, cmose)
    private_centroid = private.mean(axis=0)
    cmose_centroid = cmose.mean(axis=0)
    private_centroid_z = private_z.mean(axis=0)
    cmose_centroid_z = cmose_z.mean(axis=0)
    return {
        "private_count": int(private.shape[0]),
        "cmose_count": int(cmose.shape[0]),
        "feature_dim": int(dim),
        "centroid_cosine_distance_raw": _json_safe_float(cosine(private_centroid, cmose_centroid)),
        "centroid_l2_raw": _json_safe_float(np.linalg.norm(private_centroid - cmose_centroid)),
        "centroid_l2_pooled_z": _json_safe_float(np.linalg.norm(private_centroid_z - cmose_centroid_z)),
        "centroid_rms_shift_pooled_z": _json_safe_float(
            np.linalg.norm(private_centroid_z - cmose_centroid_z) / np.sqrt(dim)
        ),
        "mean_standardized_wasserstein": _json_safe_float(featurewise["standardized_wasserstein"].mean()),
        "median_standardized_wasserstein": _json_safe_float(featurewise["standardized_wasserstein"].median()),
        "mean_ks_statistic": _json_safe_float(featurewise["ks_statistic"].mean()),
        "median_ks_statistic": _json_safe_float(featurewise["ks_statistic"].median()),
        "top_20_mean_shift_features": featurewise.sort_values(
            "abs_standardized_mean_diff", ascending=False
        ).head(20)[["feature", "abs_standardized_mean_diff", "standardized_wasserstein", "ks_statistic"]].to_dict(
            orient="records"
        ),
    }, featurewise


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _write_markdown(report: dict, group_summary: pd.DataFrame, out_path: Path) -> None:
    openface = report["openface"]
    i3d = report["i3d"]
    q_private = openface["quality"]["private"]
    q_cmose = openface["quality"]["cmose"]
    valid = openface.get("high_confidence_valid_only")

    lines = [
        "# CMOSE vs Accepted Private Feature-Space Comparison",
        "",
        "This comparison uses extracted feature files only: OpenFace CSV files and I3D `.npy` embeddings. The private side is filtered by `data/private/accepted.csv`; no labels or downstream model outputs are used.",
        "",
        "## Overall Answer",
        "",
        (
            f"- OpenFace distributions are strongly shifted when using all extractor rows: mean standardized Wasserstein "
            f"{_fmt(openface['overall']['mean_standardized_wasserstein'])}, mean KS "
            f"{_fmt(openface['overall']['mean_ks_statistic'])}."
        ),
        (
            f"- I3D distributions show a comparable marginal shift and clear centroid movement: centroid cosine distance "
            f"{_fmt(i3d['centroid_cosine_distance_raw'])}, centroid RMS shift "
            f"{_fmt(i3d['centroid_rms_shift_pooled_z'])}, mean standardized Wasserstein "
            f"{_fmt(i3d['mean_standardized_wasserstein'])}."
        ),
        (
            "- The private set is not just a smaller sample of the same feature distribution; "
            "the largest OpenFace gaps are geometric/pose related and the I3D embedding centroid also moves."
        ),
        "",
        "## Dataset Coverage",
        "",
        f"- Private accepted clips from manifest: {_fmt(report['private_filter']['accepted_rows'])} of {_fmt(report['private_filter']['manifest_rows'])}.",
        f"- Private accepted OpenFace CSV files: {_fmt(openface['private_total_files'])}; sampled files: {_fmt(openface['private_sampled_files'])}; sampled rows: {_fmt(openface['private_sampled_rows'])}.",
        f"- CMOSE OpenFace CSV files: {_fmt(openface['cmose_total_files'])}; sampled files: {_fmt(openface['cmose_sampled_files'])}; sampled rows: {_fmt(openface['cmose_sampled_rows'])}.",
        f"- Common OpenFace feature columns: {_fmt(openface['schema']['common_feature_count'])}; column order identical: {_fmt(openface['schema']['column_order_identical'])}.",
        f"- Private I3D vectors: {_fmt(i3d['private_count'])}; CMOSE I3D vectors: {_fmt(i3d['cmose_count'])}; dimension: {_fmt(i3d['feature_dim'])}.",
        "",
        "## OpenFace Group Distances",
        "",
        "| group | features | mean Wasserstein z | median Wasserstein z | mean KS | mean abs mean shift z |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in group_summary.to_dict(orient="records"):
        lines.append(
            "| {group} | {feature_count} | {wmean} | {wmed} | {ks} | {smd} |".format(
                group=row["group"],
                feature_count=int(row["feature_count"]),
                wmean=_fmt(row["mean_standardized_wasserstein"]),
                wmed=_fmt(row["median_standardized_wasserstein"]),
                ks=_fmt(row["mean_ks_statistic"]),
                smd=_fmt(row["mean_abs_standardized_mean_diff"]),
            )
        )

    lines += [
        "",
        "## OpenFace Extraction Quality",
        "",
        "| dataset | mean confidence | median confidence | confidence >= 0.80 | confidence >= 0.95 | success rate |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| private | {_fmt(q_private.get('confidence_mean'))} | {_fmt(q_private.get('confidence_median'))} | "
            f"{_fmt(q_private.get('confidence_ge_0_80_rate'))} | {_fmt(q_private.get('confidence_ge_0_95_rate'))} | "
            f"{_fmt(q_private.get('success_rate'))} |"
        ),
        (
            f"| CMOSE | {_fmt(q_cmose.get('confidence_mean'))} | {_fmt(q_cmose.get('confidence_median'))} | "
            f"{_fmt(q_cmose.get('confidence_ge_0_80_rate'))} | {_fmt(q_cmose.get('confidence_ge_0_95_rate'))} | "
            f"{_fmt(q_cmose.get('success_rate'))} |"
        ),
        "",
    ]
    if valid:
        lines += [
            "## High-Confidence OpenFace Sensitivity",
            "",
            "This repeats the OpenFace comparison after filtering to `success == 1` and `confidence >= 0.80` before sampling.",
            "",
            f"- Private valid sampled rows: {_fmt(valid['private_sampled_rows'])}; CMOSE valid sampled rows: {_fmt(valid['cmose_sampled_rows'])}.",
            (
                f"- Valid-only mean standardized Wasserstein: "
                f"{_fmt(valid['overall']['mean_standardized_wasserstein'])}; mean KS: "
                f"{_fmt(valid['overall']['mean_ks_statistic'])}."
            ),
            "",
            "| group | features | mean Wasserstein z | median Wasserstein z | mean KS | mean abs mean shift z |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in valid["groups"]:
            lines.append(
                "| {group} | {feature_count} | {wmean} | {wmed} | {ks} | {smd} |".format(
                    group=row["group"],
                    feature_count=int(row["feature_count"]),
                    wmean=_fmt(row["mean_standardized_wasserstein"]),
                    wmed=_fmt(row["median_standardized_wasserstein"]),
                    ks=_fmt(row["mean_ks_statistic"]),
                    smd=_fmt(row["mean_abs_standardized_mean_diff"]),
                )
            )
        lines.append("")

    lines += [
        "## I3D Distances",
        "",
        f"- Raw centroid cosine distance: {_fmt(i3d['centroid_cosine_distance_raw'])}.",
        f"- Raw centroid L2 distance: {_fmt(i3d['centroid_l2_raw'])}.",
        f"- Pooled-z centroid L2 distance: {_fmt(i3d['centroid_l2_pooled_z'])}.",
        f"- Pooled-z centroid RMS shift per dimension: {_fmt(i3d['centroid_rms_shift_pooled_z'])}.",
        f"- Mean / median standardized Wasserstein: {_fmt(i3d['mean_standardized_wasserstein'])} / {_fmt(i3d['median_standardized_wasserstein'])}.",
        f"- Mean / median KS statistic: {_fmt(i3d['mean_ks_statistic'])} / {_fmt(i3d['median_ks_statistic'])}.",
        "",
        "## Interpretation Notes",
        "",
        "- Standardized Wasserstein is computed per feature after pooled z-scoring; values are in standard-deviation units.",
        "- KS is the maximum empirical CDF gap per feature; 0 means identical marginal distributions and 1 means fully separated.",
        "- OpenFace pixel-coordinate groups can reflect camera framing/resolution and face scale, not only behavior.",
        "- The report intentionally avoids classifier/domain-prediction accuracy because that would introduce another model.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    private_openface = REPO_ROOT / "data" / "private" / "features" / "openface"
    cmose_openface = _resolve_cmose_openface_dir()
    private_i3d_root = REPO_ROOT / "data" / "private" / "features" / "i3d"
    cmose_i3d_root = REPO_ROOT / "data" / "CMOSE" / "features" / "i3d"
    accepted_csv = REPO_ROOT / args.private_accepted_csv

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    accepted_private = _load_accepted_private_entries(accepted_csv)
    private_openface_files = _accepted_openface_files(accepted_private)
    cmose_openface_files = _list_csv_files(cmose_openface)
    accepted_i3d_stems = _accepted_i3d_stems(accepted_private)

    common_cols, schema = _common_openface_columns(private_openface_files, cmose_openface_files)
    private_of = _sample_openface(
        private_openface_files,
        common_cols,
        max_rows=args.max_openface_rows,
        per_file_cap=args.openface_per_file_cap,
        seed=args.seed,
    )
    cmose_of = _sample_openface(
        cmose_openface_files,
        common_cols,
        max_rows=args.max_openface_rows,
        per_file_cap=args.openface_per_file_cap,
        seed=args.seed + 1,
    )
    if private_of.features.size == 0 or cmose_of.features.size == 0:
        raise RuntimeError("OpenFace sampling produced no comparable rows.")

    openface_featurewise = _featurewise_distances(private_of.features, cmose_of.features, common_cols)
    openface_group_summary = _group_summary(openface_featurewise)

    private_of_valid = _sample_openface(
        private_openface_files,
        common_cols,
        max_rows=args.max_openface_rows,
        per_file_cap=args.openface_per_file_cap,
        seed=args.seed + 2,
        require_success=True,
        min_confidence=0.80,
    )
    cmose_of_valid = _sample_openface(
        cmose_openface_files,
        common_cols,
        max_rows=args.max_openface_rows,
        per_file_cap=args.openface_per_file_cap,
        seed=args.seed + 3,
        require_success=True,
        min_confidence=0.80,
    )
    if private_of_valid.features.size == 0 or cmose_of_valid.features.size == 0:
        raise RuntimeError("Valid-only OpenFace sampling produced no comparable rows.")
    openface_valid_featurewise = _featurewise_distances(
        private_of_valid.features, cmose_of_valid.features, common_cols
    )
    openface_valid_group_summary = _group_summary(openface_valid_featurewise)

    private_i3d, private_i3d_files = _load_i3d(private_i3d_root, accepted_i3d_stems)
    cmose_i3d, cmose_i3d_files = _load_i3d(cmose_i3d_root)
    if private_i3d.size == 0 or cmose_i3d.size == 0:
        raise RuntimeError("I3D feature loading produced no comparable vectors.")
    if private_i3d.shape[1] != cmose_i3d.shape[1]:
        raise ValueError(f"I3D dimensions differ: private {private_i3d.shape[1]} vs CMOSE {cmose_i3d.shape[1]}")
    i3d_summary, i3d_featurewise = _i3d_summary(private_i3d, cmose_i3d)

    report = {
        "inputs": {
            "private_openface": str(private_openface),
            "cmose_openface": str(cmose_openface),
            "private_i3d": str(private_i3d_root),
            "cmose_i3d": str(cmose_i3d_root),
            "private_accepted_csv": str(accepted_csv),
            "max_openface_rows_per_dataset": args.max_openface_rows,
            "openface_per_file_cap": args.openface_per_file_cap,
            "seed": args.seed,
        },
        "private_filter": {
            "manifest_rows": int(len(pd.read_csv(accepted_csv, usecols=["is_accepted"]))),
            "accepted_rows": int(len(accepted_private)),
            "accepted_openface_files_found": int(len(private_openface_files)),
            "accepted_i3d_stems": int(len(accepted_i3d_stems)),
        },
        "openface": {
            "schema": schema,
            "private_total_files": private_of.total_files,
            "cmose_total_files": cmose_of.total_files,
            "private_sampled_files": private_of.sampled_files,
            "cmose_sampled_files": cmose_of.sampled_files,
            "private_sampled_rows": int(private_of.features.shape[0]),
            "cmose_sampled_rows": int(cmose_of.features.shape[0]),
            "overall": {
                "mean_standardized_wasserstein": _json_safe_float(
                    openface_featurewise["standardized_wasserstein"].mean()
                ),
                "median_standardized_wasserstein": _json_safe_float(
                    openface_featurewise["standardized_wasserstein"].median()
                ),
                "mean_ks_statistic": _json_safe_float(openface_featurewise["ks_statistic"].mean()),
                "median_ks_statistic": _json_safe_float(openface_featurewise["ks_statistic"].median()),
                "mean_abs_standardized_mean_diff": _json_safe_float(
                    openface_featurewise["abs_standardized_mean_diff"].mean()
                ),
                "median_abs_standardized_mean_diff": _json_safe_float(
                    openface_featurewise["abs_standardized_mean_diff"].median()
                ),
            },
            "groups": _records(openface_group_summary),
            "quality": {
                "private": _quality_summary(private_of.quality),
                "cmose": _quality_summary(cmose_of.quality),
            },
            "top_30_mean_shift_features": _records(
                openface_featurewise.sort_values("abs_standardized_mean_diff", ascending=False).head(30)
            ),
            "high_confidence_valid_only": {
                "filter": {"success": 1, "confidence_min": 0.80},
                "private_sampled_files": private_of_valid.sampled_files,
                "cmose_sampled_files": cmose_of_valid.sampled_files,
                "private_sampled_rows": int(private_of_valid.features.shape[0]),
                "cmose_sampled_rows": int(cmose_of_valid.features.shape[0]),
                "overall": {
                    "mean_standardized_wasserstein": _json_safe_float(
                        openface_valid_featurewise["standardized_wasserstein"].mean()
                    ),
                    "median_standardized_wasserstein": _json_safe_float(
                        openface_valid_featurewise["standardized_wasserstein"].median()
                    ),
                    "mean_ks_statistic": _json_safe_float(openface_valid_featurewise["ks_statistic"].mean()),
                    "median_ks_statistic": _json_safe_float(openface_valid_featurewise["ks_statistic"].median()),
                    "mean_abs_standardized_mean_diff": _json_safe_float(
                        openface_valid_featurewise["abs_standardized_mean_diff"].mean()
                    ),
                    "median_abs_standardized_mean_diff": _json_safe_float(
                        openface_valid_featurewise["abs_standardized_mean_diff"].median()
                    ),
                },
                "groups": _records(openface_valid_group_summary),
                "quality": {
                    "private": _quality_summary(private_of_valid.quality),
                    "cmose": _quality_summary(cmose_of_valid.quality),
                },
            },
        },
        "i3d": i3d_summary | {
            "private_files": len(private_i3d_files),
            "cmose_files": len(cmose_i3d_files),
        },
    }

    json_path = out_dir / "feature_space_dataset_comparison.json"
    md_path = out_dir / "feature_space_dataset_comparison.md"
    openface_featurewise_path = out_dir / "feature_space_openface_featurewise.csv"
    openface_group_path = out_dir / "feature_space_openface_groups.csv"
    openface_valid_featurewise_path = out_dir / "feature_space_openface_valid_featurewise.csv"
    openface_valid_group_path = out_dir / "feature_space_openface_valid_groups.csv"
    i3d_featurewise_path = out_dir / "feature_space_i3d_featurewise.csv"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, openface_group_summary, md_path)
    openface_featurewise.to_csv(openface_featurewise_path, index=False)
    openface_group_summary.to_csv(openface_group_path, index=False)
    openface_valid_featurewise.to_csv(openface_valid_featurewise_path, index=False)
    openface_valid_group_summary.to_csv(openface_valid_group_path, index=False)
    i3d_featurewise.to_csv(i3d_featurewise_path, index=False)

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {openface_featurewise_path}")
    print(f"wrote {openface_group_path}")
    print(f"wrote {openface_valid_featurewise_path}")
    print(f"wrote {openface_valid_group_path}")
    print(f"wrote {i3d_featurewise_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DOMAIN_DIFFERENCE_DIR),
        help="Output directory relative to repo root.",
    )
    parser.add_argument(
        "--private-accepted-csv",
        default="data/private/accepted.csv",
        help="Private manifest used to keep only accepted clips.",
    )
    parser.add_argument("--max-openface-rows", type=int, default=60000, help="Rows sampled per dataset.")
    parser.add_argument("--openface-per-file-cap", type=int, default=128, help="Max sampled rows per OpenFace CSV.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for reproducible OpenFace sampling.")
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
