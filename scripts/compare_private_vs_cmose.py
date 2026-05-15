"""Compare private dataset features against CMOSE and write a report."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.feature_analysis.domain_analysis import (
    centroid_cosine_distance,
    compute_domain_gap_score,
    feature_group_wasserstein,
)
from src.output_paths import DOMAIN_DIFFERENCE_DIR

META_COLS = ["frame", "face_id", "timestamp", "confidence", "success"]


@dataclass
class OpenFaceDomain:
    name: str
    root: Path
    csv_paths: list[Path]
    feature_columns: list[str]
    total_frames: int


def _discover_openface_domain(name: str, root: Path, max_scan_files: int = 2000) -> OpenFaceDomain:
    csv_paths = sorted(root.glob("*.csv"))
    if not csv_paths:
        return OpenFaceDomain(name, root, [], [], 0)

    feature_sets: list[set[str]] = []
    total_frames = 0
    for path in csv_paths[:max_scan_files]:
        df = pd.read_csv(path, usecols=lambda c: c in META_COLS, low_memory=False)
        total_frames += len(df)
        cols = pd.read_csv(path, nrows=1, low_memory=False).columns.tolist()
        feature_sets.append(set(c for c in cols if c not in META_COLS))

    common_cols = sorted(set.intersection(*feature_sets)) if feature_sets else []
    return OpenFaceDomain(name, root, csv_paths, common_cols, total_frames)


def _sample_openface_rows(
    csv_paths: list[Path],
    feature_cols: list[str],
    *,
    max_rows: int = 50000,
    per_file_cap: int = 256,
    random_state: int = 42,
) -> np.ndarray:
    if not csv_paths or not feature_cols:
        return np.empty((0, len(feature_cols)), dtype=np.float32)

    rng = np.random.default_rng(random_state)
    pieces: list[np.ndarray] = []
    remaining = max_rows
    for path in csv_paths:
        if remaining <= 0:
            break
        df = pd.read_csv(path, usecols=feature_cols, low_memory=False)
        if df.empty:
            continue
        take = min(len(df), per_file_cap, remaining)
        idx = rng.choice(len(df), size=take, replace=False)
        arr = df.iloc[idx].to_numpy(dtype=np.float32, copy=False)
        pieces.append(arr)
        remaining -= take
    if not pieces:
        return np.empty((0, len(feature_cols)), dtype=np.float32)
    return np.vstack(pieces).astype(np.float32, copy=False)


def _openface_group_map(cols: list[str]) -> dict[str, list[int]]:
    groups = {
        "gaze": [],
        "pose": [],
        "au": [],
        "eye_landmark_2d": [],
        "face_landmark_2d": [],
        "landmark_3d": [],
        "other": [],
    }
    for i, c in enumerate(cols):
        cl = c.lower()
        if cl.startswith("gaze_"):
            groups["gaze"].append(i)
        elif cl.startswith("pose_"):
            groups["pose"].append(i)
        elif cl.startswith("au"):
            groups["au"].append(i)
        elif cl.startswith("eye_lmk_") or "eye_lmk" in cl:
            groups["eye_landmark_2d"].append(i)
        elif cl.startswith("x_") or cl.startswith("y_"):
            groups["face_landmark_2d"].append(i)
        elif cl.startswith("x_") or cl.startswith("y_") or cl.startswith("z_"):
            groups["landmark_3d"].append(i)
        else:
            groups["other"].append(i)
    return groups


def _load_private_i3d(root: Path) -> np.ndarray:
    files = sorted(root.glob("*.npy"))
    mats = [np.load(p).astype(np.float32) for p in files]
    return np.stack(mats, axis=0) if mats else np.empty((0, 1024), dtype=np.float32)


def _load_cmose_i3d_from_json(path: Path) -> np.ndarray:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[np.ndarray] = []
    for meta in raw.values():
        embeds = meta.get("embeds")
        if isinstance(embeds, list) and len(embeds) == 1024:
            rows.append(np.asarray(embeds, dtype=np.float32))
    return np.stack(rows, axis=0) if rows else np.empty((0, 1024), dtype=np.float32)


def _to_markdown(report: dict) -> str:
    of = report["openface"]
    i3d = report["i3d"]
    colcmp = of["column_name_comparison"]
    lines = [
        "# Private vs CMOSE Comparison",
        "",
        "## OpenFace",
        f"- Private files: {of['private_file_count']}",
        f"- CMOSE files: {of['cmose_file_count']}",
        f"- Common feature columns: {of['common_feature_count']}",
        "",
        "### OpenFace Column Names",
        f"- Private sample file: {colcmp['private_sample_file']}",
        f"- CMOSE sample file: {colcmp['cmose_sample_file']}",
        f"- Private columns: {colcmp['private_column_count']}",
        f"- CMOSE columns: {colcmp['cmose_column_count']}",
        f"- Missing from private: {colcmp['missing_from_private_count']}",
        f"- Extra in private: {colcmp['extra_in_private_count']}",
        f"- Column order identical: {colcmp['order_identical']}",
        f"- Private sampled rows: {of['private_sample_rows']}",
        f"- CMOSE sampled rows: {of['cmose_sample_rows']}",
    ]
    if of["mean_wasserstein_common"] is not None:
        lines.append(f"- Mean Wasserstein distance (common features): {of['mean_wasserstein_common']:.6f}")
    if of["group_wasserstein"]:
        lines.append("- Group Wasserstein distances:")
        for k, v in sorted(of["group_wasserstein"].items()):
            lines.append(f"  - {k}: {v:.6f}")
    else:
        lines.append("- Group Wasserstein distances: unavailable")

    lines += [
        "",
        "## I3D",
        f"- Private vectors: {i3d['private_count']}",
        f"- CMOSE vectors (from final_data_1.json): {i3d['cmose_count']}",
        f"- Feature dimension: {i3d['feature_dim']}",
    ]
    if i3d["centroid_cosine_distance"] is not None:
        lines.append(f"- Centroid cosine distance: {i3d['centroid_cosine_distance']:.6f}")
    lines.append(f"- Composite domain gap score: {report['domain_gap_score']:.6f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    repo = REPO_ROOT
    output_dir = repo / DOMAIN_DIFFERENCE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    private_openface_root = repo / "data" / "private" / "features" / "openface"
    cmose_openface_root = repo / "data" / "CMOSE" / "secondFeature" / "secondFeature"
    private_i3d_root = repo / "data" / "private" / "features" / "i3d"
    cmose_json_path = repo / "data" / "CMOSE" / "final_data_1.json"

    private_of = _discover_openface_domain("private", private_openface_root)
    cmose_of = _discover_openface_domain("cmose", cmose_openface_root)
    common_cols = sorted(set(private_of.feature_columns).intersection(cmose_of.feature_columns))

    private_header: list[str] = []
    cmose_header: list[str] = []
    private_sample_file = ""
    cmose_sample_file = ""
    if private_of.csv_paths:
        private_sample_file = private_of.csv_paths[0].name
        private_header = pd.read_csv(private_of.csv_paths[0], nrows=0, low_memory=False).columns.tolist()
    if cmose_of.csv_paths:
        cmose_sample_file = cmose_of.csv_paths[0].name
        cmose_header = pd.read_csv(cmose_of.csv_paths[0], nrows=0, low_memory=False).columns.tolist()

    private_set = set(private_header)
    cmose_set = set(cmose_header)
    missing_from_private = sorted(cmose_set - private_set)
    extra_in_private = sorted(private_set - cmose_set)
    order_identical = private_header == cmose_header

    private_of_sample = _sample_openface_rows(private_of.csv_paths, common_cols, random_state=42)
    cmose_of_sample = _sample_openface_rows(cmose_of.csv_paths, common_cols, random_state=43)

    mean_w = None
    group_w: dict[str, float] = {}
    group_values: list[float] = []
    if len(common_cols) > 0 and len(private_of_sample) > 0 and len(cmose_of_sample) > 0:
        mean_w = feature_group_wasserstein(private_of_sample, cmose_of_sample)
        groups = _openface_group_map(common_cols)
        for gname, idxs in groups.items():
            if not idxs:
                continue
            w = feature_group_wasserstein(private_of_sample[:, idxs], cmose_of_sample[:, idxs])
            group_w[gname] = float(w)
            group_values.append(float(w))

    private_i3d = _load_private_i3d(private_i3d_root)
    cmose_i3d = _load_cmose_i3d_from_json(cmose_json_path)
    cos_dist = None
    if len(private_i3d) > 0 and len(cmose_i3d) > 0:
        cos_dist = centroid_cosine_distance(private_i3d, cmose_i3d)
    domain_gap = compute_domain_gap_score(group_values, cos_dist or 0.0)

    report = {
        "openface": {
            "private_root": str(private_openface_root),
            "cmose_root": str(cmose_openface_root),
            "private_file_count": len(private_of.csv_paths),
            "cmose_file_count": len(cmose_of.csv_paths),
            "private_total_frames_scanned": private_of.total_frames,
            "cmose_total_frames_scanned": cmose_of.total_frames,
            "private_feature_count_common_within_domain": len(private_of.feature_columns),
            "cmose_feature_count_common_within_domain": len(cmose_of.feature_columns),
            "common_feature_count": len(common_cols),
            "private_sample_rows": int(private_of_sample.shape[0]),
            "cmose_sample_rows": int(cmose_of_sample.shape[0]),
            "mean_wasserstein_common": float(mean_w) if mean_w is not None else None,
            "group_wasserstein": group_w,
            "column_name_comparison": {
                "private_sample_file": private_sample_file,
                "cmose_sample_file": cmose_sample_file,
                "private_column_count": len(private_header),
                "cmose_column_count": len(cmose_header),
                "missing_from_private_count": len(missing_from_private),
                "extra_in_private_count": len(extra_in_private),
                "order_identical": order_identical,
                "missing_from_private": missing_from_private,
                "extra_in_private": extra_in_private,
            },
        },
        "i3d": {
            "private_root": str(private_i3d_root),
            "cmose_json": str(cmose_json_path),
            "private_count": int(private_i3d.shape[0]),
            "cmose_count": int(cmose_i3d.shape[0]),
            "feature_dim": int(private_i3d.shape[1]) if private_i3d.size else (int(cmose_i3d.shape[1]) if cmose_i3d.size else 0),
            "centroid_cosine_distance": float(cos_dist) if cos_dist is not None else None,
        },
        "domain_gap_score": float(domain_gap),
    }

    json_path = output_dir / "private_vs_cmose_comparison.json"
    md_path = output_dir / "private_vs_cmose_comparison.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")


if __name__ == "__main__":
    main()
