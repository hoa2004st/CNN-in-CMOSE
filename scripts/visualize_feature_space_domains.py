"""Visualize CMOSE vs accepted private feature spaces with t-SNE/UMAP.

The private dataset is filtered through data/private/accepted.csv. Each plotted
point is one clip/person feature sample:
  - OpenFace: per-CSV mean feature vector over valid frames
  - I3D: one .npy embedding
  - OpenFace + I3D: concatenated standardized modality vectors
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.style import domain_color
from src.output_paths import DOMAIN_DIFFERENCE_DIR

META_COLS = {"frame", "face_id", "timestamp", "confidence", "success"}


@dataclass(frozen=True)
class SamplePaths:
    sample_id: str
    openface_csv: Path
    i3d_npy: Path


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    matrix: np.ndarray
    labels: list[str]
    sample_ids: list[str]


def _load_accepted_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"clip_id", "openface_csv", "is_accepted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df[pd.to_numeric(df["is_accepted"], errors="coerce").eq(1)].copy()


def _manifest_path(value: object) -> Path:
    return REPO_ROOT / str(value).replace("\\", "/")


def _private_samples(accepted_csv: Path, i3d_root: Path) -> list[SamplePaths]:
    accepted = _load_accepted_manifest(accepted_csv)
    samples: list[SamplePaths] = []
    for row in accepted.itertuples(index=False):
        sample_id = str(row.clip_id)
        openface_csv = _manifest_path(row.openface_csv)
        i3d_npy = i3d_root / f"{sample_id}.npy"
        if openface_csv.exists() and i3d_npy.exists():
            samples.append(SamplePaths(sample_id, openface_csv, i3d_npy))
    return sorted(samples, key=lambda s: s.sample_id)


def _cmose_samples(openface_root: Path, i3d_root: Path, n: int, seed: int) -> list[SamplePaths]:
    i3d_files = sorted(i3d_root.glob("*.npy"))
    rng = np.random.default_rng(seed)
    shuffled = list(i3d_files)
    rng.shuffle(shuffled)

    samples: list[SamplePaths] = []
    for i3d_npy in shuffled:
        openface_csv = openface_root / f"{i3d_npy.stem}.csv"
        if openface_csv.exists():
            samples.append(SamplePaths(i3d_npy.stem, openface_csv, i3d_npy))
        if len(samples) >= n:
            break
    if len(samples) < n:
        raise RuntimeError(f"Only found {len(samples)} CMOSE samples with both OpenFace and I3D; need {n}.")
    return sorted(samples, key=lambda s: s.sample_id)


def _openface_columns(private_csv: Path, cmose_csv: Path) -> list[str]:
    private_cols = pd.read_csv(private_csv, nrows=0).columns.tolist()
    cmose_cols = pd.read_csv(cmose_csv, nrows=0).columns.tolist()
    cmose_set = set(cmose_cols)
    return [c for c in private_cols if c not in META_COLS and c in cmose_set]


def _aggregate_openface(
    path: Path,
    feature_columns: list[str],
    *,
    require_success: bool,
    min_confidence: float | None,
) -> np.ndarray | None:
    usecols = set(feature_columns) | {"success", "confidence"}
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    if df.empty:
        return None

    numeric = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(numeric.to_numpy(dtype=np.float32, copy=False)).all(axis=1)
    if require_success and "success" in df.columns:
        mask &= pd.to_numeric(df["success"], errors="coerce").eq(1).to_numpy()
    if min_confidence is not None and "confidence" in df.columns:
        mask &= pd.to_numeric(df["confidence"], errors="coerce").ge(min_confidence).to_numpy()

    if not mask.any():
        return None
    return numeric.loc[mask].to_numpy(dtype=np.float32, copy=False).mean(axis=0)


def _load_i3d(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32).reshape(-1)


def _load_blocks(
    private_samples: list[SamplePaths],
    cmose_samples: list[SamplePaths],
    feature_columns: list[str],
    *,
    require_success: bool,
    min_confidence: float | None,
) -> tuple[FeatureBlock, FeatureBlock, FeatureBlock, dict]:
    openface_rows: list[np.ndarray] = []
    i3d_rows: list[np.ndarray] = []
    labels: list[str] = []
    sample_ids: list[str] = []
    dropped: dict[str, list[str]] = {"private": [], "CMOSE": []}

    for label, samples in (("private", private_samples), ("CMOSE", cmose_samples)):
        for sample in samples:
            of = _aggregate_openface(
                sample.openface_csv,
                feature_columns,
                require_success=require_success,
                min_confidence=min_confidence,
            )
            if of is None:
                dropped[label].append(sample.sample_id)
                continue
            openface_rows.append(of)
            i3d_rows.append(_load_i3d(sample.i3d_npy))
            labels.append(label)
            sample_ids.append(sample.sample_id)

    openface = np.vstack(openface_rows).astype(np.float32, copy=False)
    i3d = np.vstack(i3d_rows).astype(np.float32, copy=False)
    openface_z = StandardScaler().fit_transform(openface)
    i3d_z = StandardScaler().fit_transform(i3d)
    fused = np.hstack([openface_z, i3d_z]).astype(np.float32, copy=False)

    return (
        FeatureBlock("OpenFace", openface, labels, sample_ids),
        FeatureBlock("I3D", i3d, labels, sample_ids),
        FeatureBlock("OpenFace + I3D", fused, labels, sample_ids),
        dropped,
    )


def _preprocess_for_embedding(matrix: np.ndarray, seed: int, pca_dims: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(matrix)
    n_components = min(pca_dims, scaled.shape[1], scaled.shape[0] - 1)
    if n_components < 2:
        return scaled
    return PCA(n_components=n_components, random_state=seed).fit_transform(scaled)


def _tsne(matrix: np.ndarray, seed: int, pca_dims: int) -> np.ndarray:
    reduced = _preprocess_for_embedding(matrix, seed, pca_dims)
    perplexity = max(5, min(30, (reduced.shape[0] - 1) // 3))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(reduced)


def _umap(matrix: np.ndarray, seed: int, pca_dims: int) -> np.ndarray | None:
    try:
        import umap  # type: ignore
    except Exception:
        return None
    reduced = _preprocess_for_embedding(matrix, seed, pca_dims)
    return umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.1, random_state=seed).fit_transform(reduced)


def _plot(coords: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    for label in ("CMOSE", "private"):
        idx = np.array([x == label for x in labels])
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=20,
            c=domain_color(label),
            alpha=0.72,
            edgecolors="none",
            label=f"{label} (n={int(idx.sum())})",
        )
    ax.set_title(title)
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    ax.grid(alpha=0.18)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _slug(name: str) -> str:
    return name.lower().replace(" + ", "_plus_").replace(" ", "_")


def run(args: argparse.Namespace) -> None:
    accepted_csv = REPO_ROOT / args.private_accepted_csv
    private_i3d_root = REPO_ROOT / "data" / "private" / "features" / "i3d"
    cmose_i3d_root = REPO_ROOT / "data" / "CMOSE" / "features" / "i3d"
    cmose_openface_root = REPO_ROOT / "data" / "CMOSE" / "secondFeature" / "secondFeature"

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    private = _private_samples(accepted_csv, private_i3d_root)
    if args.max_samples:
        private = private[: args.max_samples]
    cmose = _cmose_samples(cmose_openface_root, cmose_i3d_root, len(private), args.seed)
    feature_columns = _openface_columns(private[0].openface_csv, cmose[0].openface_csv)

    openface, i3d, fused, dropped = _load_blocks(
        private,
        cmose,
        feature_columns,
        require_success=args.require_success,
        min_confidence=args.min_confidence,
    )
    blocks = [openface, i3d, fused]

    coord_rows: list[dict] = []
    generated: list[str] = []
    for block in blocks:
        methods = {"tsne": _tsne(block.matrix, args.seed, args.pca_dims)}
        umap_coords = _umap(block.matrix, args.seed, args.pca_dims)
        if umap_coords is not None:
            methods["umap"] = umap_coords

        for method, coords in methods.items():
            filename = f"{method}_{_slug(block.name)}.png"
            out_path = out_dir / filename
            _plot(coords, block.labels, f"{method.upper()} - {block.name}", out_path)
            generated.append(str(out_path))

            for sample_id, label, xy in zip(block.sample_ids, block.labels, coords):
                coord_rows.append(
                    {
                        "method": method,
                        "feature_set": block.name,
                        "sample_id": sample_id,
                        "dataset": label,
                        "x": float(xy[0]),
                        "y": float(xy[1]),
                    }
                )

    coordinates_path = out_dir / "embedding_coordinates.csv"
    pd.DataFrame(coord_rows).to_csv(coordinates_path, index=False)

    summary = {
        "private_accepted_samples_requested": len(private),
        "cmose_samples_requested": len(cmose),
        "plotted_samples": {
            "private": int(sum(label == "private" for label in openface.labels)),
            "CMOSE": int(sum(label == "CMOSE" for label in openface.labels)),
        },
        "openface_feature_dim": int(openface.matrix.shape[1]),
        "i3d_feature_dim": int(i3d.matrix.shape[1]),
        "fused_feature_dim": int(fused.matrix.shape[1]),
        "openface_filter": {
            "require_success": args.require_success,
            "min_confidence": args.min_confidence,
        },
        "dropped_samples": {k: len(v) for k, v in dropped.items()},
        "methods": ["tsne"] + (["umap"] if any("umap_" in Path(p).name for p in generated) else []),
        "generated": generated + [str(coordinates_path)],
    }
    summary_path = out_dir / "visualization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_path = out_dir / "visualization_summary.md"
    md_lines = [
        "# CMOSE vs Accepted Private Feature-Space Visualization",
        "",
        f"- Private accepted samples requested: {summary['private_accepted_samples_requested']}",
        f"- CMOSE samples requested: {summary['cmose_samples_requested']}",
        f"- Plotted private samples: {summary['plotted_samples']['private']}",
        f"- Plotted CMOSE samples: {summary['plotted_samples']['CMOSE']}",
        f"- OpenFace dimensions: {summary['openface_feature_dim']}",
        f"- I3D dimensions: {summary['i3d_feature_dim']}",
        f"- OpenFace + I3D dimensions: {summary['fused_feature_dim']}",
        f"- OpenFace filter: success required = {args.require_success}, min confidence = {args.min_confidence}",
        f"- Embedding methods generated: {', '.join(summary['methods'])}",
        "",
        "## Files",
        "",
    ]
    md_lines.extend(f"- `{Path(path).name}`" for path in summary["generated"])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"wrote {summary_path}")
    print(f"wrote {md_path}")
    for path in generated:
        print(f"wrote {path}")
    print(f"wrote {coordinates_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-accepted-csv", default="data/private/accepted.csv")
    parser.add_argument(
        "--output-dir",
        default=str(DOMAIN_DIFFERENCE_DIR / "feature_space_visualizations"),
    )
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on accepted private samples.")
    parser.add_argument("--pca-dims", type=int, default=50)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--require-success", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
