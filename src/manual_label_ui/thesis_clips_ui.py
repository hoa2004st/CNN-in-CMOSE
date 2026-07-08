"""Thesis clip browser — browse per-clip model predictions and tag clips for thesis.

Reads a wide predictions_by_clip.csv (with predicted_label__<run> columns),
shows all model predictions side by side, computes agreement/difficulty
categories, and lets you mark clips as thesis examples with a tag and notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import sys
from collections import Counter
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.style import CLASS_COLORS, CLASS_LABELS
from src.output_paths import (
    NAIVE_ASSESSMENT_DIR,
    HYBRID_ASSESSMENT_DIR,
    MANUAL_LABELS_CSV,
)

# Long-format prediction matrices (one row per model x clip), filtered at load
# time to a single (train_group, test_set) slice. The naive file holds the simple
# per-modality baseline models; the hybrid file enumerates many fusion
# architectures, which are all kept and ranked by QWK so the UI shows one per
# hybrid family (defaulting to the best QWK arch) and can switch it on the spot.
NAIVE_MATRIX_CSV = NAIVE_ASSESSMENT_DIR / "full_matrix_predictions.csv"
HYBRID_MATRIX_CSV = HYBRID_ASSESSMENT_DIR / "hybrid_matrix_predictions.csv"

ALL_LOSSES = ("ce", "ordinal", "weighted_ce")

LABELS = [
    {"id": index, "name": label, "color": CLASS_COLORS[label]}
    for index, label in enumerate(CLASS_LABELS)
]
LABEL_BY_ID = {row["id"]: row["name"] for row in LABELS}

THESIS_TAGS = ["easy", "hard", "cross_group", "interesting"]

OUTPUT_COLUMNS = ["clip_id", "thesis_tag", "thesis_notes", "agreement_rate", "majority_label"]


def _model_group(model: str) -> str:
    if "openface_temporal_i3d_hybrid" in model:
        return "OF+I3D-Hybrid"
    if "openface_temporal_hybrid" in model:
        return "OF-Hybrid"
    return "Naive"


def _group_css_class(group: str) -> str:
    return {"Naive": "g-naive", "OF-Hybrid": "g-of-hybrid", "OF+I3D-Hybrid": "g-of-i3d-hybrid"}.get(group, "g-naive")


@dataclass(frozen=True)
class Config:
    repo_root: Path
    naive_csv: Path
    hybrid_csv: Path
    accepted_csv: Path
    manual_labels_csv: Path
    output_csv: Path
    train_group: str
    test_set: str
    losses: tuple[str, ...]
    host: str
    port: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Browse per-clip model predictions; tag clips for thesis examples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--naive_csv",
        default=str(NAIVE_MATRIX_CSV),
        help="Long-format naive predictions matrix (train_group,test_set,model,loss,clip_id,...).",
    )
    parser.add_argument(
        "--hybrid_csv",
        default=str(HYBRID_MATRIX_CSV),
        help="Long-format hybrid predictions matrix (adds model_type,arch_key columns).",
    )
    parser.add_argument(
        "--train_group",
        default="combined",
        help="Training regime to show (train_group column): e.g. combined, cmose, daisee.",
    )
    parser.add_argument(
        "--test_set",
        default="private",
        help="Test set whose clips to browse (test_set column): private, cmose_test, daisee_test.",
    )
    parser.add_argument(
        "--losses",
        default="ce",
        help="Comma-separated loss names to include (ce,ordinal,weighted_ce), or 'all'.",
    )
    parser.add_argument("--accepted_csv", default="data/private/accepted.csv")
    parser.add_argument("--manual_labels_csv", default=str(MANUAL_LABELS_CSV))
    parser.add_argument(
        "--output_csv",
        default="data/private/thesis_clips.csv",
        help="CSV to write thesis tag selections.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    return parser


def _resolve_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def _to_float(v: str | None, default: float = 0.0) -> float:
    try:
        return float(v) if v not in (None, "") else default
    except ValueError:
        return default


def _to_int(v: str | None, default: int = 0) -> int:
    try:
        return int(float(v)) if v not in (None, "") else default
    except ValueError:
        return default


def _agreement(votes: list[int]) -> tuple[float, int, int]:
    n = len(votes)
    if n == 0:
        return 1.0, 0, 0
    counts = Counter(votes)
    maj = max(counts, key=lambda k: (counts[k], -k))
    total_pairs = n * (n - 1) // 2
    agreeing = sum(c * (c - 1) // 2 for c in counts.values())
    rate = agreeing / total_pairs if total_pairs > 0 else 1.0
    return rate, maj, counts[maj]


def _entropy(votes: list[int], n_classes: int = 4) -> float:
    n = len(votes)
    if n == 0:
        return 0.0
    probs = [c / n for c in Counter(votes).values()]
    raw = -sum(p * math.log(p) for p in probs if p > 0)
    return raw / math.log(n_classes) if n_classes > 1 else 0.0


def _load_tags(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {r["clip_id"]: r for r in _read_csv(path) if r.get("clip_id")}


def _upsert_tag(config: Config, payload: dict[str, Any]) -> dict[str, str]:
    clip_id = str(payload.get("clip_id", "")).strip()
    tag = str(payload.get("thesis_tag", "")).strip()
    notes = str(payload.get("thesis_notes", "")).strip()
    if not clip_id:
        raise ValueError("clip_id is required")
    if tag and tag not in THESIS_TAGS:
        raise ValueError(f"thesis_tag must be one of {THESIS_TAGS} or empty string")
    existing = _load_tags(config.output_csv)
    row: dict[str, str] = {
        "clip_id": clip_id,
        "thesis_tag": tag,
        "thesis_notes": notes,
        "agreement_rate": str(payload.get("agreement_rate", "")),
        "majority_label": str(payload.get("majority_label", "")),
    }
    if tag:
        existing[clip_id] = row
    else:
        existing.pop(clip_id, None)
    _write_csv(config.output_csv, [existing[k] for k in sorted(existing)], OUTPUT_COLUMNS)
    return row


# The two hybrid families the browser shows one architecture from, plus the stable
# query-param key each arch selector uses. The baseline (Naive) group is always
# shown in full and has no architecture choice.
HYBRID_GROUPS = ("OF-Hybrid", "OF+I3D-Hybrid")
GROUP_PARAM = {"OF-Hybrid": "arch_ofhybrid", "OF+I3D-Hybrid": "arch_ofi3d"}


def _iter_matrix_rows(path: Path, prefix: str) -> Iterator[list[str]]:
    # These matrices contain no quoted or comma-bearing fields, so a plain split is
    # both correct and much faster than csv.reader across millions of rows. The raw
    # `prefix` (train_group,test_set,) is matched before splitting so the vast
    # majority of the multi-million-row hybrid file is skipped cheaply.
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        if not f.readline():  # skip header
            return
        for line in f:
            if not line.startswith(prefix):
                continue
            line = line.rstrip("\r\n")
            if line:
                yield line.split(",")


def _qwk(mat: list[list[int]], k: int = 4) -> float | None:
    """Quadratic weighted kappa from a k x k confusion matrix (rows=true, cols=pred)."""
    n = sum(sum(row) for row in mat)
    if n == 0:
        return None
    row_tot = [sum(mat[i]) for i in range(k)]
    col_tot = [sum(mat[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    denom_w = (k - 1) ** 2
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / denom_w
            num += w * mat[i][j]
            den += w * (row_tot[i] * col_tot[j] / n)
    if den == 0:
        return 1.0
    return 1.0 - num / den


@dataclass
class SliceData:
    clip_true: dict[str, int | None]                        # clip_id -> true class id or None
    order: list[str]                                        # clip ids, first-seen order
    baseline: dict[str, list[dict[str, Any]]]               # clip_id -> naive pred dicts
    hybrid: dict[str, dict[str, dict[str, list[tuple]]]]    # group -> arch -> clip -> [(loss,pid,conf,model)]
    qwk: dict[str, dict[str, float | None]]                 # group -> arch -> qwk
    archs_ranked: dict[str, list[str]]                      # group -> archs sorted by qwk desc
    best_arch: dict[str, str]                               # group -> best-QWK arch


# Cache the parsed slice so the multi-million-row hybrid matrix is scanned once per
# (files + slice), not on every /api/data request or arch switch.
_SLICE_CACHE: dict[tuple, SliceData] = {}


def _pred_dict(run: str, model: str, pred_id: int, conf: float) -> dict[str, Any]:
    group = _model_group(model)
    return {
        "run": run,
        "group": group,
        "group_css": _group_css_class(group),
        "label": LABEL_BY_ID.get(pred_id, ""),
        "label_id": pred_id,
        "confidence": conf,
    }


def _load_slice(config: Config) -> SliceData:
    """Filter both matrices to one (train_group, test_set) slice, keeping every
    hybrid architecture in memory and ranking them by QWK so the UI can switch
    architecture on the spot."""
    key = (str(config.naive_csv), str(config.hybrid_csv),
           config.train_group, config.test_set, config.losses)
    cached = _SLICE_CACHE.get(key)
    if cached is not None:
        return cached

    losses = set(config.losses)
    primary_loss = config.losses[0] if config.losses else "ce"
    prefix = f"{config.train_group},{config.test_set},"

    clip_true: dict[str, int | None] = {}
    order: list[str] = []
    baseline: dict[str, list[dict[str, Any]]] = {}
    hybrid: dict[str, dict[str, dict[str, list[tuple]]]] = {g: {} for g in HYBRID_GROUPS}
    conf: dict[str, dict[str, list[list[int]]]] = {g: {} for g in HYBRID_GROUPS}

    def _note(clip_id: str, true_raw: str) -> None:
        if clip_id not in clip_true:
            clip_true[clip_id] = _to_int(true_raw) if true_raw not in ("", None) else None
            order.append(clip_id)

    # Naive: train_group,test_set,model,loss,clip_id,true_id,predicted_id,is_correct,confidence,...
    if config.naive_csv.exists():
        for r in _iter_matrix_rows(config.naive_csv, prefix):
            if len(r) < 9:
                continue
            model, loss, clip_id = r[2], r[3], r[4]
            if not clip_id or loss not in losses:
                continue
            _note(clip_id, r[5])
            baseline.setdefault(clip_id, []).append(
                _pred_dict(f"{model}/{loss}", model, _to_int(r[6]), _to_float(r[8]))
            )

    # Hybrid: train_group,test_set,model,model_type,loss,arch_key,clip_id,true_id,predicted_id,is_correct,confidence,...
    if config.hybrid_csv.exists():
        for r in _iter_matrix_rows(config.hybrid_csv, prefix):
            if len(r) < 11:
                continue
            model, loss, arch, clip_id = r[2], r[4], r[5], r[6]
            if not clip_id or loss not in losses:
                continue
            group = _model_group(model)
            if group not in hybrid:
                continue
            _note(clip_id, r[7])
            pid, c = _to_int(r[8]), _to_float(r[10])
            hybrid[group].setdefault(arch, {}).setdefault(clip_id, []).append((loss, pid, c, model))
            if loss == primary_loss:
                t = clip_true.get(clip_id)
                if t is not None and 0 <= t < 4 and 0 <= pid < 4:
                    mat = conf[group].setdefault(arch, [[0] * 4 for _ in range(4)])
                    mat[t][pid] += 1

    qwk = {g: {a: _qwk(conf[g].get(a, [[0] * 4 for _ in range(4)])) for a in hybrid[g]}
           for g in HYBRID_GROUPS}
    archs_ranked: dict[str, list[str]] = {}
    best_arch: dict[str, str] = {}
    for g in HYBRID_GROUPS:
        archs = sorted(
            hybrid[g].keys(),
            key=lambda a: (-(qwk[g][a] if qwk[g][a] is not None else float("-inf")), a),
        )
        archs_ranked[g] = archs
        best_arch[g] = archs[0] if archs else ""

    data = SliceData(clip_true, order, baseline, hybrid, qwk, archs_ranked, best_arch)
    _SLICE_CACHE[key] = data
    return data


def _build_clips(config: Config, selected: dict[str, str] | None = None) -> dict[str, Any]:
    s = _load_slice(config)
    selected = selected or {}
    # Resolve the arch shown for each hybrid group: caller's choice if valid, else best QWK.
    chosen: dict[str, str] = {}
    for g in HYBRID_GROUPS:
        a = selected.get(g)
        chosen[g] = a if a in s.hybrid[g] else s.best_arch[g]

    video_paths: dict[str, str] = {}
    if config.accepted_csv.exists():
        for row in _read_csv(config.accepted_csv):
            cid, cp = row.get("clip_id", ""), row.get("clip_path", "")
            if cid and cp:
                video_paths[cid] = Path(cp).as_posix()

    manual: dict[str, dict[str, str]] = {}
    if config.manual_labels_csv.exists():
        for row in _read_csv(config.manual_labels_csv):
            cid = row.get("clip_id", "")
            if cid:
                manual[cid] = row

    saved_tags = _load_tags(config.output_csv)

    clips: list[dict[str, Any]] = []
    for clip_id in s.order:
        # Baseline (all naive models) + the chosen architecture from each hybrid group.
        preds: list[dict[str, Any]] = list(s.baseline.get(clip_id, []))
        for g in HYBRID_GROUPS:
            for loss, pid, conf, model in s.hybrid[g].get(chosen[g], {}).get(clip_id, ()):
                preds.append(_pred_dict(f"{model}/{loss}", model, pid, conf))
        if not preds:
            continue

        # Dataset true label (now populated for private clips too)
        true_id = s.clip_true.get(clip_id)
        true_label = LABEL_BY_ID.get(true_id, "") if true_id is not None else ""

        man = manual.get(clip_id)
        manual_id = _to_int(man.get("manual_label_id")) if man else None
        manual_label = man.get("manual_label", "") if man else ""

        # Best available ground truth
        gt_id = manual_id if manual_id is not None else true_id
        gt_label = manual_label or true_label
        gt_source = "manual" if manual_id is not None else ("dataset" if true_id is not None else "")

        votes = [p["label_id"] for p in preds]
        agreement_rate, majority_id, majority_count = _agreement(votes)
        entropy = _entropy(votes)
        majority_label = LABEL_BY_ID.get(majority_id, "")

        # Per-group majority
        group_votes: dict[str, list[int]] = {}
        for p in preds:
            group_votes.setdefault(p["group"], []).append(p["label_id"])
        group_maj: dict[str, dict[str, Any]] = {}
        for g, gv in group_votes.items():
            _, gid, _ = _agreement(gv)
            group_maj[g] = {"id": gid, "label": LABEL_BY_ID.get(gid, ""), "css": _group_css_class(g)}

        # Category tags
        cats: list[str] = []
        if agreement_rate >= 0.75:
            cats.append("high_agree")
        if agreement_rate == 1.0:
            cats.append("all_agree")
        if agreement_rate <= 0.33:
            cats.append("high_disagree")
        elif agreement_rate <= 0.5:
            cats.append("split")

        if gt_id is not None:
            if majority_id == gt_id:
                if agreement_rate >= 0.75:
                    cats.append("easy")
                if agreement_rate == 1.0:
                    cats.append("all_correct")
            else:
                cats.append("majority_wrong")
                if agreement_rate == 1.0:
                    cats.append("all_wrong")
                if agreement_rate <= 0.5:
                    cats.append("hard")

        if len(group_maj) >= 2 and len({v["id"] for v in group_maj.values()}) > 1:
            cats.append("cross_group")

        tag_row = saved_tags.get(clip_id)
        thesis_tag = tag_row.get("thesis_tag", "") if tag_row else ""
        thesis_notes = tag_row.get("thesis_notes", "") if tag_row else ""

        rel_path = video_paths.get(clip_id, "")
        clips.append({
            "clip_id": clip_id,
            "video_url": ("/media/" + rel_path) if rel_path else "",
            "gt_id": gt_id,
            "gt_label": gt_label,
            "gt_source": gt_source,
            "manual_label_id": manual_id,
            "manual_label": manual_label,
            "predictions": preds,
            "agreement_rate": agreement_rate,
            "prediction_entropy": entropy,
            "majority_label_id": majority_id,
            "majority_label": majority_label,
            "majority_count": majority_count,
            "num_models": len(preds),
            "group_majorities": group_maj,
            "categories": cats,
            "thesis_tag": thesis_tag,
            "thesis_notes": thesis_notes,
        })

    # Tagged clips first, then by clip_id
    clips.sort(key=lambda c: (not c["thesis_tag"], c["clip_id"]))

    groups = sorted({p["group"] for c in clips for p in c["predictions"]})
    tagged = sum(1 for c in clips if c["thesis_tag"])
    # Per-hybrid-group architecture menus (ranked by QWK) for the on-the-spot selectors.
    hybrids: dict[str, Any] = {}
    for g in HYBRID_GROUPS:
        hybrids[GROUP_PARAM[g]] = {
            "group": g,
            "selected": chosen[g],
            "archs": [{"arch": a, "qwk": s.qwk[g][a]} for a in s.archs_ranked[g]],
        }
    return {
        "labels": LABELS,
        "clips": clips,
        "groups": groups,
        "hybrids": hybrids,
        "tagged_count": tagged,
        "total_count": len(clips),
        "slice": f"{config.train_group} / {config.test_set}",
        "predictions_csv": f"{config.naive_csv.name} + {config.hybrid_csv.name}",
        "output_csv": str(config.output_csv),
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Thesis Clip Browser</title>
  <style>
    :root { color-scheme:light; --line:#d7dde5; --ink:#17202a; --muted:#5f6b7a; --bg:#f6f8fb; --panel:#fff; --accent:#1769aa; --ok:#1f7a4d; --warn:#c97b00; --danger:#c0392b; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Segoe UI,Arial,sans-serif; color:var(--ink); background:var(--bg); }
    header { height:50px; display:flex; align-items:center; justify-content:space-between; padding:0 16px; border-bottom:1px solid var(--line); background:var(--panel); }
    header h1 { font-size:16px; margin:0; font-weight:700; }
    .hdr-status { color:var(--muted); font-size:12px; }
    main { display:grid; grid-template-columns:280px 1fr 340px; height:calc(100vh - 50px); overflow:hidden; }

    /* ---- left sidebar ---- */
    aside.sidebar { border-right:1px solid var(--line); overflow:hidden; display:flex; flex-direction:column; background:var(--panel); }
    .filter-wrap { padding:8px; border-bottom:1px solid var(--line); flex-shrink:0; }
    .filter-wrap input { width:100%; height:30px; border:1px solid var(--line); border-radius:5px; padding:0 8px; font-size:12px; margin-bottom:6px; }
    .cat-chips { display:flex; flex-wrap:wrap; gap:3px; }
    .cat-chip { border:1px solid var(--line); background:#fff; border-radius:10px; padding:2px 7px; font-size:10.5px; cursor:pointer; white-space:nowrap; color:var(--ink); }
    .cat-chip.active { background:var(--accent); color:#fff; border-color:var(--accent); }
    .clip-scroll { overflow-y:auto; flex:1; }
    .clip-item { display:block; width:100%; text-align:left; border:0; border-bottom:1px solid #edf0f4; background:transparent; padding:8px 10px; cursor:pointer; }
    .clip-item:hover,.clip-item.active { background:#eef5fb; }
    .ci-head { display:flex; justify-content:space-between; align-items:center; font-size:12px; font-weight:700; gap:4px; }
    .ci-badges { display:flex; flex-wrap:wrap; gap:3px; margin-top:3px; }
    .badge { font-size:9.5px; padding:1px 4px; border-radius:3px; font-weight:700; }
    .b-easy { background:#d4edda; color:#155724; }
    .b-all_correct { background:#b8f0c8; color:#0a4020; }
    .b-hard { background:#f8d7da; color:#721c24; }
    .b-all_wrong { background:#f5b0b5; color:#4d0a0a; }
    .b-majority_wrong { background:#fde8e8; color:#882020; }
    .b-all_agree { background:#cce5ff; color:#004085; }
    .b-cross_group { background:#fff3cd; color:#856404; }
    .b-high_disagree { background:#ffeeba; color:#664d03; }
    .b-split { background:#e8eaf0; color:#3d4466; }
    .b-high_agree { background:#d6f0e8; color:#145040; }
    .tag-chip { font-size:10px; padding:1px 5px; border-radius:3px; font-weight:700; }
    .t-easy { background:#d4edda; color:#155724; }
    .t-hard { background:#f8d7da; color:#721c24; }
    .t-interesting { background:#e8d8f8; color:#5a2a9e; }
    .t-cross_group { background:#fff3cd; color:#856404; }

    /* ---- center ---- */
    .center { overflow-y:auto; padding:12px 14px; display:flex; flex-direction:column; gap:8px; }
    .clip-title-main { margin:0; font-size:14px; font-weight:700; }
    video { width:100%; max-height:calc(100vh - 160px); background:#000; border:1px solid var(--line); border-radius:6px; display:block; }
    .no-video { width:100%; aspect-ratio:16/9; background:#1a1a1a; border-radius:6px; display:flex; align-items:center; justify-content:center; color:#666; font-size:13px; }

    /* ---- right panel ---- */
    .info-panel { border-left:1px solid var(--line); overflow-y:auto; background:var(--panel); padding:10px; display:flex; flex-direction:column; gap:8px; }
    .card { border:1px solid var(--line); border-radius:7px; padding:9px 10px; }
    .card h3 { margin:0 0 7px; font-size:11px; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
    .stat-row { display:grid; grid-template-columns:1fr 1fr; gap:5px; }
    .stat-box { background:#fbfcfe; border:1px solid var(--line); border-radius:5px; padding:5px 7px; }
    .stat-box span { display:block; font-size:10px; color:var(--muted); }
    .stat-box strong { display:block; font-size:12.5px; margin-top:1px; }
    table { width:100%; border-collapse:collapse; }
    th,td { padding:4px 5px; border-bottom:1px solid #edf0f4; font-size:11px; text-align:left; }
    th { font-weight:700; color:var(--muted); font-size:10.5px; }
    .lchip { display:inline-block; border-left:3px solid var(--lc,var(--line)); padding-left:4px; font-weight:700; }
    .conf { text-align:right; color:var(--muted); }
    .g-naive { background:#dbeafe; color:#1e40af; font-size:10px; padding:1px 4px; border-radius:3px; }
    .g-of-hybrid { background:#dcfce7; color:#166534; font-size:10px; padding:1px 4px; border-radius:3px; }
    .g-of-i3d-hybrid { background:#fef9c3; color:#854d0e; font-size:10px; padding:1px 4px; border-radius:3px; }
    .gt-block { font-size:12px; display:flex; align-items:center; gap:6px; }
    .gt-src { font-size:10px; color:var(--muted); }
    .group-row { display:flex; align-items:center; gap:6px; margin-bottom:4px; font-size:11px; }
    .group-split-warn { font-size:10px; color:var(--warn); margin-top:3px; }
    /* grouped predictions: summary bar + expandable model rows */
    .grp { border:1px solid var(--line); border-radius:6px; margin-bottom:7px; overflow:hidden; }
    .grp-head { display:flex; align-items:center; gap:7px; width:100%; border:0; background:#fbfcfe; padding:6px 8px; cursor:pointer; text-align:left; font:inherit; }
    .grp-head:hover { background:#f0f6ff; }
    .grp-caret { font-size:9px; color:var(--muted); transition:transform .15s; flex-shrink:0; width:9px; }
    .grp.open .grp-caret { transform:rotate(90deg); }
    .grp-name { flex-shrink:0; }
    .propbar { flex:1; display:flex; height:15px; min-width:50px; border:1px solid var(--line); border-radius:3px; overflow:hidden; background:#fff; }
    .propbar span { display:block; height:100%; }
    .grp-meta { flex-shrink:0; font-size:10px; color:var(--muted); white-space:nowrap; }
    .grp-models { display:none; border-top:1px solid var(--line); padding:2px 8px 4px; }
    .grp.open .grp-models { display:block; }
    .grp-models th,.grp-models td { padding:3px 4px; }
    .tag-buttons { display:grid; grid-template-columns:1fr 1fr; gap:5px; margin-bottom:7px; }
    .tag-btn { border:1px solid var(--line); border-radius:5px; background:#fff; padding:7px; cursor:pointer; font-size:11.5px; font-weight:700; text-align:center; }
    .tag-btn:hover { background:#f0f6ff; }
    .tag-btn.sel { box-shadow:inset 0 0 0 2px var(--accent); background:#f0f6ff; color:var(--accent); }
    .clear-btn { color:var(--muted); grid-column:span 2; }
    textarea { width:100%; min-height:60px; resize:vertical; border:1px solid var(--line); border-radius:5px; padding:5px 7px; font:inherit; font-size:11.5px; margin-bottom:7px; }
    .actions { display:flex; gap:6px; align-items:center; }
    .actions button { height:32px; border-radius:5px; border:1px solid var(--line); background:#fff; padding:0 10px; cursor:pointer; font-size:12px; }
    .actions button.primary { background:var(--accent); color:#fff; border-color:var(--accent); font-weight:700; }
    .msg { font-size:11px; color:var(--muted); }
    .arch-selectors { display:flex; flex-direction:column; gap:7px; }
    .arch-row { display:flex; flex-direction:column; gap:2px; }
    .arch-row label { font-size:10px; color:var(--muted); font-weight:700; text-transform:uppercase; letter-spacing:.03em; }
    .arch-row select { width:100%; height:28px; border:1px solid var(--line); border-radius:5px; font-size:11px; padding:0 4px; background:#fff; }
    @media (max-width:1000px) { main { grid-template-columns:240px 1fr; } .info-panel { display:none; } }
  </style>
</head>
<body>
<header>
  <h1>Thesis Clip Browser</h1>
  <span class="hdr-status" id="status">Loading…</span>
</header>
<main>
  <!-- Left sidebar -->
  <aside class="sidebar">
    <div class="filter-wrap">
      <input id="search" type="search" placeholder="Search clip id…">
      <div class="cat-chips" id="catChips"></div>
    </div>
    <div class="clip-scroll"><div id="clipList"></div></div>
  </aside>

  <!-- Center: video -->
  <div class="center">
    <h2 class="clip-title-main" id="clipTitle">Select a clip</h2>
    <div id="videoWrap"><div class="no-video">No clip selected</div></div>
  </div>

  <!-- Right: predictions + tagging -->
  <div class="info-panel">
    <div class="card">
      <h3>Statistics</h3>
      <div class="stat-row">
        <div class="stat-box"><span>Agreement</span><strong id="sAgreement">—</strong></div>
        <div class="stat-box"><span>Entropy</span><strong id="sEntropy">—</strong></div>
        <div class="stat-box"><span>Majority</span><strong id="sMajority">—</strong></div>
        <div class="stat-box"><span>Models</span><strong id="sModels">—</strong></div>
      </div>
    </div>

    <div class="card" id="gtCard">
      <h3>Ground Truth</h3>
      <div id="gtContent"><span style="color:var(--muted)">No label</span></div>
    </div>

    <div class="card">
      <h3>Hybrid Architecture</h3>
      <div id="archSelectors" class="arch-selectors"></div>
    </div>

    <div class="card">
      <h3>Predictions by Group</h3>
      <div id="groupsContent"></div>
      <div class="group-split-warn" id="groupWarn" style="display:none">Groups disagree</div>
    </div>

    <div class="card">
      <h3>Thesis Tag</h3>
      <div class="tag-buttons" id="tagButtons">
        <button class="tag-btn" data-tag="easy">Easy</button>
        <button class="tag-btn" data-tag="hard">Hard</button>
        <button class="tag-btn" data-tag="cross_group">Cross-group</button>
        <button class="tag-btn" data-tag="interesting">Interesting</button>
        <button class="tag-btn clear-btn" data-tag="">Clear tag</button>
      </div>
      <textarea id="thesisNotes" placeholder="Notes for thesis…"></textarea>
      <div class="actions">
        <button class="primary" id="btnSave">Save</button>
        <button id="btnPrev">Prev</button>
        <button id="btnNext">Next</button>
        <span class="msg" id="msg"></span>
      </div>
    </div>
  </div>
</main>
<script>
  let data = {labels:[], clips:[], groups:[]};
  let filtered = [];
  let activeIndex = 0;
  let activeFilter = 'all';
  let selectedTag = null;
  let archSel = {};          // param -> chosen arch_key (persists across reloads)
  let currentClipId = null;  // preserved across arch switches
  const GROUP_ORDER = ['Naive', 'OF-Hybrid', 'OF+I3D-Hybrid'];
  const GROUP_CSS = {'Naive':'g-naive', 'OF-Hybrid':'g-of-hybrid', 'OF+I3D-Hybrid':'g-of-i3d-hybrid'};
  let openGroups = new Set();  // model groups expanded to per-model rows

  const el = id => document.getElementById(id);
  const pct = v => isFinite(v) ? `${(v*100).toFixed(1)}%` : '—';
  const num = v => isFinite(v) ? v.toFixed(3) : '—';
  const colorMap = () => Object.fromEntries(data.labels.map(l => [l.name, l.color]));

  const CAT_DISPLAY = {
    all:'All', untagged:'Untagged', tagged:'Tagged',
    easy:'Easy', hard:'Hard', cross_group:'Cross-group',
    all_agree:'All agree', all_correct:'All correct',
    all_wrong:'All wrong', majority_wrong:'Majority wrong',
    high_disagree:'High disagree', split:'Split',
  };

  const BADGE_CATS = ['all_correct','easy','hard','all_wrong','majority_wrong','all_agree','cross_group','high_disagree','split'];

  function apiUrl() {
    const p = new URLSearchParams();
    for (const k in archSel) if (archSel[k]) p.set(k, archSel[k]);
    const qs = p.toString();
    return '/api/data' + (qs ? ('?' + qs) : '');
  }

  async function loadData(preserveId, keepFilter) {
    const r = await fetch(apiUrl());
    data = await r.json();
    if (data.hybrids) for (const k in data.hybrids) if (!archSel[k]) archSel[k] = data.hybrids[k].selected;
    renderArchSelectors();
    renderFilters();
    applyFilter(keepFilter ? activeFilter : 'all', preserveId);
    updateStatus();
  }

  function renderArchSelectors() {
    const host = el('archSelectors');
    if (!data.hybrids) { host.innerHTML = ''; return; }
    host.innerHTML = Object.keys(data.hybrids).map(k => {
      const h = data.hybrids[k];
      const cur = archSel[k] || h.selected;
      const opts = h.archs.map(a => {
        const q = (a.qwk === null || a.qwk === undefined) ? '—' : Number(a.qwk).toFixed(3);
        return `<option value="${a.arch}"${a.arch === cur ? ' selected' : ''}>${a.arch} · QWK ${q}</option>`;
      }).join('');
      return `<div class="arch-row"><label>${h.group}</label><select data-param="${k}">${opts}</select></div>`;
    }).join('');
    host.querySelectorAll('select').forEach(sel => sel.addEventListener('change', () => {
      archSel[sel.dataset.param] = sel.value;
      loadData(currentClipId, true);
    }));
  }

  function updateStatus() {
    const slice = data.slice ? `${data.slice} · ` : '';
    el('status').textContent = `${slice}${data.tagged_count||0}/${data.total_count||0} tagged → ${data.output_csv}`;
  }

  function catCounts() {
    const c = {all: data.clips.length, untagged:0, tagged:0};
    for (const clip of data.clips) {
      if (clip.thesis_tag) c.tagged = (c.tagged||0)+1; else c.untagged = (c.untagged||0)+1;
      for (const cat of clip.categories) c[cat] = (c[cat]||0)+1;
    }
    return c;
  }

  function renderFilters() {
    const c = catCounts();
    const order = ['all','untagged','tagged','easy','hard','cross_group','all_agree','all_correct','all_wrong','majority_wrong','high_disagree','split'];
    el('catChips').innerHTML = order
      .filter(f => (c[f]||0) > 0 || f==='all')
      .map(f => `<button class="cat-chip${f===activeFilter?' active':''}" data-f="${f}">${CAT_DISPLAY[f]||f} (${c[f]||0})</button>`)
      .join('');
    el('catChips').querySelectorAll('button').forEach(b => b.addEventListener('click', () => applyFilter(b.dataset.f)));
  }

  function applyFilter(filter, preserveId) {
    activeFilter = filter;
    const q = el('search').value.trim().toLowerCase();
    filtered = data.clips.filter(clip => {
      if (q && !clip.clip_id.toLowerCase().includes(q)) return false;
      if (filter === 'all') return true;
      if (filter === 'untagged') return !clip.thesis_tag;
      if (filter === 'tagged') return !!clip.thesis_tag;
      return clip.categories.includes(filter);
    });
    let idx = 0;
    if (preserveId) { const i = filtered.findIndex(c => c.clip_id === preserveId); if (i >= 0) idx = i; }
    activeIndex = idx;
    renderList();
    showClip(idx);
    renderFilters();
  }

  function renderList() {
    el('clipList').innerHTML = filtered.map((clip, i) => {
      const badges = BADGE_CATS
        .filter(cat => clip.categories.includes(cat))
        .map(cat => `<span class="badge b-${cat}">${CAT_DISPLAY[cat]||cat}</span>`)
        .join('');
      const tagHtml = clip.thesis_tag
        ? `<span class="tag-chip t-${clip.thesis_tag}">${clip.thesis_tag}</span>` : '';
      return `<button class="clip-item${i===activeIndex?' active':''}" data-i="${i}">
        <div class="ci-head"><span>${clip.clip_id}</span>${tagHtml}</div>
        ${badges ? `<div class="ci-badges">${badges}</div>` : ''}
      </button>`;
    }).join('');
    el('clipList').querySelectorAll('button').forEach(b => b.addEventListener('click', () => showClip(+b.dataset.i)));
  }

  function showClip(index) {
    if (!filtered.length) return;
    activeIndex = Math.max(0, Math.min(index, filtered.length-1));
    const clip = filtered[activeIndex];
    currentClipId = clip.clip_id;
    selectedTag = clip.thesis_tag || null;
    el('clipTitle').textContent = clip.clip_id;

    // Video
    el('videoWrap').innerHTML = clip.video_url
      ? `<video controls loop preload="metadata" src="${clip.video_url}"></video>`
      : `<div class="no-video">No video for this clip</div>`;

    // Stats
    const cm = colorMap();
    el('sAgreement').textContent = pct(clip.agreement_rate);
    el('sEntropy').textContent = num(clip.prediction_entropy);
    el('sMajority').textContent = clip.majority_label || '—';
    el('sMajority').style.color = cm[clip.majority_label] || '';
    el('sModels').textContent = String(clip.num_models);

    // Ground truth
    if (clip.gt_label) {
      el('gtContent').innerHTML = `<div class="gt-block">
        <span class="lchip" style="--lc:${cm[clip.gt_label]||'var(--line)'}">${clip.gt_label}</span>
        ${clip.gt_source ? `<span class="gt-src">(${clip.gt_source})</span>` : ''}
      </div>`;
    } else {
      el('gtContent').innerHTML = `<span style="color:var(--muted)">No label available</span>`;
    }

    // Predictions grouped by model group (summary bar + expandable rows)
    renderGroups(clip, cm);

    el('thesisNotes').value = clip.thesis_notes || '';
    paintTag();
    renderList();
    el('msg').textContent = '';
  }

  function renderGroups(clip, cm) {
    // Bucket this clip's per-model predictions by model group.
    const byGroup = {};
    for (const p of clip.predictions) (byGroup[p.group] = byGroup[p.group] || []).push(p);
    const groups = GROUP_ORDER.filter(g => byGroup[g]);
    for (const g of Object.keys(byGroup)) if (!groups.includes(g)) groups.push(g);

    const gm = clip.group_majorities || {};
    el('groupsContent').innerHTML = groups.map(g => {
      const preds = byGroup[g];
      const total = preds.length;
      const counts = {};
      for (const p of preds) counts[p.label] = (counts[p.label] || 0) + 1;
      // Majority label: prefer backend's (consistent tie-break), else most frequent here.
      const majLabel = (gm[g] && gm[g].label)
        || Object.keys(counts).sort((a, b) => counts[b] - counts[a])[0] || '';
      const majCount = counts[majLabel] || 0;
      // Stacked proportion bar, segments in fixed class order for stable colors.
      const segs = data.labels.filter(l => counts[l.name]).map(l => {
        const w = (counts[l.name] / total * 100).toFixed(2);
        return `<span style="width:${w}%;background:${l.color}" title="${l.name}: ${counts[l.name]}/${total}"></span>`;
      }).join('');
      const rows = preds.map(p => {
        const c = cm[p.label] || 'var(--line)';
        const short = p.run.length > 26 ? p.run.slice(0, 26) + '…' : p.run;
        return `<tr>
          <td title="${p.run}">${short}</td>
          <td><span class="lchip" style="--lc:${c}">${p.label}</span></td>
          <td class="conf">${pct(p.confidence)}</td>
        </tr>`;
      }).join('');
      const open = openGroups.has(g) ? ' open' : '';
      return `<div class="grp${open}" data-g="${g}">
        <button class="grp-head" type="button">
          <span class="grp-caret">▶</span>
          <span class="${GROUP_CSS[g] || 'g-naive'} grp-name">${g}</span>
          <span class="propbar">${segs}</span>
          <span class="grp-meta">${majLabel ? majLabel + ' ' : ''}${majCount}/${total}</span>
        </button>
        <div class="grp-models">
          <table>
            <thead><tr><th>Run</th><th>Prediction</th><th class="conf">Conf</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>`;
    }).join('');

    el('groupsContent').querySelectorAll('.grp-head').forEach(btn => btn.addEventListener('click', () => {
      const grp = btn.closest('.grp');
      const g = grp.dataset.g;
      if (openGroups.has(g)) openGroups.delete(g); else openGroups.add(g);
      grp.classList.toggle('open');
    }));

    // Groups-disagree warning (only meaningful with >=2 groups).
    const ids = groups.map(g => gm[g] && gm[g].id).filter(v => v !== undefined && v !== null);
    const disagree = ids.length >= 2 && new Set(ids).size > 1;
    el('groupWarn').style.display = disagree ? '' : 'none';
  }

  function paintTag() {
    el('tagButtons').querySelectorAll('button').forEach(b => {
      b.classList.toggle('sel', b.dataset.tag !== '' && b.dataset.tag === selectedTag);
    });
  }

  async function saveTag() {
    const clip = filtered[activeIndex];
    if (!clip) return;
    const payload = {
      clip_id: clip.clip_id,
      thesis_tag: selectedTag || '',
      thesis_notes: el('thesisNotes').value,
      agreement_rate: clip.agreement_rate,
      majority_label: clip.majority_label,
    };
    const r = await fetch('/api/tag', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)
    });
    if (!r.ok) { el('msg').textContent = await r.text(); return; }
    clip.thesis_tag = selectedTag || '';
    clip.thesis_notes = payload.thesis_notes;
    data.tagged_count = data.clips.filter(c => c.thesis_tag).length;
    updateStatus();
    renderList();
    renderFilters();
    el('msg').textContent = 'Saved.';
  }

  el('search').addEventListener('input', () => applyFilter(activeFilter));
  el('tagButtons').querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    selectedTag = b.dataset.tag === '' ? null : (selectedTag === b.dataset.tag ? null : b.dataset.tag);
    paintTag();
  }));
  el('btnSave').addEventListener('click', saveTag);
  el('btnPrev').addEventListener('click', () => showClip(activeIndex-1));
  el('btnNext').addEventListener('click', () => showClip(activeIndex+1));
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
    if (e.key === 'ArrowLeft') showClip(activeIndex-1);
    if (e.key === 'ArrowRight') showClip(activeIndex+1);
    if (e.key.toLowerCase() === 's') saveTag();
    if (e.key === '1') { selectedTag = 'easy'; paintTag(); }
    if (e.key === '2') { selectedTag = 'hard'; paintTag(); }
    if (e.key === '3') { selectedTag = 'cross_group'; paintTag(); }
    if (e.key === '4') { selectedTag = 'interesting'; paintTag(); }
    if (e.key === '0') { selectedTag = null; paintTag(); }
  });

  loadData().catch(err => { el('status').textContent = String(err); });
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server: "ThesisClipServer"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, text: str, status: int = 200, ct: str = "text/plain") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{ct}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/":
                return self._text(HTML, ct="text/html")
            if path == "/api/data":
                qs = parse_qs(urlparse(self.path).query)
                selected = {
                    g: qs[param][0]
                    for g, param in GROUP_PARAM.items()
                    if qs.get(param)
                }
                return self._json(_build_clips(self.server.config, selected))
            if path.startswith("/media/"):
                return self._serve_media(path[len("/media/"):])
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/tag":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            self._json(_upsert_tag(self.server.config, payload))
        except ValueError as exc:
            self._text(str(exc), status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _serve_media(self, media_path: str) -> None:
        root = self.server.config.repo_root
        requested = (root / unquote(media_path)).resolve()
        try:
            requested.relative_to(root)
        except ValueError as exc:
            raise PermissionError(f"Path outside repo: {requested}") from exc
        if not requested.exists() or not requested.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = requested.stat().st_size
        start, end = 0, size - 1
        range_hdr = self.headers.get("Range")
        status = HTTPStatus.OK
        if range_hdr:
            units, _, spec = range_hdr.partition("=")
            if units.strip() == "bytes":
                first, _, last = spec.partition("-")
                start = int(first) if first else 0
                end = int(last) if last else end
                end = min(end, size - 1)
                status = HTTPStatus.PARTIAL_CONTENT
        ct = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", ct)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with requested.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class ThesisClipServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], config: Config):
        super().__init__(address, Handler)
        self.config = config


def _build_config(args: argparse.Namespace) -> Config:
    root = _resolve_root()
    if args.losses.strip().lower() == "all":
        losses: tuple[str, ...] = ALL_LOSSES
    else:
        losses = tuple(x.strip() for x in args.losses.split(",") if x.strip())
    return Config(
        repo_root=root,
        naive_csv=_resolve(root, args.naive_csv),
        hybrid_csv=_resolve(root, args.hybrid_csv),
        accepted_csv=_resolve(root, args.accepted_csv),
        manual_labels_csv=_resolve(root, args.manual_labels_csv),
        output_csv=_resolve(root, args.output_csv),
        train_group=args.train_group,
        test_set=args.test_set,
        losses=losses,
        host=args.host,
        port=args.port,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _build_config(args)
    if not config.naive_csv.exists() and not config.hybrid_csv.exists():
        print("No prediction matrices found:")
        print(f"  naive:  {config.naive_csv}")
        print(f"  hybrid: {config.hybrid_csv}")
        sys.exit(1)
    print(
        f"Loading slice train_group={config.train_group} test_set={config.test_set} "
        f"losses={','.join(config.losses)} ..."
    )
    data = _build_clips(config)   # warms the cache / fails fast on malformed CSVs
    print(f"  {data['total_count']} clips; groups: {', '.join(data['groups']) or '(none)'}")
    for meta in data["hybrids"].values():
        qwk = next((a["qwk"] for a in meta["archs"] if a["arch"] == meta["selected"]), None)
        qwk_s = f"{qwk:.3f}" if isinstance(qwk, float) else "n/a"
        print(f"  {meta['group']}: default arch {meta['selected'] or '(none)'} (QWK {qwk_s}), "
              f"{len(meta['archs'])} archs available")
    server = ThesisClipServer((config.host, config.port), config)
    url = f"http://{config.host}:{config.port}/"
    print(f"Thesis clip browser: {url}")
    print(f"Naive CSV:  {config.naive_csv}")
    print(f"Hybrid CSV: {config.hybrid_csv}")
    print(f"Output CSV: {config.output_csv}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
