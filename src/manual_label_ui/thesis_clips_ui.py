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
from typing import Any
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.style import CLASS_COLORS, CLASS_LABELS
from src.output_paths import NAIVE_ASSESSMENT_DIR, MANUAL_LABELS_CSV

LABELS = [
    {"id": index, "name": label, "color": CLASS_COLORS[label]}
    for index, label in enumerate(CLASS_LABELS)
]
LABEL_BY_ID = {row["id"]: row["name"] for row in LABELS}

THESIS_TAGS = ["easy", "hard", "cross_group", "interesting"]

OUTPUT_COLUMNS = ["clip_id", "thesis_tag", "thesis_notes", "agreement_rate", "majority_label"]

# Naive model prefixes (run keys starting with these belong to the Naive group)
_NAIVE_PREFIXES = (
    "openface_mlp",
    "tcn",
    "temporal_cnn",
    "lstm",
    "transformer",
    "i3d_mlp",
    "openface_tcn_i3d_fusion",
)


def _model_group(run_key: str) -> str:
    if "openface_temporal_i3d_hybrid" in run_key:
        return "OF+I3D-Hybrid"
    if "openface_temporal_hybrid" in run_key:
        return "OF-Hybrid"
    return "Naive"


def _group_css_class(group: str) -> str:
    return {"Naive": "g-naive", "OF-Hybrid": "g-of-hybrid", "OF+I3D-Hybrid": "g-of-i3d-hybrid"}.get(group, "g-naive")


@dataclass(frozen=True)
class Config:
    repo_root: Path
    predictions_csv: Path
    accepted_csv: Path
    manual_labels_csv: Path
    output_csv: Path
    host: str
    port: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Browse per-clip model predictions; tag clips for thesis examples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions_csv",
        default=str(NAIVE_ASSESSMENT_DIR / "predictions_by_clip.csv"),
        help="Wide per-clip predictions CSV (predicted_label__<run> columns).",
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


def _detect_runs(row: dict[str, str]) -> list[str]:
    seen: set[str] = set()
    keys: list[str] = []
    for col in row:
        if col.startswith("predicted_label__"):
            k = col[len("predicted_label__"):]
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


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


def _build_clips(config: Config) -> dict[str, Any]:
    pred_rows = _read_csv(config.predictions_csv)
    if not pred_rows:
        return {"labels": LABELS, "clips": [], "groups": [], "predictions_csv": str(config.predictions_csv), "output_csv": str(config.output_csv)}

    run_keys = _detect_runs(pred_rows[0])
    if not run_keys:
        raise ValueError(f"No predicted_label__ columns in {config.predictions_csv}")

    group_map = {rk: _model_group(rk) for rk in run_keys}

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
    for row in pred_rows:
        clip_id = row.get("clip_id", "")
        if not clip_id:
            continue

        # Dataset true label (available for CMOSE/DaiSEE splits, not for private)
        true_id_str = row.get("true_id", "")
        true_id = _to_int(true_id_str) if true_id_str else None
        true_label = row.get("true_label", "") or (LABEL_BY_ID.get(true_id, "") if true_id is not None else "")

        man = manual.get(clip_id)
        manual_id = _to_int(man.get("manual_label_id")) if man else None
        manual_label = man.get("manual_label", "") if man else ""

        # Best available ground truth
        gt_id = manual_id if manual_id is not None else true_id
        gt_label = manual_label or true_label
        gt_source = "manual" if manual_id is not None else ("dataset" if true_id is not None else "")

        # Per-model predictions
        preds: list[dict[str, Any]] = []
        for rk in run_keys:
            pred_lbl = row.get(f"predicted_label__{rk}", "")
            if not pred_lbl:
                continue
            preds.append({
                "run": rk,
                "group": group_map[rk],
                "group_css": _group_css_class(group_map[rk]),
                "label": pred_lbl,
                "label_id": _to_int(row.get(f"predicted_id__{rk}")),
                "confidence": _to_float(row.get(f"confidence__{rk}")),
            })
        if not preds:
            continue

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

    groups = sorted({group_map[rk] for rk in run_keys})
    tagged = sum(1 for c in clips if c["thesis_tag"])
    return {
        "labels": LABELS,
        "clips": clips,
        "groups": groups,
        "tagged_count": tagged,
        "total_count": len(clips),
        "predictions_csv": str(config.predictions_csv),
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

    <div class="card" id="groupCard" style="display:none">
      <h3>Model Groups</h3>
      <div id="groupContent"></div>
    </div>

    <div class="card">
      <h3>Model Predictions</h3>
      <table>
        <thead><tr><th>Run</th><th>Group</th><th>Prediction</th><th class="conf">Conf</th></tr></thead>
        <tbody id="predTbody"></tbody>
      </table>
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

  async function loadData() {
    const r = await fetch('/api/data');
    data = await r.json();
    renderFilters();
    applyFilter('all');
    updateStatus();
  }

  function updateStatus() {
    el('status').textContent = `${data.tagged_count||0}/${data.total_count||0} tagged → ${data.output_csv}`;
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

  function applyFilter(filter) {
    activeFilter = filter;
    const q = el('search').value.trim().toLowerCase();
    filtered = data.clips.filter(clip => {
      if (q && !clip.clip_id.toLowerCase().includes(q)) return false;
      if (filter === 'all') return true;
      if (filter === 'untagged') return !clip.thesis_tag;
      if (filter === 'tagged') return !!clip.thesis_tag;
      return clip.categories.includes(filter);
    });
    activeIndex = 0;
    renderList();
    showClip(0);
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

    // Group summary
    const gm = clip.group_majorities || {};
    const gs = Object.keys(gm);
    if (gs.length >= 2) {
      const allSame = new Set(gs.map(g => gm[g].id)).size === 1;
      el('groupCard').style.display = '';
      el('groupContent').innerHTML = gs.map(g => {
        const pred = gm[g];
        return `<div class="group-row">
          <span class="${pred.css}">${g}</span>
          <span class="lchip" style="--lc:${cm[pred.label]||'var(--line)'}">${pred.label}</span>
        </div>`;
      }).join('') + (!allSame ? `<div class="group-split-warn">Groups disagree</div>` : '');
    } else {
      el('groupCard').style.display = 'none';
    }

    // Predictions table
    el('predTbody').innerHTML = clip.predictions.map(p => {
      const c = cm[p.label] || 'var(--line)';
      const short = p.run.length > 24 ? p.run.slice(0,24)+'…' : p.run;
      return `<tr>
        <td title="${p.run}">${short}</td>
        <td><span class="${p.group_css}">${p.group}</span></td>
        <td><span class="lchip" style="--lc:${c}">${p.label}</span></td>
        <td class="conf">${pct(p.confidence)}</td>
      </tr>`;
    }).join('');

    el('thesisNotes').value = clip.thesis_notes || '';
    paintTag();
    renderList();
    el('msg').textContent = '';
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
                return self._json(_build_clips(self.server.config))
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
    return Config(
        repo_root=root,
        predictions_csv=_resolve(root, args.predictions_csv),
        accepted_csv=_resolve(root, args.accepted_csv),
        manual_labels_csv=_resolve(root, args.manual_labels_csv),
        output_csv=_resolve(root, args.output_csv),
        host=args.host,
        port=args.port,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _build_config(args)
    _build_clips(config)   # fail fast if predictions CSV is missing / malformed
    server = ThesisClipServer((config.host, config.port), config)
    url = f"http://{config.host}:{config.port}/"
    print(f"Thesis clip browser: {url}")
    print(f"Predictions CSV:     {config.predictions_csv}")
    print(f"Output CSV:          {config.output_csv}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
