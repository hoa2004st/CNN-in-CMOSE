"""Small local web UI for manually labeling private clips.

The server intentionally uses only the Python standard library plus the CSV
files already produced by the project. It serves accepted private MP4 clips,
shows selected retained model suggestions, and writes manual labels to CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import sys
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
from src.output_paths import MANUAL_LABELS_CSV, PRIVATE_ASSESSMENT_DIR


LABELS = [
    {"id": index, "name": label, "color": CLASS_COLORS[label]}
    for index, label in enumerate(CLASS_LABELS)
]
LABEL_BY_ID = {row["id"]: row["name"] for row in LABELS}
UI_MODEL_RUNS = [
    ("fusion", ("openface_tcn_i3d_fusion/ce",)),
    ("i3d-mlp", ("i3d_mlp/ce",)),
    ("openface-transformer", ("transformer/ce",)),
    ("openface-tcn", ("tcn/weighted_ce", "temporal_cnn/weighted_ce")),
]
MANUAL_COLUMNS = [
    "clip_id",
    "manual_label_id",
    "manual_label",
    "notes",
    "source_majority_label",
    "source_agreement_rate",
    "source_mean_confidence",
]


@dataclass(frozen=True)
class UiConfig:
    repo_root: Path
    accepted_csv: Path
    agreement_csv: Path
    output_csv: Path
    host: str
    port: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local UI for manually labeling accepted private videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--accepted_csv", default="data/private/accepted.csv")
    parser.add_argument(
        "--agreement_csv",
        default=str(PRIVATE_ASSESSMENT_DIR / "predictions_by_clip.csv"),
    )
    parser.add_argument(
        "--output_csv",
        default=str(MANUAL_LABELS_CSV),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def as_int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value)) if value not in (None, "") else default
    except ValueError:
        return default


def load_manual_labels(output_csv: Path) -> dict[str, dict[str, str]]:
    if not output_csv.exists():
        return {}
    rows = read_csv_rows(output_csv)
    return {row["clip_id"]: row for row in rows if row.get("clip_id")}


def ui_agreement_metrics(suggestions: list[dict[str, Any]]) -> dict[str, Any]:
    votes = [int(item["label_id"]) for item in suggestions]
    confidences = [float(item["confidence"]) for item in suggestions]
    counts = {label["id"]: votes.count(label["id"]) for label in LABELS}
    majority_class_id = min(
        counts,
        key=lambda label_id: (-counts[label_id], label_id),
    )
    total_pairs = len(votes) * (len(votes) - 1) // 2
    agreeing_pairs = sum(count * (count - 1) // 2 for count in counts.values())
    vote_probs = [count / len(votes) for count in counts.values()]
    nonzero_vote_probs = [prob for prob in vote_probs if prob > 0.0]
    vote_entropy = -sum(prob * math.log(prob) for prob in nonzero_vote_probs)
    return {
        "num_models": len(votes),
        "agreement_rate": agreeing_pairs / total_pairs if total_pairs else 1.0,
        "prediction_entropy": vote_entropy / math.log(len(LABELS)),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "majority_class_id": majority_class_id,
        "majority_label": LABEL_BY_ID[majority_class_id],
        "majority_vote_count": counts[majority_class_id],
    }


def upsert_manual_label(config: UiConfig, payload: dict[str, Any]) -> dict[str, str]:
    clip_id = str(payload.get("clip_id", "")).strip()
    label_id = as_int(str(payload.get("manual_label_id", "")), default=-1)
    notes = str(payload.get("notes", "")).strip()
    if not clip_id:
        raise ValueError("clip_id is required")
    if label_id not in LABEL_BY_ID:
        raise ValueError(f"manual_label_id must be one of {sorted(LABEL_BY_ID)}")

    existing = load_manual_labels(config.output_csv)
    row = {
        "clip_id": clip_id,
        "manual_label_id": str(label_id),
        "manual_label": LABEL_BY_ID[label_id],
        "notes": notes,
        "source_majority_label": str(payload.get("source_majority_label", "")),
        "source_agreement_rate": str(payload.get("source_agreement_rate", "")),
        "source_mean_confidence": str(payload.get("source_mean_confidence", "")),
    }
    existing[clip_id] = row
    rows = [existing[key] for key in sorted(existing)]
    write_csv_rows(config.output_csv, rows, MANUAL_COLUMNS)
    return row


def load_ui_data(config: UiConfig) -> dict[str, Any]:
    accepted_rows = read_csv_rows(config.accepted_csv)
    accepted = {
        row["clip_id"]: row
        for row in accepted_rows
        if row.get("clip_id") and str(row.get("is_accepted", "")).strip() == "1"
    }
    agreement_rows = read_csv_rows(config.agreement_csv)
    manual = load_manual_labels(config.output_csv)

    clips: list[dict[str, Any]] = []
    for row in agreement_rows:
        clip_id = row.get("clip_id", "")
        if clip_id not in accepted:
            continue
        manifest_row = accepted[clip_id]
        suggestions = []
        for display_name, source_runs in UI_MODEL_RUNS:
            source_run = next(
                (
                    candidate
                    for candidate in source_runs
                    if f"predicted_label__{candidate}" in row
                ),
                source_runs[0],
            )
            predicted_label = row.get(f"predicted_label__{source_run}", "")
            if not predicted_label:
                raise ValueError(
                    f"{config.agreement_csv} is missing prediction columns for {source_run}"
                )
            suggestions.append(
                {
                    "run": display_name,
                    "source_run": source_run,
                    "label": predicted_label,
                    "label_id": as_int(row.get(f"predicted_id__{source_run}")),
                    "confidence": as_float(row.get(f"confidence__{source_run}")),
                }
            )
        metrics = ui_agreement_metrics(suggestions)
        clip_path = manifest_row.get("clip_path", "")
        rel_clip_path = str(Path(clip_path).as_posix()) if clip_path else ""
        manual_row = manual.get(clip_id)
        clips.append(
            {
                "clip_id": clip_id,
                "clip_path": rel_clip_path,
                "video_url": "/media/" + rel_clip_path,
                "majority_label": metrics["majority_label"],
                "majority_class_id": metrics["majority_class_id"],
                "majority_vote_count": metrics["majority_vote_count"],
                "num_models": metrics["num_models"],
                "agreement_rate": metrics["agreement_rate"],
                "prediction_entropy": metrics["prediction_entropy"],
                "mean_confidence": metrics["mean_confidence"],
                "suggestions": suggestions,
                "manual_label_id": (
                    as_int(manual_row.get("manual_label_id")) if manual_row else None
                ),
                "manual_label": manual_row.get("manual_label", "") if manual_row else "",
                "notes": manual_row.get("notes", "") if manual_row else "",
            }
        )

    clips.sort(
        key=lambda item: (
            item["manual_label_id"] is not None,
            -item["prediction_entropy"],
            item["agreement_rate"],
            item["clip_id"],
        )
    )
    return {
        "labels": LABELS,
        "clips": clips,
        "output_csv": str(config.output_csv),
        "agreement_csv": str(config.agreement_csv),
        "accepted_csv": str(config.accepted_csv),
        "labeled_count": sum(1 for item in clips if item["manual_label_id"] is not None),
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Private Video Manual Labeling</title>
  <style>
    :root { color-scheme: light; --line:#d7dde5; --ink:#17202a; --muted:#5f6b7a; --bg:#f6f8fb; --panel:#fff; --accent:#1769aa; --ok:#1f7a4d; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Segoe UI, Arial, sans-serif; color: var(--ink); background: var(--bg); }
    header { height: 56px; display: flex; align-items: center; justify-content: space-between; padding: 0 18px; border-bottom: 1px solid var(--line); background: var(--panel); }
    header h1 { font-size: 18px; margin: 0; font-weight: 650; }
    header .status { color: var(--muted); font-size: 13px; }
    main { display: grid; grid-template-columns: minmax(260px, 350px) 1fr; height: calc(100vh - 56px); }
    aside { border-right: 1px solid var(--line); overflow: auto; background: var(--panel); }
    .toolbar { position: sticky; top: 0; z-index: 1; padding: 10px; border-bottom: 1px solid var(--line); background: var(--panel); }
    input[type="search"] { width: 100%; height: 34px; border: 1px solid var(--line); border-radius: 6px; padding: 0 10px; font-size: 14px; }
    .clip-list { list-style: none; margin: 0; padding: 0; }
    .clip-item { width: 100%; text-align: left; border: 0; border-bottom: 1px solid #edf0f4; background: transparent; padding: 10px 12px; cursor: pointer; display: grid; gap: 4px; }
    .clip-item:hover, .clip-item.active { background: #eef5fb; }
    .clip-title { display: flex; justify-content: space-between; gap: 8px; font-size: 13px; font-weight: 650; }
    .clip-meta { color: var(--muted); font-size: 12px; }
    .done { color: var(--ok); font-weight: 650; }
    .workspace { overflow: auto; padding: 18px; }
    .layout { display: grid; grid-template-columns: minmax(420px, 1.35fr) minmax(320px, .9fr); gap: 18px; align-items: start; }
    video { width: 100%; max-height: calc(100vh - 160px); background: #000; border: 1px solid var(--line); border-radius: 6px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .panel h2 { margin: 0 0 10px; font-size: 16px; }
    .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px; }
    .metric { border: 1px solid var(--line); border-radius: 6px; padding: 8px; background: #fbfcfe; }
    .metric span { display: block; color: var(--muted); font-size: 11px; }
    .metric strong { display: block; margin-top: 2px; font-size: 14px; }
    table { width: 100%; border-collapse: collapse; margin-bottom: 14px; }
    th, td { padding: 8px 6px; border-bottom: 1px solid #edf0f4; font-size: 13px; text-align: left; }
    th { color: var(--muted); font-weight: 650; }
    .confidence { text-align: right; font-variant-numeric: tabular-nums; }
    .label-options { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-bottom: 10px; }
    .label-chip { border-left: 4px solid var(--label-color, var(--line)); padding-left: 6px; font-weight: 650; }
    .label-options button { border: 1px solid var(--line); border-left: 5px solid var(--label-color, var(--line)); border-radius: 6px; background: #fff; padding: 10px; cursor: pointer; font-weight: 650; }
    .label-options button.selected { border-color: var(--label-color, var(--accent)); background: #f7fbff; color: var(--ink); box-shadow: inset 0 0 0 1px var(--label-color, var(--accent)); }
    textarea { width: 100%; min-height: 76px; resize: vertical; border: 1px solid var(--line); border-radius: 6px; padding: 8px; font: inherit; margin-bottom: 10px; }
    .actions { display: flex; gap: 8px; align-items: center; }
    .actions button { height: 36px; border-radius: 6px; border: 1px solid var(--line); background: #fff; padding: 0 12px; cursor: pointer; }
    .actions button.primary { background: var(--accent); color: #fff; border-color: var(--accent); font-weight: 650; }
    .message { color: var(--muted); font-size: 13px; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; height: auto; }
      aside { max-height: 36vh; border-right: 0; border-bottom: 1px solid var(--line); }
      .layout { grid-template-columns: 1fr; }
      video { max-height: 52vh; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Private Video Manual Labeling</h1>
    <div class="status" id="status">Loading</div>
  </header>
  <main>
    <aside>
      <div class="toolbar"><input id="search" type="search" placeholder="Search clip id"></div>
      <div id="clipList" class="clip-list"></div>
    </aside>
    <section class="workspace">
      <div class="layout">
        <div>
          <h2 id="clipTitle">Select a clip</h2>
          <video id="video" controls loop preload="metadata"></video>
        </div>
        <div class="panel">
          <h2>Suggestions</h2>
          <div class="metrics">
            <div class="metric"><span>Majority</span><strong id="majority">-</strong></div>
            <div class="metric"><span>Agreement</span><strong id="agreement">-</strong></div>
            <div class="metric"><span>Mean conf.</span><strong id="meanConf">-</strong></div>
          </div>
          <table>
            <thead><tr><th>Run</th><th>Label</th><th class="confidence">Conf.</th></tr></thead>
            <tbody id="suggestions"></tbody>
          </table>
          <h2>Manual Label</h2>
          <div id="labelOptions" class="label-options"></div>
          <textarea id="notes" placeholder="Notes"></textarea>
          <div class="actions">
            <button class="primary" id="save">Save</button>
            <button id="prev">Prev</button>
            <button id="next">Next</button>
            <span class="message" id="message"></span>
          </div>
        </div>
      </div>
    </section>
  </main>
  <script>
    let data = {labels: [], clips: []};
    let filtered = [];
    let activeIndex = 0;
    let selectedLabel = null;

    const el = id => document.getElementById(id);
    const fmtPct = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "-";
    const fmtNum = value => Number.isFinite(value) ? value.toFixed(3) : "-";
    const labelColors = () => Object.fromEntries(data.labels.map(label => [label.name, label.color]));

    async function loadData() {
      const response = await fetch("/api/data");
      data = await response.json();
      filtered = data.clips;
      renderLabels();
      renderList();
      showClip(0);
      updateStatus();
    }

    function updateStatus() {
      el("status").textContent = `${data.labeled_count || 0}/${data.clips.length} labeled -> ${data.output_csv}`;
    }

    function renderLabels() {
      el("labelOptions").innerHTML = data.labels.map(label =>
        `<button data-label="${label.id}" style="--label-color:${label.color}" title="Set manual label">${label.name}</button>`
      ).join("");
      el("labelOptions").querySelectorAll("button").forEach(button => {
        button.addEventListener("click", () => {
          selectedLabel = Number(button.dataset.label);
          paintSelectedLabel();
        });
      });
    }

    function renderList() {
      const q = el("search").value.trim().toLowerCase();
      filtered = data.clips.filter(clip => clip.clip_id.toLowerCase().includes(q));
      el("clipList").innerHTML = filtered.map((clip, index) => `
        <button class="clip-item ${index === activeIndex ? "active" : ""}" data-index="${index}">
          <span class="clip-title"><span>${clip.clip_id}</span><span>${clip.manual_label ? "<span class='done'>Saved</span>" : ""}</span></span>
          <span class="clip-meta">${clip.majority_label} | agreement ${fmtPct(clip.agreement_rate)} | entropy ${fmtNum(clip.prediction_entropy)}</span>
        </button>
      `).join("");
      el("clipList").querySelectorAll("button").forEach(button => {
        button.addEventListener("click", () => showClip(Number(button.dataset.index)));
      });
    }

    function showClip(index) {
      if (!filtered.length) return;
      activeIndex = Math.max(0, Math.min(index, filtered.length - 1));
      const clip = filtered[activeIndex];
      selectedLabel = clip.manual_label_id;
      el("clipTitle").textContent = clip.clip_id;
      el("video").src = clip.video_url;
      el("majority").textContent = `${clip.majority_label} (${clip.majority_vote_count}/${clip.num_models})`;
      el("majority").style.color = labelColors()[clip.majority_label] || "";
      el("agreement").textContent = fmtPct(clip.agreement_rate);
      el("meanConf").textContent = fmtPct(clip.mean_confidence);
      el("notes").value = clip.notes || "";
      const colors = labelColors();
      el("suggestions").innerHTML = clip.suggestions.map(suggestion => `
        <tr>
          <td>${suggestion.run}</td>
          <td><span class="label-chip" style="--label-color:${colors[suggestion.label] || "var(--muted)"}">${suggestion.label}</span></td>
          <td class="confidence">${fmtPct(suggestion.confidence)}</td>
        </tr>
      `).join("");
      paintSelectedLabel();
      renderList();
      el("message").textContent = "";
    }

    function paintSelectedLabel() {
      el("labelOptions").querySelectorAll("button").forEach(button => {
        button.classList.toggle("selected", Number(button.dataset.label) === selectedLabel);
      });
    }

    async function saveLabel() {
      const clip = filtered[activeIndex];
      if (!clip || selectedLabel === null || selectedLabel === undefined) {
        el("message").textContent = "Choose a label first.";
        return;
      }
      const payload = {
        clip_id: clip.clip_id,
        manual_label_id: selectedLabel,
        notes: el("notes").value,
        source_majority_label: clip.majority_label,
        source_agreement_rate: clip.agreement_rate,
        source_mean_confidence: clip.mean_confidence
      };
      const response = await fetch("/api/label", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        el("message").textContent = await response.text();
        return;
      }
      const saved = await response.json();
      clip.manual_label_id = Number(saved.manual_label_id);
      clip.manual_label = saved.manual_label;
      clip.notes = saved.notes;
      data.labeled_count = data.clips.filter(item => item.manual_label).length;
      updateStatus();
      renderList();
      el("message").textContent = "Saved.";
    }

    el("search").addEventListener("input", () => { activeIndex = 0; renderList(); showClip(0); });
    el("save").addEventListener("click", saveLabel);
    el("prev").addEventListener("click", () => showClip(activeIndex - 1));
    el("next").addEventListener("click", () => showClip(activeIndex + 1));
    document.addEventListener("keydown", event => {
      if (event.target.tagName === "TEXTAREA" || event.target.tagName === "INPUT") return;
      if (event.key >= "1" && event.key <= "4") { selectedLabel = Number(event.key) - 1; paintSelectedLabel(); }
      if (event.key === "ArrowLeft") showClip(activeIndex - 1);
      if (event.key === "ArrowRight") showClip(activeIndex + 1);
      if (event.key.toLowerCase() === "s") saveLabel();
    });
    loadData().catch(error => { el("status").textContent = String(error); });
  </script>
</body>
</html>
"""


class ManualLabelHandler(BaseHTTPRequestHandler):
    server: "ManualLabelServer"

    def log_message(self, format: str, *args: Any) -> None:
        print(f'{self.address_string()} - {format % args}')

    def send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_text(self, text: str, status: int = 200, content_type: str = "text/plain") -> None:
        payload = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/":
                return self.send_text(HTML, content_type="text/html")
            if path == "/api/data":
                return self.send_json(load_ui_data(self.server.config))
            if path.startswith("/media/"):
                return self.serve_media(path[len("/media/") :])
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/label":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body) if body else {}
            saved = upsert_manual_label(self.server.config, payload)
            self.send_json(saved)
        except ValueError as exc:
            self.send_text(str(exc), status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def serve_media(self, media_path: str) -> None:
        repo_root = self.server.config.repo_root
        requested = (repo_root / unquote(media_path)).resolve()
        try:
            requested.relative_to(repo_root)
        except ValueError as exc:
            raise PermissionError(f"Refusing to serve outside repo: {requested}") from exc
        if not requested.exists() or not requested.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Media not found")
            return

        file_size = requested.stat().st_size
        start, end = 0, file_size - 1
        range_header = self.headers.get("Range")
        status = HTTPStatus.OK
        if range_header:
            units, _, range_spec = range_header.partition("=")
            if units.strip() == "bytes":
                first, _, last = range_spec.partition("-")
                start = int(first) if first else 0
                end = int(last) if last else end
                end = min(end, file_size - 1)
                status = HTTPStatus.PARTIAL_CONTENT

        content_type = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        content_length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        with requested.open("rb") as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class ManualLabelServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], config: UiConfig):
        super().__init__(server_address, ManualLabelHandler)
        self.config = config


def build_config(args: argparse.Namespace) -> UiConfig:
    repo_root = resolve_repo_root()
    return UiConfig(
        repo_root=repo_root,
        accepted_csv=resolve_path(repo_root, args.accepted_csv),
        agreement_csv=resolve_path(repo_root, args.agreement_csv),
        output_csv=resolve_path(repo_root, args.output_csv),
        host=args.host,
        port=args.port,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = build_config(args)
    load_ui_data(config)
    server = ManualLabelServer((config.host, config.port), config)
    url = f"http://{config.host}:{config.port}/"
    print(f"Manual labeling UI: {url}")
    print(f"Writing labels to: {config.output_csv}")
    print("Press Ctrl+C in this terminal to exit.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping manual labeling UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
