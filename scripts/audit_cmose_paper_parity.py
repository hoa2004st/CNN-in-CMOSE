"""Audit current CMOSE baseline implementation against paper-stated specs.

This script checks only explicitly stated items from the CVPRW 2024 paper.
It does not claim equivalence to unpublished author training code.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check(condition: bool, message: str) -> dict[str, str]:
    return {"status": "PASS" if condition else "FAIL", "check": message}


def main() -> None:
    baseline_cfg = yaml.safe_load(read_text(REPO / "configs" / "baseline.yaml"))
    main_py = read_text(REPO / "main.py")
    mocorank_py = read_text(REPO / "src" / "training" / "mocorank.py")
    model_py = read_text(REPO / "src" / "models" / "cmose_baseline.py")
    openface_py = read_text(REPO / "src" / "features" / "extract_openface.py")

    checks: list[dict[str, str]] = []

    checks.append(check(baseline_cfg.get("optimizer") == "AdamW", "baseline optimizer is AdamW"))
    checks.append(check(float(baseline_cfg.get("weight_decay", -1)) == 1e-3, "weight_decay is 1e-3"))
    checks.append(check(int(baseline_cfg.get("batch_size", -1)) == 256, "batch size is 256"))
    checks.append(check(int(baseline_cfg.get("score_pool_size", -1)) == 2048, "score pool size is 2048"))
    checks.append(check(int(baseline_cfg.get("epochs", -1)) == 1200, "epochs is 1200"))
    checks.append(check(float(baseline_cfg.get("lr_init", -1)) == 5e-4, "initial lr is 5e-4"))
    checks.append(check(float(baseline_cfg.get("lr_final", -1)) == 5e-7, "final lr is 5e-7"))
    checks.append(
        check(str(baseline_cfg.get("scheduler", "")).lower() == "cosineannealing", "scheduler is CosineAnnealing")
    )
    checks.append(check(float(baseline_cfg.get("momentum_update", -1)) == 0.999, "momentum update is 0.999"))
    checks.append(check(int(baseline_cfg.get("C", -1)) == 128, "hidden dim C is 128"))
    checks.append(check(int(baseline_cfg.get("T", -1)) == 10, "temporal chunks T is 10"))

    checks.append(
        check("--strict_paper_baseline" in main_py and "--baseline_config" in main_py, "strict paper mode exists")
    )
    checks.append(check("CosineAnnealingLR" in mocorank_py, "CosineAnnealingLR is implemented"))
    checks.append(check("strict_pool_init" in mocorank_py, "strict balanced score-pool init path exists"))
    checks.append(check("FIFO" in mocorank_py and "ScorePool" in mocorank_py, "score pool FIFO structure exists"))
    checks.append(check("Dropout(p=float(dropout))" in read_text(REPO / "src" / "models" / "backbone.py"), "dropout in model blocks exists"))
    checks.append(check("dropout=0.5" in model_py, "attention MLP uses dropout=0.5"))
    checks.append(check("dropout=0.2" in model_py, "TCN uses dropout=0.2"))
    checks.append(check("chunk_count: int = 10" in openface_py, "OpenFace chunk count default is 10"))
    checks.append(check("standard_frames = 250" in openface_py, "OpenFace chunking uses 250 frames"))
    checks.append(check("chunk.min(axis=0)" in openface_py and "chunk.max(axis=0)" in openface_py and "chunk.var(axis=0)" in openface_py, "chunk min/max/var features are computed"))

    report = {
        "summary": {
            "pass": sum(1 for item in checks if item["status"] == "PASS"),
            "fail": sum(1 for item in checks if item["status"] == "FAIL"),
        },
        "checks": checks,
        "limitations": [
            "No official public author training code repository was found in paper/project page links.",
            "Paper does not fully specify all implementation minutiae (e.g., exact split file mapping implementation details).",
            "This audit checks paper-stated items and local implementation only.",
        ],
    }

    out_dir = REPO / "outputs" / "comparisons" / "paper_parity_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "audit_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote audit report: {out_path}")
    print(f"PASS={report['summary']['pass']} FAIL={report['summary']['fail']}")


if __name__ == "__main__":
    main()

