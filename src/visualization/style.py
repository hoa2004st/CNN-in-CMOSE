"""Canonical colors and labels for training-log and feature-space visualizations."""

from __future__ import annotations

from collections.abc import Iterable

from matplotlib import colors as mcolors

CLASS_LABELS = ["Highly Disengage", "Disengage", "Engage", "Highly Engage"]
CLASS_LABEL_SHORT = {
    "Highly Disengage": "HD",
    "Disengage": "DE",
    "Engage": "EG",
    "Highly Engage": "HE",
}
CLASS_COLORS = {
    "Highly Disengage": "#0072B2",
    "Disengage": "#D55E00",
    "Engage": "#009E73",
    "Highly Engage": "#CC79A7",
}

# Models shown in model-assessment charts/tables. The fusion model
# (openface_tcn_i3d_fusion) is intentionally excluded from assessment; the
# assessment viz scripts treat this list as the model allowlist. Display names
# and colors for the fusion key are kept below for backward compatibility with
# historical per-run training artifacts.
MODEL_ORDER = [
    "openface_mlp",
    "temporal_cnn",
    "lstm",
    "transformer",
    "i3d_mlp",
]
MODEL_DISPLAY_NAMES = {
    "openface_mlp": "openface_mlp",
    "temporal_cnn": "openface_tcn",
    "lstm": "openface_lstm",
    "transformer": "openface_transformer",
    "i3d_mlp": "i3d_mlp",
    "openface_tcn_i3d_fusion": "fusion",
}
MODEL_DISPLAY_TO_KEY = {display: key for key, display in MODEL_DISPLAY_NAMES.items()}
MODEL_COLORS = {
    "openface_mlp": "#0072B2",
    "temporal_cnn": "#56B4E9",
    "lstm": "#009E73",
    "transformer": "#8BC34A",
    "i3d_mlp": "#D55E00",
    "openface_tcn_i3d_fusion": "#E69F00",
}

COMPARISON_LOSS_NAMES = ["cross_entropy", "weighted_cross_entropy", "ordinal"]
COMPARISON_LOSS_SLUGS = ["ce", "weighted_ce", "ordinal"]
LOSS_NAME_TO_SLUG = {
    "cross_entropy": "ce",
    "weighted_cross_entropy": "weighted_ce",
    "ordinal": "ordinal",
}
LOSS_SLUG_TO_NAME = {
    "ce": "cross_entropy",
    "weighted_ce": "weighted_cross_entropy",
    "ordinal": "ordinal",
}
LOSS_DISPLAY_NAMES = {
    "cross_entropy": "CE",
    "ce": "CE",
    "weighted_cross_entropy": "Weighted CE",
    "weighted_ce": "Weighted CE",
    "ordinal": "Ordinal",
}
LOSS_TINT = {
    "ce": 0.00,
    "weighted_ce": 0.25,
    "ordinal": 0.45,
}

# One colour per model family in base-vs-hybrid comparison charts (no grays):
# every "base" element is blue, every "hybrid" element is vermillion.
FAMILY_COLORS = {
    "base": "#0072B2",
    "hybrid": "#D55E00",
}

DOMAIN_COLORS = {
    "CMOSE": "#0072B2",
    "cmose": "#0072B2",
    "private": "#D55E00",
    "Private": "#D55E00",
}

FALLBACK_COLOR = "#4D4D4D"

CURVE_COLORS = {
    "train_loss": "#4D4D4D",
    "eval_loss": "#D55E00",
    "eval_accuracy": "#0072B2",
    "eval_macro_accuracy": "#56B4E9",
    "eval_f1_macro": "#009E73",
    "eval_f1_weighted": "#59A14F",
    "eval_mae": "#CC79A7",
    "eval_mse": "#A23B72",
    "best_epoch": "#D55E00",
}

FALLBACK_COLORS = [
    "#4D4D4D",
    "#8C564B",
    "#6F4E7C",
    "#7F7F7F",
    "#17BECF",
]


def stable_fallback_color(value: object) -> str:
    text = str(value)
    if not text:
        return FALLBACK_COLOR
    index = sum((pos + 1) * ord(char) for pos, char in enumerate(text)) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[index]


def class_color(label: object) -> str:
    return CLASS_COLORS.get(str(label), stable_fallback_color(label))


def class_palette(labels: Iterable[object] | None = None) -> dict[str, str]:
    ordered = CLASS_LABELS if labels is None else [str(label) for label in labels]
    return {label: class_color(label) for label in ordered}


def class_color_list(labels: Iterable[object]) -> list[str]:
    return [class_color(label) for label in labels]


def display_model_name(model_name: object) -> str:
    return MODEL_DISPLAY_NAMES.get(str(model_name), str(model_name))


def model_key(model_or_display: object) -> str:
    value = str(model_or_display)
    return MODEL_DISPLAY_TO_KEY.get(value, value)


def model_color(model_or_display: object) -> str:
    key = model_key(model_or_display)
    return MODEL_COLORS.get(key, stable_fallback_color(key))


def model_palette(labels: Iterable[object]) -> dict[str, str]:
    return {str(label): model_color(label) for label in labels}


def normalize_loss_slug(loss_name_or_slug: object | None) -> str | None:
    if loss_name_or_slug is None:
        return None
    value = str(loss_name_or_slug)
    return LOSS_NAME_TO_SLUG.get(value, value)


def loss_display_name(loss_name_or_slug: object) -> str:
    return LOSS_DISPLAY_NAMES.get(str(loss_name_or_slug), str(loss_name_or_slug))


def blend_with_white(hex_color: str, amount: float) -> str:
    amount = max(0.0, min(1.0, float(amount)))
    rgb = mcolors.to_rgb(hex_color)
    blended = tuple(channel + (1.0 - channel) * amount for channel in rgb)
    return mcolors.to_hex(blended)


def run_color(model_or_display: object, loss_name_or_slug: object | None = None) -> str:
    base = model_color(model_or_display)
    slug = normalize_loss_slug(loss_name_or_slug)
    if slug is None:
        return base
    return blend_with_white(base, LOSS_TINT.get(slug, 0.15))


def run_palette(rows: Iterable[dict[str, object]], label_key: str) -> dict[str, str]:
    palette: dict[str, str] = {}
    for row in rows:
        label = str(row[label_key])
        model = row.get("base_model") or row.get("model") or label
        loss = row.get("loss") or row.get("loss_slug")
        palette[label] = run_color(model, loss)
    return palette


def domain_color(domain: object) -> str:
    return DOMAIN_COLORS.get(str(domain), stable_fallback_color(domain))


def domain_color_list(domains: Iterable[object]) -> list[str]:
    return [domain_color(domain) for domain in domains]
