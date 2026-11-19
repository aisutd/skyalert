"""Flood segmentation metrics from the architecture plan: flood IoU/Dice, precision/recall, valid
coverage, dry-scene false alarms, permanent-water errors and (projected grids only) flooded area.
"""
from collections import defaultdict

import numpy as np

from .contract import FLOOD, IGNORE, PERMANENT_WATER

N = 3
DRY_ALARM_KM2 = 0.05  # a dry scene raises a false alarm if the model maps at least this much flood (500 px at 10 m)


def confusion(pred, y):
    """3x3 counts over labelled pixels: rows truth, columns prediction."""
    v = y != IGNORE
    return np.bincount(y[v].astype(np.int64) * N + pred[v].astype(np.int64), minlength=N * N).reshape(N, N)


def flood_scores(cm):
    tp = int(cm[FLOOD, FLOOD]); fp = int(cm[:, FLOOD].sum()) - tp; fn = int(cm[FLOOD, :].sum()) - tp
    water_t, water_p, water_tp = cm[1:, :].sum(), cm[:, 1:].sum(), cm[1:, 1:].sum()
    pw = cm[PERMANENT_WATER].sum()
    return {
        "flood_iou": tp / max(tp + fp + fn, 1),
        "flood_dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "flood_precision": tp / max(tp + fp, 1) if tp + fp else None,
        "flood_recall": tp / max(tp + fn, 1) if tp + fn else None,
        "any_water_iou": float(water_tp / max(water_t + water_p - water_tp, 1)),
        # permanent-water errors, both directions
        "permwater_called_flood": float(cm[PERMANENT_WATER, FLOOD] / pw) if pw else None,
        "flood_called_permwater": float(cm[FLOOD, PERMANENT_WATER] / cm[FLOOD].sum()) if cm[FLOOD].sum() else None,
        "pixels_scored": int(cm.sum()),
        "flood_pixels_true": int(cm[FLOOD].sum()),
        "flood_pixels_predicted": int(cm[:, FLOOD].sum()),
    }


def pixel_km2(profile):
    """Area of one pixel in km^2, or None when the grid is not projected in metres."""
    crs = profile.get("crs")
    if crs is None or not crs.is_projected or (crs.linear_units or "").lower() not in ("metre", "meter", "m"):
        return None
    t = profile["transform"]
    return abs(t.a * t.e - t.b * t.d) / 1e6


def evaluate(preds, samples):
    """Pooled and per-event report for predictions aligned with samples (dicts with y, valid, profile, event_id)."""
    total_cm = np.zeros((N, N), dtype=np.int64)
    per_event = defaultdict(lambda: np.zeros((N, N), dtype=np.int64))
    tiles, dry = [], []
    px_total = px_valid = 0
    area_true = area_pred = 0.0
    area_ok = True
    for p, s in zip(preds, samples):
        cm = confusion(p, s["y"])
        total_cm += cm
        per_event[s["event_id"]] += cm
        px_total += s["y"].size
        px_valid += int(s["valid"].sum())
        km2 = pixel_km2(s["profile"])
        if km2 is None:
            area_ok = False
        else:
            area_true += cm[FLOOD].sum() * km2
            area_pred += cm[:, FLOOD].sum() * km2
        sc = flood_scores(cm)
        tiles.append({"sample_id": s["sample_id"], "event_id": s["event_id"], "valid_coverage": round(float(s["valid"].mean()), 4),
                      "flood_iou": round(sc["flood_iou"], 4), "flood_px_true": sc["flood_pixels_true"], "flood_px_pred": sc["flood_pixels_predicted"]})
        if sc["flood_pixels_true"] == 0 and cm.sum() > 0:
            false_km2 = sc["flood_pixels_predicted"] * km2 if km2 else None
            dry.append({"sample_id": s["sample_id"], "false_flood_px": sc["flood_pixels_predicted"], "false_flood_km2": false_km2,
                        "alarm": bool(false_km2 is not None and false_km2 >= DRY_ALARM_KM2), "valid_px": int(cm.sum())})
    event_rows = []
    for ev, cm in sorted(per_event.items()):
        sc = flood_scores(cm)
        event_rows.append({"event_id": ev, "tiles": sum(t["event_id"] == ev for t in tiles), "flood_iou": round(sc["flood_iou"], 4),
                           "flood_dice": round(sc["flood_dice"], 4), "flood_px_true": sc["flood_pixels_true"],
                           "flood_px_pred": sc["flood_pixels_predicted"]})
    wet_events = [r for r in event_rows if r["flood_px_true"] > 0]
    dry_px = sum(d["valid_px"] for d in dry)
    pooled = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in flood_scores(total_cm).items()}
    return {
        "pooled": pooled,
        "mean_event_flood_iou": round(float(np.mean([r["flood_iou"] for r in wet_events])), 4) if wet_events else None,
        "valid_coverage": round(px_valid / max(px_total, 1), 4),
        "dry_scenes": {
            "count": len(dry),
            "with_false_alarm": sum(d["alarm"] for d in dry),
            "false_alarm_rate": round(sum(d["alarm"] for d in dry) / len(dry), 4) if dry else None,
            "alarm_threshold_km2": DRY_ALARM_KM2,
            "false_flood_pixel_rate": round(sum(d["false_flood_px"] for d in dry) / max(dry_px, 1), 6),
        },
        "area_km2": {"true": round(area_true, 2), "predicted": round(area_pred, 2)} if area_ok else "not computed: a sample is not on a projected metre grid",
        "confusion_rows_truth_cols_pred": total_cm.tolist(),
        "classes": ["background", "permanent water", "flooded water"],
        "tiles": len(samples),
        "events": len(per_event),
        "events_with_flood": len(wet_events),
        "per_event": event_rows,
        "per_tile": tiles,
    }
