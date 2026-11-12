"""Scores for next-day fire forecasts, and the v2 model vs the baselines on one split.

Every score is pooled over pixels of all tiles (sum of hits / sum of ...), like the training
notebook. "New fire" = label pixels that were not burning on day t (label & ~fire_t): where the
fire spread to. Persistence scores 0 on it by construction.

    python -m training.evaluate_v2 --run data/retrain/runs/<name> --split val

The test split is refused unless --allow-test is given: it is scored once, at the end (plan R6).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

THRESHOLD_BINS = 1000


def _bool(array):
    return np.asarray(array) > 0.5


def counts(predicted, label, fire_t):
    """Pixel counts for one set of forecasts: all arrays (tiles, H, W); predicted/label/fire_t 0/1."""
    p, y, today = _bool(predicted), _bool(label), _bool(fire_t)
    new = y & ~today
    return {
        "hits": int((p & y).sum()),
        "forecast_px": int(p.sum()),
        "burned_px": int(y.sum()),
        "union": int((p | y).sum()),
        "new_hits": int((p & new).sum()),
        "new_burned_px": int(new.sum()),
        "forecast_outside_today_px": int((p & ~today).sum()),
    }


def scores_from_counts(c):
    def ratio(a, b):
        return a / b if b else float("nan")

    precision, recall = ratio(c["hits"], c["forecast_px"]), ratio(c["hits"], c["burned_px"])
    return {
        "precision": precision,
        "recall": recall,
        "iou": ratio(c["hits"], c["union"]),
        "f1": ratio(2 * c["hits"], c["forecast_px"] + c["burned_px"]),
        "new_fire_recall": ratio(c["new_hits"], c["new_burned_px"]),
        "new_fire_precision": ratio(c["new_hits"], c["forecast_outside_today_px"]),
        "forecast_px": c["forecast_px"],
        "burned_px": c["burned_px"],
    }


def pooled_scores(predicted, label, fire_t):
    return scores_from_counts(counts(predicted, label, fire_t))


def average_precision(probabilities, label):
    """Area under the precision-recall curve, the training notebook's definition:
    sort pixels by probability (descending); AP = sum(precision@k * hit_k) / number of positives."""
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    y = _bool(label).ravel()
    if not y.any():
        return float("nan")
    hits = y[np.argsort(-p, kind="stable")].astype(np.float64)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def threshold_curve(probabilities, label, bins=THRESHOLD_BINS):
    """Pooled IoU for thresholds 0, 1/bins, ..., 1 (forecast = probability >= threshold)."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64).ravel(), 0, 1)
    y = _bool(label).ravel()
    index = np.minimum((p * bins).astype(np.int64), bins - 1)  # bin b holds [b/bins, (b+1)/bins)
    pos = np.bincount(index[y], minlength=bins)
    neg = np.bincount(index[~y], minlength=bins)
    # Forecast at threshold b/bins = every pixel in bins >= b
    tp = np.cumsum(pos[::-1])[::-1]
    fp = np.cumsum(neg[::-1])[::-1]
    fn = y.sum() - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / (tp + fp + fn)
    thresholds = np.arange(bins) / bins
    return thresholds, np.nan_to_num(iou, nan=0.0)


def choose_threshold(probabilities, label, bins=THRESHOLD_BINS):
    """The threshold (a multiple of 1/bins) that maximizes pooled IoU. Use on val only."""
    thresholds, iou = threshold_curve(probabilities, label, bins)
    best = int(np.argmax(iou))
    return float(thresholds[best]), float(iou[best])


def per_tile_hits_union(predicted, label):
    p, y = _bool(predicted), _bool(label)
    return (p & y).sum(axis=(1, 2)).astype(np.int64), (p | y).sum(axis=(1, 2)).astype(np.int64)


def per_tile_iou(predicted, label):
    hits, union = per_tile_hits_union(predicted, label)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, hits / np.maximum(union, 1), np.nan)


def bootstrap_iou_difference(predicted_a, predicted_b, label, reps=1000, seed=0):
    """Pooled IoU(a) - IoU(b), and a 95% interval from resampling tiles with replacement."""
    hits_a, union_a = per_tile_hits_union(predicted_a, label)
    hits_b, union_b = per_tile_hits_union(predicted_b, label)
    n = len(hits_a)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(reps, n))
    diff = hits_a[draws].sum(1) / np.maximum(union_a[draws].sum(1), 1) - hits_b[draws].sum(1) / np.maximum(
        union_b[draws].sum(1), 1
    )
    point = hits_a.sum() / max(union_a.sum(), 1) - hits_b.sum() / max(union_b.sum(), 1)
    low, high = np.percentile(diff, [2.5, 97.5])
    return {"difference": float(point), "ci_low": float(low), "ci_high": float(high)}


# ----------------------------------------------------------------------------- CLI


GROUP_COLUMNS = ("stratum",)  # reported automatically when tiles.csv has them; --by adds more


def split_groups(data_dir, split, n, columns=GROUP_COLUMNS):
    """{"all": every tile, "<column>=<value>": tile indices, ...} from data_dir/tiles.csv.

    The split's rows of tiles.csv, ordered by `index`, must line up with the split's arrays.
    Columns missing from tiles.csv (or a missing tiles.csv) are skipped.
    """
    import pandas as pd

    groups = {"all": np.arange(n)}
    path = Path(data_dir) / "tiles.csv"
    if not path.exists():
        return groups
    tiles = pd.read_csv(path)
    tiles = tiles[tiles["split"] == split].sort_values("index")
    if len(tiles) != n or not np.array_equal(tiles["index"].to_numpy(), np.arange(n)):
        raise ValueError(f"tiles.csv rows for {split} don't line up with its {n} tiles")
    for column in columns:
        if column in tiles.columns:
            for value in sorted(tiles[column].dropna().unique(), key=str):
                groups[f"{column}={value}"] = np.flatnonzero((tiles[column] == value).to_numpy())
    return groups


def evaluate_split(run_folder, split, data_dir=None, allow_test=False, batch_size=64, reps=1000, by=()):
    """Model (at its saved threshold) and every baseline on one split, overall and per group
    (stratum when tiles.csv has it, plus any `by` columns such as region). Returns a list of rows."""
    from backend import spread_model_v2 as v2
    from training import baselines

    if split == "test" and not allow_test:
        raise SystemExit("The test split is scored once, at the end (plan R6). Pass --allow-test to do that.")
    data_dir = Path(data_dir) if data_dir else v2.RETRAIN_DIR
    inputs = np.load(data_dir / f"{split}_inputs.npy", mmap_mode="r")
    label = np.load(data_dir / f"{split}_labels.npy", mmap_mode="r")
    model, stats, manifest = v2.load_run(run_folder)
    channels = tuple(manifest["channels"])
    threshold = float(manifest["threshold"])

    probabilities = np.empty(label.shape, dtype=np.float32)
    for start in range(0, len(label), 512):  # bounded memory: 512 tiles of raw inputs at a time
        chunk = np.asarray(inputs[start : start + 512], dtype=np.float32)
        probabilities[start : start + 512] = v2.predict(v2.select_channels(chunk, channels), model, stats, channels, batch_size)
    fire_t = np.asarray(inputs[:, v2.CHANNELS.index("fire_t")], dtype=np.float32)
    label = np.asarray(label)
    forecasts = {f"model ({Path(run_folder).name}, thr {threshold:.3f})": probabilities >= threshold}
    for name, forecast in baselines.all_baselines(inputs).items():
        forecasts[name] = forecast

    rows = []
    groups = split_groups(data_dir, split, len(label), GROUP_COLUMNS + tuple(by))
    for group, idx in groups.items():
        persistence = forecasts["persistence"][idx]
        for name, forecast in forecasts.items():
            row = {"group": group, "tiles": len(idx), "forecast": name,
                   **pooled_scores(forecast[idx], label[idx], fire_t[idx])}
            if name.startswith("model"):
                row["pr_auc"] = average_precision(probabilities[idx], label[idx])
            if name != "persistence" and len(idx):
                diff = bootstrap_iou_difference(forecast[idx], persistence, label[idx], reps)
                row.update({f"vs_persistence_{k}": v for k, v in diff.items()})
            rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--out", default=None, help="CSV to write (default: <run>/eval_<split>.csv)")
    parser.add_argument("--by", nargs="*", default=(), help="extra tiles.csv columns to report by, e.g. region")
    args = parser.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import pandas as pd
    import torch

    torch.set_num_threads(4)
    rows = evaluate_split(args.run, args.split, args.data_dir, args.allow_test, by=tuple(args.by))
    table = pd.DataFrame(rows)
    out = Path(args.out) if args.out else Path(args.run) / f"eval_{args.split}{'_' + Path(args.data_dir).name if args.data_dir else ''}.csv"
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.round(4).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
