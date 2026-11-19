"""Turn the v2 spread model's scores into real chances (probability calibration).

The model's outputs are scores, not probabilities: trained with BCE + Dice, they cluster near 0 and
1. On the 2023 val fires, a score of 0.99 or more near a fire meant fire was really there the next
day only about 71% of the time. The dashboard shows a "chance of fire", so the score has to be
mapped to how often fire actually followed.

Two maps, both fitted on the val split (2023) with isotonic regression (monotone, so the ranking
and the forecast's likely area don't change), weighted by 1 / sample_prob so they match the
natural mix of fires rather than the 50/50 growing/small sample:
- "pixel": score of one 1 km pixel -> chance that pixel burns the next day
- "near": highest score within NEAR_PX pixels of a burning pixel -> chance of any fire within that
  window the next day (the dashboard's "chance of fire within 2 km")

The held-out test split (2024-2025) is used once, only to report how well the maps hold up.
Writes models/spread_v2/calibration.json.

    python -m training.calibrate_v2
"""

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from backend.spread_model_v2 import CHANNELS, load_run, predict  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "retrain"
MODEL_DIR = REPO / "models" / "spread_v2"
NEAR_PX = 2  # must match backend/spread_forecast.NEAR_PX
PIXEL_SAMPLE_EVERY = 7  # every 7th pixel is plenty for the pixel map and keeps memory small
SEED = 2026
# No chance on the map rests on fewer than this many examples: scores are grouped into bins this
# big before the isotonic fit, so a handful of fires at the very top can't produce a "100%"
MIN_PER_BIN = {"near": 150, "pixel": 300}


def isotonic(x, y, weight):
    """Weighted isotonic regression (pool adjacent violators). Returns block (x, fitted y) knots."""
    order = np.argsort(x, kind="stable")
    x, y, weight = x[order], y[order].astype(float), weight[order].astype(float)
    # Merge equal x first, so each knot is one distinct score
    values, starts = np.unique(x, return_index=True)
    sums = np.add.reduceat(y * weight, starts)
    weights = np.add.reduceat(weight, starts)
    block_y, block_w, block_lo, block_hi = [], [], [], []
    for value, total, w in zip(values, sums, weights):
        block_y.append(total / w)
        block_w.append(w)
        block_lo.append(value)
        block_hi.append(value)
        while len(block_y) > 1 and block_y[-2] > block_y[-1]:  # violator: pool with the previous block
            w_new = block_w[-2] + block_w[-1]
            y_new = (block_y[-2] * block_w[-2] + block_y[-1] * block_w[-1]) / w_new
            hi = block_hi[-1]
            for stack in (block_y, block_w, block_lo, block_hi):
                stack.pop()
            block_y[-1], block_w[-1], block_hi[-1] = y_new, w_new, hi
    # Knots at each block's ends, so np.interp is flat inside a block and linear between blocks
    knot_x, knot_y = [], []
    for lo, hi, value in zip(block_lo, block_hi, block_y):
        knot_x += [lo, hi] if hi > lo else [lo]
        knot_y += [value, value] if hi > lo else [value]
    return np.array(knot_x), np.array(knot_y)


def binned_isotonic(x, y, weight, min_per_bin):
    """Isotonic regression over bins of at least min_per_bin consecutive examples (sorted by score).

    Each bin becomes one point (its weighted mean score, weighted fire rate). Knots are those points,
    so np.interp is linear between bin centres and flat beyond the first and last bins.
    """
    order = np.argsort(x, kind="stable")
    x, y, weight = x[order], y[order].astype(float), weight[order].astype(float)
    starts = np.arange(0, len(x), min_per_bin)
    if len(x) - starts[-1] < min_per_bin and len(starts) > 1:
        starts = starts[:-1]  # fold a short last bin into the one before it
    weights = np.add.reduceat(weight, starts)
    centres = np.add.reduceat(x * weight, starts) / weights
    rates = np.add.reduceat(y * weight, starts) / weights
    knot_x, knot_y = isotonic(centres, rates, weights)
    return knot_x, knot_y


def apply(knots, scores):
    return np.interp(scores, knots["x"], knots["y"])


def collect(split, model, stats):
    """(near score, near label, near weight), (pixel score, pixel label, pixel weight) for one split."""
    tiles = pd.read_csv(DATA / "tiles.csv")
    sub = tiles[tiles.split == split].sort_values("index")
    inputs = np.load(DATA / f"{split}_inputs.npy", mmap_mode="r")
    labels = np.load(DATA / f"{split}_labels.npy", mmap_mode="r")
    fire = CHANNELS.index("fire_t")
    near, pixel = ([], [], []), ([], [], [])
    for start in range(0, len(sub), 256):
        chunk = sub.iloc[start : start + 256]
        idx = chunk["index"].to_numpy()
        scores = predict(np.asarray(inputs[idx], dtype=np.float32), model, stats)
        today = np.asarray(inputs[idx, fire]) == 1
        burned = np.asarray(labels[idx]) == 1
        weights = 1.0 / chunk["sample_prob"].to_numpy()
        for k in range(len(idx)):
            rows, cols = np.nonzero(today[k])
            if len(rows):
                # The burning pixel nearest the tile centre, like the fire a person clicks live
                j = np.argmin((rows - 31) ** 2 + (cols - 31) ** 2)
                window = (slice(max(0, rows[j] - NEAR_PX), rows[j] + NEAR_PX + 1), slice(max(0, cols[j] - NEAR_PX), cols[j] + NEAR_PX + 1))
                near[0].append(scores[k][window].max())
                near[1].append(burned[k][window].any())
                near[2].append(weights[k])
            flat = slice(None, None, PIXEL_SAMPLE_EVERY)
            pixel[0].append(scores[k].ravel()[flat])
            pixel[1].append(burned[k].ravel()[flat])
            pixel[2].append(np.full(len(scores[k].ravel()[flat]), weights[k]))
    near = tuple(np.asarray(part, dtype=float) for part in near)
    pixel = tuple(np.concatenate(part).astype(float) for part in pixel)
    return near, pixel


def reliability(scores, labels, weights, edges=(0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)):
    """Weighted mean predicted vs observed per bin, and the weighted Brier score."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (scores >= lo) & (scores < hi)
        if m.any():
            w = weights[m]
            rows.append({"bin": f"{lo:.1f}-{min(hi, 1):.1f}", "n": int(m.sum()),
                         "predicted": round(float(np.average(scores[m], weights=w)), 3),
                         "observed": round(float(np.average(labels[m], weights=w)), 3)})
    brier = float(np.average((scores - labels) ** 2, weights=weights))
    return {"bins": rows, "brier": round(brier, 4)}


def main():
    torch.set_num_threads(4)
    model, stats, manifest = load_run(MODEL_DIR)
    (nx, ny, nw), (px, py, pw) = collect("val", model, stats)
    near = dict(zip(("x", "y"), (v.tolist() for v in binned_isotonic(nx, ny, nw, MIN_PER_BIN["near"]))))
    pixel = dict(zip(("x", "y"), (v.tolist() for v in binned_isotonic(px, py, pw, MIN_PER_BIN["pixel"]))))

    (tx, ty, tw), (qx, qy, qw) = collect("test", model, stats)
    check = {
        "near_raw": reliability(tx, ty, tw),
        "near_calibrated": reliability(apply(near, tx), ty, tw),
        "pixel_raw": reliability(qx, qy, qw),
        "pixel_calibrated": reliability(apply(pixel, qx), qy, qw),
    }
    result = {
        "about": "Maps from the model's score to the real chance of fire the next day (isotonic, fitted on val 2023, "
                 "weighted to the natural mix of fires). Checked once on test 2024-2025 (test_check).",
        "near_px": NEAR_PX,
        "min_per_bin": MIN_PER_BIN,
        "threshold_score": manifest["threshold"],
        "near": near,
        "pixel": pixel,
        "fitted_on": {"split": "val", "near_fires": int(len(nx)), "pixels": int(len(px))},
        "test_check": check,
    }
    (MODEL_DIR / "calibration.json").write_text(json.dumps(result, indent=1) + "\n")
    for name, table in check.items():
        print(name, "Brier", table["brier"])
        for row in table["bins"]:
            print(f"   {row['bin']:8s} n={row['n']:8d} predicted {row['predicted']:.3f} observed {row['observed']:.3f}")
    print("wrote", MODEL_DIR / "calibration.json")


if __name__ == "__main__":
    main()
