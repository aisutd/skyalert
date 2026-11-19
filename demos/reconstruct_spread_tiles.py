"""Rebuild real training tiles from scratch and compare every channel (plan step 2b).

Some Next Day Wildfire Spread tiles have known locations: their elevation channel matched a
real DEM (docs/spread_forecast_plan.md, "How the training tiles were made"). For each one this
script:
  1. refines the location (elevation correlation over whole-pixel shifts),
  2. finds the fire day t by matching the stored weather against every gridMET day 2012-2020,
     then confirms it with FIRMS MODIS Terra detections against the stored fire masks,
  3. decides which gridMET days the release used, and the PDSI pentad convention,
  4. rebuilds the whole tile with backend.spread_inputs.build_inputs and compares channels,
  5. measures what our approximations cost the model's forecast.

Everything goes through the backend's own code (gridmet, terrain, fire_points, spread_grid,
spread_inputs, spread_model). Writes data/demos/spread_reconstruction/report.md and CSVs.
Caches downloads in data/cache/reconstruction/. Run: python demos/reconstruct_spread_tiles.py
"""

import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from backend import fire_points, gridmet, terrain  # noqa: E402
from backend.firms_client import get_map_key  # noqa: E402
from backend.spread_grid import PIXEL_DEG, TileGrid  # noqa: E402
from backend.spread_inputs import WEATHER_CHANNELS, build_inputs  # noqa: E402
from backend.spread_model import INPUT_FEATURES, THRESHOLD, predict  # noqa: E402

OUT = REPO / "data" / "demos" / "spread_reconstruction"
TILE_DIR = OUT / "tiles"
CACHE = REPO / "data" / "cache" / "reconstruction"
NDWS = REPO / "data" / "ndws"
YEARS = range(2012, 2021)  # the NDWS paper's span
SEARCH_VARS = ("tmmx", "tmmn", "vs", "sph")  # day-to-day distinctive, no 0/360 wrap
WORKERS = 6  # concurrent tiles; each tile makes one request at a time

# Tiles located by matching their elevation channel against a real DEM (NCC 0.96-0.998, only
# under a vertical flip of the stored array). (shard, record, centre lat, centre lon of the
# stored pixel [31, 31], about +-0.5 px). One tile (train_00 #773) had no confident match and
# is left out.
LOCATED = [
    ("test_01", 677, 36.3472, -118.7163), ("eval_00", 939, 36.5718, -118.7612),
    ("train_00", 806, 36.3203, -118.6534), ("train_00", 839, 36.1586, -118.6893),
    ("train_00", 552, 36.8952, -119.1115), ("train_00", 240, 36.8144, -119.0127),
    ("train_00", 401, 36.9042, -118.9409), ("train_00", 508, 36.7964, -119.0127),
    ("eval_00", 453, 36.2394, -118.6175), ("test_01", 272, 36.8593, -118.9049),
    ("train_00", 868, 36.7515, -118.8420), ("test_01", 276, 36.8503, -118.8780),
    ("train_00", 416, 36.7694, -118.8151), ("train_00", 911, 36.2125, -118.5815),
    ("eval_00", 445, 36.7245, -119.0487), ("train_00", 156, 36.3293, -118.5366),
    ("train_00", 210, 37.6139, -119.6954), ("train_00", 672, 36.7066, -118.5636),
    ("train_00", 379, 36.8054, -118.6534), ("eval_00", 344, 37.2096, -119.3900),
    ("train_00", 923, 36.7964, -118.6265), ("train_00", 565, 36.8144, -118.6534),
    ("train_00", 476, 36.8054, -118.6354), ("train_00", 828, 34.0745, -116.7849),
    ("eval_00", 179, 45.9323, -121.4472), ("eval_00", 816, 38.3505, -119.8841),
    ("train_00", 531, 47.6480, -120.4860), ("train_00", 195, 40.8208, -122.4802),
    ("test_01", 269, 40.8927, -123.3875), ("train_00", 520, 45.4472, -115.2129),
    ("eval_00", 150, 33.7062, -111.3321), ("test_01", 73, 44.1446, -115.6171),
    ("test_01", 194, 40.0393, -121.2765), ("eval_00", 658, 34.8740, -113.2905),
    ("eval_00", 159, 36.5808, -105.9961), ("test_01", 635, 43.9560, -110.1553),
    ("test_01", 250, 43.1475, -117.0185), ("test_01", 653, 44.7824, -111.0177),
    ("test_01", 644, 46.7767, -111.9520), ("train_00", 622, 33.6882, -108.6102),
    ("test_01", 157, 43.5068, -111.7094), ("test_01", 102, 43.1205, -123.7289),
    ("train_00", 281, 45.1507, -120.9261),
]
SHARD_FILES = {
    "train_00": "next_day_wildfire_spread_train_00.tfrecord",
    "eval_00": "next_day_wildfire_spread_eval_00.tfrecord",
    "test_01": "next_day_wildfire_spread_test_01.tfrecord",
}
# Physically valid ranges for comparing stored channels (outside = export fill value)
VALID = {"th": (0, 360), "tmmn": (200, 330), "tmmx": (200, 330), "vs": (0, 60), "sph": (0, 0.05),
         "pr": (-5, 500), "erc": (-5, 200), "pdsi": (-15, 15), "elevation": (-500, 5000)}
WINDOWS = {  # name -> days relative to t
    "t-1,t (code lag=0)": (-1, 0),
    "t only (paper)": (0,),
    "t-2,t-1 (code lag=1)": (-2, -1),
    "t-1 only": (-1,),
}


def setup_caches():
    for module, name in ((gridmet, "gridmet"), (terrain, "terrain"), (fire_points, "firms")):
        folder = CACHE / name
        folder.mkdir(parents=True, exist_ok=True)
        module.CACHE_DIR = folder


def read_stored_tiles():
    """The located tiles' 12 inputs and label, as stored (raw units, south-up)."""
    cells = {c.get("id"): c.source for c in nbformat.read(REPO / "notebooks" / "train_spread.ipynb", as_version=4).cells}
    namespace = {}
    exec(cells["b2e0d003"].split("ON_KAGGLE = ")[0], namespace)
    exec(cells["b2e0d007"].split("data, rows = {}, []")[0], namespace)
    wanted = {}
    for shard, record, *_ in LOCATED:
        wanted.setdefault(shard, set()).add(record)
    stored = {}
    for shard, records in wanted.items():
        for position, grids in enumerate(namespace["read_records"](NDWS / SHARD_FILES[shard])):
            if position in records:
                stored[(shard, position)] = grids
    return stored


def masked_stats(ours, theirs, name):
    """Correlation and MAE over pixels where both are finite and the stored value is valid."""
    ours, theirs = np.asarray(ours, float), np.asarray(theirs, float)
    keep = np.isfinite(ours) & np.isfinite(theirs)
    if name in VALID:
        low, high = VALID[name]
        keep &= (theirs >= low) & (theirs <= high)
    if keep.sum() < 10:
        return {"corr": np.nan, "mae": np.nan, "n": int(keep.sum())}
    a, b = ours[keep], theirs[keep]
    corr = np.corrcoef(a, b)[0, 1] if a.std() > 1e-9 and b.std() > 1e-9 else np.nan
    return {"corr": float(corr), "mae": float(np.abs(a - b).mean()), "n": int(keep.sum()),
            "ours_min": float(a.min()), "ours_max": float(a.max()), "stored_min": float(b.min()), "stored_max": float(b.max())}


def mask_scores(predicted, actual, valid):
    """Pixel precision, recall and IoU of a 0/1 prediction against a 0/1 truth, on valid pixels."""
    p, a = (predicted == 1) & valid, (actual == 1) & valid
    hits, n_p, n_a, union = int((p & a).sum()), int(p.sum()), int(a.sum()), int((p | a).sum())
    return {"hits": hits, "predicted": n_p, "actual": n_a, "union": union,
            "precision": hits / n_p if n_p else np.nan, "recall": hits / n_a if n_a else np.nan,
            "iou": hits / union if union else np.nan}


# ----------------------------------------------------------------------------- 1. location


def refine_location(lat, lon, stored_elevation, reach=3):
    """The whole-pixel shift of the DEM-match centre that best reproduces the stored elevation."""
    results = []
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            grid = TileGrid.around(lat + dy * PIXEL_DEG, lon + dx * PIXEL_DEG)
            corr = masked_stats(terrain.elevation(grid), stored_elevation, "elevation")["corr"]
            results.append((corr, dy, dx, grid))
    results.sort(key=lambda r: -r[0])
    best, second = results[0], results[1]
    centre = best[3].center
    return {"lat": centre[0], "lon": centre[1], "elev_corr": best[0], "elev_corr_second": second[0],
            "shift_px": (best[1], best[2])}


# ----------------------------------------------------------------------------- 2. fire day


def _layer_basis(grid, lat_nodes, lon_nodes, lats, lons, origin=None):
    """Matrix M (4096 x nodes) so that bicubic(nodes) = M @ nodes for finite nodes.

    Built by running gridmet.bicubic on each unit node field, so it is exactly the backend's
    interpolation (Keys weights sum to 1, so the renormalisation is a no-op on finite fields).
    """
    origin = origin or (gridmet.NODE_ORIGIN_LAT, gridmet.NODE_ORIGIN_LON)
    probe, node_lats, node_lons = gridmet.node_grid(np.zeros((len(lats), len(lons))), lats, lons, lat_nodes, lon_nodes, *origin)
    columns = []
    for k in range(probe.size):
        unit = np.zeros(probe.size)
        unit[k] = 1.0
        columns.append(gridmet.bicubic(unit.reshape(probe.shape), node_lats, node_lons, grid.lats, grid.lons).ravel())
    return np.stack(columns, axis=1)


def node_box(grid, origin):
    """Node index ranges and the lat/lon box to fetch, exactly as gridmet.weather_layer computes them."""
    lat_nodes = gridmet._node_range(grid.lats.min(), grid.lats.max(), origin[0])
    lon_nodes = gridmet._node_range(grid.lons.min(), grid.lons.max(), origin[1])
    south, north = origin[0] + lat_nodes[0] * gridmet.NODE_DEG, origin[0] + (lat_nodes[1] + 1) * gridmet.NODE_DEG
    west, east = origin[1] + lon_nodes[0] * gridmet.NODE_DEG, origin[1] + (lon_nodes[1] + 1) * gridmet.NODE_DEG
    return lat_nodes, lon_nodes, (south, west, north, east)


NEW_ANCHOR = (gridmet.NODE_ORIGIN_LAT, gridmet.NODE_ORIGIN_LON)  # gridMET's corner (fixed in 5bef496)
OLD_ANCHOR = (0.0, 0.0)  # before 5bef496


def weather_layer_anchor(grid, var, days, origin):
    """gridmet.weather_layer with an explicit node anchor (to compare the old and fixed anchors).

    Same calls in the same order; with origin = NEW_ANCHOR it equals gridmet.weather_layer exactly
    (checked by check_anchor_mirror).
    """
    days = sorted(set(days))
    lat_nodes, lon_nodes, box = node_box(grid, origin)
    values, lats, lons = gridmet.fetch(var, days, *box)
    nodes, node_lats, node_lons = gridmet.node_grid(gridmet._mean_over_days(values), lats, lons, lat_nodes, lon_nodes, *origin)
    return gridmet.bicubic(nodes, node_lats, node_lons, grid.lats, grid.lons).astype(np.float32)


def check_anchor_mirror(grid, day):
    for var in WEATHER_CHANNELS:
        ours = weather_layer_anchor(grid, var, [day - dt.timedelta(days=1), day], NEW_ANCHOR)
        theirs, _ = gridmet.weather_layer(grid, var, [day - dt.timedelta(days=1), day])
        assert np.array_equal(ours, theirs, equal_nan=True), f"anchor mirror differs from weather_layer for {var}"


def daily_layers(grid, var, year, origin=NEW_ANCHOR):
    """Every day of one gridMET year as tile layers: (days, layers[n, 64, 64]).

    Same steps as gridmet.weather_layer (fetch, node_grid block mean, bicubic) for each single
    day, with one request for the whole year.
    """
    days = gridmet.available_days(var, year)
    lat_nodes, lon_nodes, box = node_box(grid, origin)
    values, lats, lons = gridmet.fetch(var, days, *box)
    nodes = np.stack([gridmet.node_grid(plane, lats, lons, lat_nodes, lon_nodes, *origin)[0] for plane in values])
    # Rare NaN nodes (coast, border) take the day's mean so the search can still score the day
    day_means = np.nanmean(nodes.reshape(len(days), -1), axis=1)
    nodes = np.where(np.isfinite(nodes), nodes, day_means[:, None, None])
    basis = _layer_basis(grid, lat_nodes, lon_nodes, lats, lons, origin)
    layers = (nodes.reshape(len(days), -1) @ basis.T).reshape(len(days), 64, 64)
    return days, layers


def find_fire_day(grid, stored):
    """Score every 2012-2020 day (single-day and two-day-mean windows) against the stored weather."""
    candidates = {}  # (window kind, day) -> list of normalized MAE per var
    for var in SEARCH_VARS:
        target = stored[var].astype(float)
        low, high = VALID[var]
        keep = (target >= low) & (target <= high)
        for year in YEARS:
            days, layers = daily_layers(grid, var, year)
            single = np.abs(layers - target)[:, keep].mean(axis=1)
            pair_layers = (layers[1:] + layers[:-1]) / 2
            pair = np.abs(pair_layers - target)[:, keep].mean(axis=1)
            for i, day in enumerate(days):
                candidates.setdefault(("single", day), {})[var] = single[i]
                if i:
                    candidates.setdefault(("pair", day), {})[var] = pair[i - 1]
    # Normalise each var by its median mismatch across all candidates, then average over vars
    medians = {var: np.median([c[var] for c in candidates.values() if var in c]) for var in SEARCH_VARS}
    scored = sorted(
        ((np.mean([c[var] / medians[var] for var in SEARCH_VARS]), key) for key, c in candidates.items()
         if all(var in c for var in SEARCH_VARS)),
        key=lambda item: item[0],
    )
    best_score, (kind, day) = scored[0]
    second = next(s for s in scored if abs((s[1][1] - day).days) > 3)
    return {"window_kind": kind, "weather_day": day, "score": float(best_score),
            "second_score": float(second[0]), "second_day": second[1][1], "second_kind": second[1][0],
            "best_mae": {var: float(candidates[(kind, day)][var]) for var in SEARCH_VARS}}


def terra_mask(grid, day):
    """MODIS Terra detections on one UTC day, rasterized on the tile (no VIIRS fallback)."""
    south, west, north, east = grid.bounds
    pad = fire_points.QUERY_PAD_DEG
    for attempt in range(4):  # FIRMS occasionally times out; archive answers are cached once they arrive
        try:
            table = fire_points.fire_points(south - pad, west - pad, north + pad, east + pad, day, "MODIS")
            break
        except fire_points.FirmsError:
            if attempt == 3:
                raise
            time.sleep(15 * (attempt + 1))
    mask, info = fire_points.fire_mask(grid, day, prefer="MODIS", satellites=("Terra",), points={"MODIS": table})
    return mask, info, table


def confirm_fire_day(grid, stored, weather_day):
    """Which t makes FIRMS Terra masks on t and t+1 match the stored PrevFireMask and FireMask."""
    prev, label = stored["PrevFireMask"], stored["FireMask"]
    rows = []
    for offset in (-1, 0, 1, 2):
        t = weather_day + dt.timedelta(days=offset)
        mask_t, _, _ = terra_mask(grid, t)
        mask_next, _, _ = terra_mask(grid, t + dt.timedelta(days=1))
        prev_scores = mask_scores(mask_t, prev, prev != -1)
        label_scores = mask_scores(mask_next, label, label != -1)
        rows.append({"t": t, "offset_from_weather_day": offset, "prev_iou": prev_scores["iou"],
                     "label_iou": label_scores["iou"], "prev": prev_scores, "label": label_scores})
    combined = lambda r: np.nansum([r["prev_iou"], r["label_iou"]])
    rows.sort(key=lambda r: -combined(r))
    return rows[0], rows


# ----------------------------------------------------------------------------- 3. windows and PDSI


def compare_windows(grid, stored, t):
    out = {}
    for name, offsets in WINDOWS.items():
        days = [t + dt.timedelta(days=o) for o in offsets]
        for var in WEATHER_CHANNELS:
            layer, _ = gridmet.weather_layer(grid, var, days)
            out[(name, var, "fixed")] = masked_stats(layer, stored[var], var)
            old = weather_layer_anchor(grid, var, days, OLD_ANCHOR)
            out[(name, var, "old (0,0)")] = masked_stats(old, stored[var], var)
    return out


def compare_pdsi(grid, stored, t):
    pentads = [p for p in gridmet.available_days(gridmet.PDSI, 0) if abs((p - t).days) <= 20]
    rows = []
    for pentad in pentads:
        layer, _ = gridmet.pdsi_layer(grid, pentad)  # latest pentad <= pentad is the pentad itself
        stats = masked_stats(layer, stored["pdsi"], "pdsi")
        rows.append({"pentad": pentad, "offset_days": (pentad - t).days, **stats})
    rows.sort(key=lambda r: r["mae"])
    rule = max(p for p in gridmet.available_days(gridmet.PDSI, 0) if p <= t)
    return rows, rule


# ----------------------------------------------------------------------------- per tile


def process_tile(item, stored_tiles):
    shard, record, lat0, lon0 = item
    path = TILE_DIR / f"{shard}_{record}.json"
    if path.exists():
        return json.loads(path.read_text())
    stored = stored_tiles[(shard, record)]
    started = time.time()
    where = refine_location(lat0, lon0, stored["elevation"])
    grid = TileGrid.around(where["lat"], where["lon"])
    found = find_fire_day(grid, stored)
    best_t, fire_rows = confirm_fire_day(grid, stored, found["weather_day"])
    t = best_t["t"]
    windows = compare_windows(grid, stored, t)
    pdsi_rows, pdsi_rule = compare_pdsi(grid, stored, t)
    result = {
        "shard": shard, "record": record, **{k: v for k, v in where.items()}, "shift_px": list(where["shift_px"]),
        "weather_match": {**found, "weather_day": found["weather_day"].isoformat(), "second_day": found["second_day"].isoformat()},
        "t": t.isoformat(),
        "fire_candidates": [
            {"t": r["t"].isoformat(), "offset_from_weather_day": r["offset_from_weather_day"],
             "prev_iou": r["prev_iou"], "label_iou": r["label_iou"],
             "prev_hits": r["prev"]["hits"], "prev_stored": r["prev"]["actual"], "prev_ours": r["prev"]["predicted"],
             "label_hits": r["label"]["hits"], "label_stored": r["label"]["actual"], "label_ours": r["label"]["predicted"]}
            for r in fire_rows
        ],
        "windows": [{"window": w, "var": v, "anchor": a, **s} for (w, v, a), s in windows.items()],
        "pdsi": [{**r, "pentad": r["pentad"].isoformat()} for r in pdsi_rows],
        "pdsi_rule_pentad": pdsi_rule.isoformat(),
        "stored_prev_fire_px": int((stored["PrevFireMask"] == 1).sum()),
        "stored_label_fire_px": int((stored["FireMask"] == 1).sum()),
        "seconds": round(time.time() - started, 1),
    }
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, default=float, indent=1))
    print(f"{shard} #{record}: t={t} ({found['window_kind']} weather day {found['weather_day']}, "
          f"score {found['score']:.3f} vs {found['second_score']:.3f}), prev IoU {best_t['prev_iou']:.2f}, "
          f"{result['seconds']}s", flush=True)
    return result


# ----------------------------------------------------------------------------- 4 and 5. rebuild and model


def rebuild_and_score(result, stored, best_window):
    """Rebuild the tile with build_inputs and run the model variants."""
    t = dt.date.fromisoformat(result["t"])
    grid = TileGrid.around(result["lat"], result["lon"])
    _, _, table = terra_mask(grid, t)
    weather_days = [t + dt.timedelta(days=o) for o in WINDOWS[best_window]]
    rebuilt = build_inputs(result["lat"], result["lon"], t, weather_days=weather_days, strict=True,
                           points={"MODIS": table})
    lagged = build_inputs(result["lat"], result["lon"], t, weather_days=[t - dt.timedelta(days=3), t - dt.timedelta(days=2)],
                          strict=True, points={"MODIS": table})
    stored_stack = np.stack([stored[name] for name in INPUT_FEATURES]).astype(np.float32)
    label = stored["FireMask"]
    valid = label != -1

    channels = []
    for c, name in enumerate(INPUT_FEATURES):
        if name == "PrevFireMask":
            s = mask_scores(rebuilt.inputs[c], stored[name], stored[name] != -1)
            channels.append({"channel": name, "corr": np.nan, "mae": np.nan, "iou": s["iou"],
                             "precision": s["precision"], "recall": s["recall"], "stored_px": s["actual"], "ours_px": s["predicted"]})
        else:
            channels.append({"channel": name, **masked_stats(rebuilt.inputs[c], stored[name], name)})
    label_ours, _ = fire_points.fire_mask(grid, t + dt.timedelta(days=1), prefer="MODIS", satellites=("Terra",),
                                          points={"MODIS": terra_mask(grid, t + dt.timedelta(days=1))[2]})
    s = mask_scores(label_ours, label, valid)
    channels.append({"channel": "FireMask (label, day t+1)", "iou": s["iou"], "precision": s["precision"],
                     "recall": s["recall"], "stored_px": s["actual"], "ours_px": s["predicted"]})

    fill = stored_stack.copy()
    for name in ("NDVI", "population"):
        fill[INPUT_FEATURES.index(name)] = np.nan
    stored_weather_ours_fire = stored_stack.copy()
    stored_weather_ours_fire[INPUT_FEATURES.index("PrevFireMask")] = rebuilt.inputs[INPUT_FEATURES.index("PrevFireMask")]
    ours_weather_stored_fire = rebuilt.inputs.copy()
    ours_weather_stored_fire[INPUT_FEATURES.index("PrevFireMask")] = stored_stack[INPUT_FEATURES.index("PrevFireMask")]
    variants = {
        "a stored inputs": stored_stack,
        "b stored, NDVI+population filled": fill,
        "c rebuilt (ours)": rebuilt.inputs,
        "d rebuilt, weather t-3,t-2 (live lag)": lagged.inputs,
        "e stored, our fire mask": stored_weather_ours_fire,
        "f rebuilt, stored fire mask": ours_weather_stored_fire,
    }
    probabilities = dict(zip(variants, predict(np.stack(list(variants.values())))))
    reference = probabilities["a stored inputs"]
    models = []
    for name, probability in probabilities.items():
        vs_label = mask_scores((probability >= THRESHOLD).astype(int), label, valid)
        vs_a = mask_scores((probability >= THRESHOLD).astype(int), (reference >= THRESHOLD).astype(int), np.ones_like(valid))
        models.append({"variant": name, "corr_vs_a": float(np.corrcoef(probability.ravel(), reference.ravel())[0, 1]),
                       "iou_vs_a": vs_a["iou"], "mean_prob": float(probability.mean()),
                       **{f"label_{k}": v for k, v in vs_label.items()}})
    persistence = mask_scores((stored["PrevFireMask"] == 1).astype(int), label, valid)
    models.append({"variant": "persistence (stored PrevFireMask)", **{f"label_{k}": v for k, v in persistence.items()}})
    return channels, models, rebuilt.layers


# ----------------------------------------------------------------------------- main


def firms_usage():
    try:
        response = requests.get("https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/",
                                params={"MAP_KEY": get_map_key()}, timeout=30)
        return response.json().get("current_transactions")
    except Exception as error:  # never let the key reach the output
        return f"unavailable ({type(error).__name__})"


def pooled(frame, prefix="label_"):
    hits, predicted, actual, union = (frame[f"{prefix}{k}"].sum() for k in ("hits", "predicted", "actual", "union"))
    return pd.Series({"precision": hits / predicted if predicted else np.nan, "recall": hits / actual if actual else np.nan,
                      "iou": hits / union if union else np.nan, "tiles": len(frame)})


def main():
    setup_caches()
    OUT.mkdir(parents=True, exist_ok=True)
    usage_before = firms_usage()
    print("FIRMS transactions before:", usage_before, flush=True)
    stored_tiles = read_stored_tiles()
    print(f"Read {len(stored_tiles)} located tiles", flush=True)

    first = LOCATED[0]
    check_anchor_mirror(TileGrid.around(first[2], first[3]), dt.date(2020, 10, 3))
    print("anchor mirror matches gridmet.weather_layer exactly", flush=True)

    def safe(item):
        try:
            return process_tile(item, stored_tiles)
        except Exception as error:
            print(f"FAILED {item[0]} #{item[1]}: {type(error).__name__}: {error}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = [r for r in pool.map(safe, LOCATED) if r is not None]

    # Per-tile summary
    summary = pd.DataFrame([{
        "tile": f"{r['shard']} #{r['record']}", "lat": round(r["lat"], 5), "lon": round(r["lon"], 5),
        "shift_px": tuple(r["shift_px"]), "elev_corr": round(r["elev_corr"], 4), "t": r["t"],
        "weather_kind": r["weather_match"]["window_kind"], "weather_day": r["weather_match"]["weather_day"],
        "score": round(r["weather_match"]["score"], 3), "second_score": round(r["weather_match"]["second_score"], 3),
        "separation": round(r["weather_match"]["second_score"] / max(r["weather_match"]["score"], 1e-9), 2),
        "t_minus_weather_day": r["fire_candidates"][0]["offset_from_weather_day"],
        "prev_iou": r["fire_candidates"][0]["prev_iou"], "label_iou": r["fire_candidates"][0]["label_iou"],
        "next_best_prev_iou": r["fire_candidates"][1]["prev_iou"],
        "stored_prev_px": r["stored_prev_fire_px"], "stored_label_px": r["stored_label_fire_px"],
        "pdsi_best_offset": r["pdsi"][0]["offset_days"], "pdsi_rule_offset": (dt.date.fromisoformat(r["pdsi_rule_pentad"]) - dt.date.fromisoformat(r["t"])).days,
    } for r in results])
    summary.to_csv(OUT / "tiles.csv", index=False)

    # Trustworthy dates: weather match clearly separated and fire masks agree
    confident = summary[(summary.separation >= 1.5) & (summary.prev_iou > 0.2)]
    confident_ids = set(confident.tile)

    windows = pd.DataFrame([{**w, "tile": f"{r['shard']} #{r['record']}"} for r in results for w in r["windows"]])
    windows.to_csv(OUT / "windows.csv", index=False)
    conf_windows = windows[windows.tile.isin(confident_ids)]
    win_table = (conf_windows[conf_windows.anchor == "fixed"]
                 .groupby(["var", "window"])[["corr", "mae"]].median().unstack("window"))

    pdsi = pd.DataFrame([{**p, "tile": f"{r['shard']} #{r['record']}", "t": r["t"]} for r in results for p in r["pdsi"]])
    pdsi.to_csv(OUT / "pdsi.csv", index=False)

    best_window = win_table["mae"].drop(index=["th", "pr"], errors="ignore").rank(axis=1).mean().idxmin()
    anchor_table = (conf_windows[conf_windows.window == best_window]
                    .groupby(["var", "anchor"])[["corr", "mae"]].median().unstack("anchor"))
    print("Best weather window:", best_window, flush=True)

    channel_rows, model_rows, layer_meta = [], [], {}
    for r in results:
        tile = f"{r['shard']} #{r['record']}"
        if tile not in confident_ids:
            continue
        channels, models, layers = rebuild_and_score(r, stored_tiles[(r["shard"], r["record"])], best_window)
        channel_rows += [{"tile": tile, **c} for c in channels]
        model_rows += [{"tile": tile, **m} for m in models]
        layer_meta[tile] = layers
        print(f"rebuilt {tile}", flush=True)
    channels = pd.DataFrame(channel_rows)
    models = pd.DataFrame(model_rows)
    channels.to_csv(OUT / "channels.csv", index=False)
    models.to_csv(OUT / "models.csv", index=False)

    usage_after = firms_usage()
    print("FIRMS transactions after:", usage_after, flush=True)
    write_report(summary, confident, win_table, best_window, pdsi, channels, models, usage_before, usage_after,
                 anchor_table)


def markdown_table(frame):
    """A DataFrame or Series as a markdown table (no tabulate dependency)."""
    frame = frame.to_frame() if isinstance(frame, pd.Series) else frame
    frame = frame.reset_index()
    frame.columns = [" ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c) for c in frame.columns]

    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{value:.3f}"
        return str(value)

    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def write_report(summary, confident, win_table, best_window, pdsi, channels, models, usage_before, usage_after,
                 anchor_table):
    fmt = markdown_table
    conf_pdsi = pdsi[pdsi.tile.isin(set(confident.tile))]
    best_pdsi = conf_pdsi.loc[conf_pdsi.groupby("tile")["mae"].idxmin()]
    pdsi_counts = best_pdsi.offset_days.value_counts().sort_index()
    channel_table = channels[channels.channel != "PrevFireMask"].groupby("channel")[["corr", "mae"]].median()
    fire = channels[channels.channel.isin(["PrevFireMask", "FireMask (label, day t+1)"])]
    fire_pooled = fire.groupby("channel").apply(
        lambda f: pd.Series({"stored_px": f.stored_px.sum(), "ours_px": f.ours_px.sum(),
                             "median_iou": f.iou.median(), "median_precision": f.precision.median(),
                             "median_recall": f.recall.median()}), include_groups=False)
    model_pooled = models.groupby("variant").apply(pooled, include_groups=False)
    model_agree = models.groupby("variant")[["corr_vs_a", "iou_vs_a", "mean_prob"]].median()
    text = f"""# Reconstruction check (plan step 2b)

Generated by `demos/reconstruct_spread_tiles.py`. {len(summary)} DEM-located training tiles;
{len(confident)} have a confidently identified fire day (weather match at least 1.5 times better
than any date more than 3 days away, and FIRMS Terra fire on t overlapping the stored
PrevFireMask with IoU > 0.2). Everything below the date table uses only those tiles.

## Fire day per tile

`weather_kind` is whether a single gridMET day or the mean of two days matched best;
`t_minus_weather_day` is how t (from the fire masks) relates to the matched weather day.

{fmt(summary.set_index('tile'))}

## Which gridMET days the release used

Median over confident tiles of per-tile correlation and MAE between our layer and the stored
channel, for each window relative to t. Best window by mean MAE rank (excluding th): **{best_window}**.

{fmt(win_table)}

### Node-grid anchor, before and after the 5bef496 fix

Window {best_window}; median per-tile correlation and MAE against the stored channel, with the
20 km node grid anchored at (0, 0) (old) and at gridMET's corner (fixed):

{fmt(anchor_table)}

## PDSI pentad

Offset (pentad date minus t, days) of the best-matching pentad per confident tile:

{markdown_table(pdsi_counts)}

Our rule (latest pentad dated on or before t) gives offsets of
{sorted(set(int(x) for x in confident.pdsi_rule_offset))} on these tiles.

## Whole-tile rebuild with build_inputs (window: {best_window})

Median per-tile correlation and MAE, our channel vs stored:

{fmt(channel_table)}

Fire masks (FIRMS MODIS Terra points vs stored MOD14A1, stored -1 pixels excluded):

{fmt(fire_pooled)}

## What the approximations cost the model

Pooled pixel scores against the stored FireMask label (-1 excluded), threshold {THRESHOLD}:

{fmt(model_pooled)}

Agreement with variant (a), the forecast from the stored inputs (median per tile):

{fmt(model_agree)}

FIRMS transactions: {usage_before} before, {usage_after} after (out of 5000 per 10 minutes).
"""
    (OUT / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
