"""Backtest the live spread forecast on past fires (plan step 3).

Builds model inputs for past fire days with the same code the API will use
(backend.spread_inputs.build_inputs), runs the trained model (backend.spread_model.predict), and
scores the forecast against the fire MODIS Terra actually saw the next day. Persistence
("tomorrow's fire = today's fire") is scored on the same tiles. See docs/spread_forecast_plan.md,
"Backtest design (step 3)".

Labels come from FIRMS MODIS Terra points on day t+1, rasterized on the same tile grid. FIRMS
points carry no cloud information, so there are no uncertain (-1) pixels: every pixel without a
detection counts as "no fire", which is stricter than the training labels.

Stages, both re-runnable:
  build  fetch FIRMS for sampled days, build every variant's inputs, save one .npz per tile
         under data/demos/backtest/tiles/ (already-built tiles are skipped)
  score  load the saved tiles, run the model, write report.md and CSVs

    python demos/backtest_spread.py --quick          # ~40 tiles, to check the code
    python demos/backtest_spread.py                  # full run (resumable: --max-minutes)
    python demos/backtest_spread.py --stage score    # re-score saved tiles only
"""

import argparse
import datetime as dt
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from backend import fire_points, gridmet, terrain  # noqa: E402
from backend.firms_client import FirmsError, fetch_area, get_map_key  # noqa: E402
from backend.spread_inputs import SpreadInputError, build_inputs  # noqa: E402
from backend.spread_grid import TileGrid  # noqa: E402
from backend.spread_model import INPUT_FEATURES, THRESHOLD, predict  # noqa: E402

OUTPUT = REPO / "data" / "demos" / "backtest"
CACHE = REPO / "data" / "cache" / "backtest"
SEED = 20261111

YEARS = range(2018, 2025)  # VIIRS NOAA-20 archive starts 2018-04-01
SEASON = ((6, 1), (10, 26))  # chunk start dates; a chunk covers 5 days, so the last ends Oct 30
CHUNK_DAYS = 5  # FIRMS DAY_RANGE maximum
FULL_CHUNKS = 42
QUICK_CHUNKS = 6
TILES_PER_DAY = 2
QUICK_TILES = 40
MIN_SEPARATION_KM = 50
# Tile centres stay inside gridMET's coverage (lat 25.07-49.40, lon -124.77 to -67.06)
CENTER_BOX = (25.6, -124.4, 48.9, -67.4)  # south, west, north, east
CONUS_AREA = "-125,24,-66,50"
KM_SUBSET_EVERY = 2  # every 2nd tile also gets the "km" geometry build (>= 100 tiles in a full run)

WEATHER_CHANNELS = ("th", "vs", "tmmn", "tmmx", "sph", "pr", "erc")


# ---------------------------------------------------------------- FIRMS


def firms_status():
    """Transactions used in the current 10-minute window (the key is never printed)."""
    try:
        response = requests.get(
            "https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/",
            params={"MAP_KEY": get_map_key()},
            timeout=30,
        )
        data = response.json()
        return {"current_transactions": data.get("current_transactions"), "limit": data.get("transaction_limit")}
    except Exception as error:  # status is informational only
        return {"error": type(error).__name__}


def chunk_points(source, start):
    """All CONUS detections from `source` for the 5 days from `start`, cached on disk."""
    path = CACHE / "firms" / f"{source}_{start.isoformat()}_{CHUNK_DAYS}d.csv"
    if path.exists():
        return pd.read_csv(path)
    table = fetch_area(source, area=CONUS_AREA, days=CHUNK_DAYS, date=start)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def day_points(table, source, day):
    """One UTC day of a chunk, tidied by the backend's own normalizer (satellite names etc.)."""
    south, west, north, east = (24.0, -125.0, 50.0, -66.0)
    return fire_points._tidy(table, source, day, south, west, north, east)


# ---------------------------------------------------------------- sampling


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def chunk_starts(count, rng):
    """Random 5-day windows in the fire season, spread over the years, no overlaps."""
    starts = set()
    per_year = math.ceil(count / len(YEARS))
    for year in YEARS:
        first = dt.date(year, *SEASON[0])
        last = dt.date(year, *SEASON[1])
        options = [first + dt.timedelta(days=d) for d in range(0, (last - first).days + 1, CHUNK_DAYS)]
        starts.update(rng.sample(options, min(per_year, len(options))))
    return sorted(rng.sample(sorted(starts), min(count, len(starts))))


def sample_cases(starts, rng):
    """Tile centres: Terra fire pixels on day t, at most TILES_PER_DAY per day, >= 50 km apart."""
    cases = []
    for start in starts:
        modis = chunk_points("MODIS_SP", start)
        for offset in range(CHUNK_DAYS - 1):  # day t+1 must be in the same chunk
            day = start + dt.timedelta(days=offset)
            points = day_points(modis, "MODIS_SP", day)
            terra = points[
                (points.satellite == "Terra")
                & points.latitude.between(CENTER_BOX[0], CENTER_BOX[2])
                & points.longitude.between(CENTER_BOX[1], CENTER_BOX[3])
            ]
            order = list(range(len(terra)))
            rng.shuffle(order)
            chosen = []
            for index in order:
                lat, lon = float(terra.latitude.iloc[index]), float(terra.longitude.iloc[index])
                if all(haversine_km(lat, lon, a, b) >= MIN_SEPARATION_KM for a, b in chosen):
                    chosen.append((lat, lon))
                if len(chosen) == TILES_PER_DAY:
                    break
            for lat, lon in chosen:
                cases.append({"chunk": start.isoformat(), "day": day.isoformat(), "lat": lat, "lon": lon})
    return cases


# ---------------------------------------------------------------- building


def case_id(case):
    return f"{case['day']}_{case['lat']:.4f}_{case['lon']:.4f}"


def terra_label(grid, modis_next):
    terra = modis_next[modis_next.satellite == "Terra"]
    return grid.rasterize(terra.latitude.to_numpy(float), terra.longitude.to_numpy(float))


def build_case(case, with_km):
    """Every variant's inputs for one tile, built with backend.spread_inputs.build_inputs."""
    start = dt.date.fromisoformat(case["chunk"])
    day = dt.date.fromisoformat(case["day"])
    nxt = day + dt.timedelta(days=1)
    modis_chunk = chunk_points("MODIS_SP", start)
    viirs_chunk = chunk_points("VIIRS_NOAA20_SP", start)
    modis_t = day_points(modis_chunk, "MODIS_SP", day)
    modis_t1 = day_points(modis_chunk, "MODIS_SP", nxt)
    viirs_t = day_points(viirs_chunk, "VIIRS_NOAA20_SP", day)
    both = {"MODIS": modis_t, "VIIRS": viirs_t}
    lat, lon = case["lat"], case["lon"]
    d = lambda n: day - dt.timedelta(days=n)  # noqa: E731

    variants = {
        # baseline: degree grid, MODIS Terra, training weather window (t-1, t)
        "base": dict(weather_days=[d(1), day], points=both),
        "weather_t": dict(weather_days=[day], points=both),
        "weather_t2_t1": dict(weather_days=[d(2), d(1)], points=both),
        "weather_live_t3_t2": dict(weather_days=[d(3), d(2)], points=both),
        "fire_terra_aqua": dict(weather_days=[d(1), day], points=both, fire_satellites=("Terra", "Aqua")),
        # Only VIIRS points are given, so a tile without VIIRS gets no fire (no MODIS fallback)
        "fire_viirs_n20": dict(weather_days=[d(1), day], points={"VIIRS": viirs_t}, fire_source="VIIRS"),
    }
    if with_km:
        variants["geometry_km"] = dict(weather_days=[d(1), day], points=both, geometry="km")

    arrays, meta, failures = {}, {"case": case, "variants": {}}, {}
    for name, kwargs in variants.items():
        try:
            built = build_inputs(lat, lon, day, strict=True, **kwargs)
        except (SpreadInputError, FirmsError, gridmet.GridmetError, terrain.TerrainError) as error:
            failures[name] = f"{type(error).__name__}: {error}"
            if name == "base":
                break
            continue
        arrays[f"inputs_{name}"] = built.inputs
        label = terra_label(built.grid, modis_t1)
        if name == "base":
            arrays["label"] = label
        elif name == "geometry_km":
            arrays["label_km"] = label
        fire = built.layers["PrevFireMask"]
        meta["variants"][name] = {
            "bounds": built.grid.bounds,
            "fire_source": fire.get("family"),
            "fire_fallback": fire.get("fallback"),
            "fire_points": fire.get("n_points"),
            "nan_share": {k: v.get("nan_share") for k, v in built.layers.items() if v.get("nan_share")},
        }
    meta["failures"] = failures
    return arrays, meta


def run_build(cases, max_minutes):
    tiles = OUTPUT / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    built_with = OUTPUT / "built_with.txt"
    if not built_with.exists():
        built_with.write_text(code_version() + "\n")
    started = time.time()
    done = failed = skipped = 0
    for position, case in enumerate(cases):
        path = tiles / f"{case_id(case)}.npz"
        if path.exists() or path.with_suffix(".failed.json").exists():
            skipped += 1
            continue
        if (time.time() - started) / 60 > max_minutes:
            print(f"Stopping after {max_minutes} min; run again to continue.", flush=True)
            break
        tile_started = time.time()
        arrays, meta = build_case(case, with_km=position % KM_SUBSET_EVERY == 0)
        if "inputs_base" not in arrays:
            path.with_suffix(".failed.json").write_text(json.dumps(meta, indent=1))
            failed += 1
            print(f"[{position + 1}/{len(cases)}] {case_id(case)} FAILED: {meta['failures']}", flush=True)
            continue
        np.savez_compressed(path, meta=json.dumps(meta), **arrays)
        done += 1
        print(
            f"[{position + 1}/{len(cases)}] {case_id(case)} {time.time() - tile_started:.1f}s"
            f" label px {int(arrays['label'].sum())} variants {len([k for k in arrays if k.startswith('inputs_')])}"
            + (f" failures {list(meta['failures'])}" if meta["failures"] else ""),
            flush=True,
        )
    remaining = len(cases) - skipped - done - failed
    print(f"built {done}, failed {failed}, already there {skipped}, remaining {remaining}", flush=True)
    return remaining


# ---------------------------------------------------------------- scoring

# The 8 symmetries of a square, as (forward, inverse) on the last two axes
TRANSFORMS = {
    "identity": (lambda a: a, lambda a: a),
    "flip_rows": (lambda a: a[..., ::-1, :], lambda a: a[..., ::-1, :]),
    "flip_cols": (lambda a: a[..., :, ::-1], lambda a: a[..., :, ::-1]),
    "rot180": (lambda a: a[..., ::-1, ::-1], lambda a: a[..., ::-1, ::-1]),
    "transpose": (lambda a: np.swapaxes(a, -1, -2), lambda a: np.swapaxes(a, -1, -2)),
    "rot90": (lambda a: np.rot90(a, 1, axes=(-2, -1)), lambda a: np.rot90(a, -1, axes=(-2, -1))),
    "rot270": (lambda a: np.rot90(a, -1, axes=(-2, -1)), lambda a: np.rot90(a, 1, axes=(-2, -1))),
    "anti_transpose": (
        lambda a: np.swapaxes(a[..., ::-1, ::-1], -1, -2),
        lambda a: np.swapaxes(a[..., ::-1, ::-1], -1, -2),
    ),
}


def average_precision(probabilities, targets):
    """Area under the precision-recall curve, the same definition as train_spread.ipynb."""
    order = np.argsort(-probabilities, kind="stable")
    hits = targets[order].astype(np.float64)
    if hits.sum() == 0:
        return float("nan")
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def counts(predicted, actual):
    """Per-tile hits, predicted, actual, union (arrays of shape (tiles,))."""
    axes = (1, 2)
    return {
        "hits": (predicted & actual).sum(axes),
        "predicted": predicted.sum(axes),
        "actual": actual.sum(axes),
        "union": (predicted | actual).sum(axes),
    }


def pooled(c, index=None):
    sel = (lambda v: v[index].sum(-1)) if index is not None else (lambda v: v.sum())
    hits, predicted, actual, union = (sel(c[k]) for k in ("hits", "predicted", "actual", "union"))
    return {
        "precision": hits / np.maximum(predicted, 1),
        "recall": hits / np.maximum(actual, 1),
        "iou": hits / np.maximum(union, 1),
    }


def score(name, probabilities, label, groups, persistence=None, reps=1000, seed=SEED):
    """Model (and optional persistence) scores with tile- and chunk-bootstrap 95% intervals."""
    actual = label == 1
    model = counts(probabilities >= THRESHOLD, actual)
    row = {"variant": name, "tiles": len(label), "label_fire_px": int(actual.sum())}
    p = pooled(model)
    row.update({f"model_{k}": float(v) for k, v in p.items()})
    row["model_pr_auc"] = average_precision(probabilities.ravel(), actual.ravel())
    union = model["union"] > 0
    row["model_mean_tile_iou"] = float((model["hits"][union] / model["union"][union]).mean()) if union.any() else float("nan")
    row["model_forecast_px"] = int(model["predicted"].sum())

    rng = np.random.default_rng(seed)
    tile_index = rng.integers(0, len(label), (reps, len(label)))
    # Cluster bootstrap: resample whole 5-day chunks (tiles in a chunk share fires and weather)
    unique_groups = np.unique(groups)
    members = [np.nonzero(groups == g)[0] for g in unique_groups]
    chunk_index = [np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))]) for _ in range(reps)]

    def interval(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    model_iou_tiles = pooled(model, tile_index)["iou"]
    row["model_iou_ci"] = interval(model_iou_tiles)
    if persistence is not None:
        pers = counts(persistence == 1, actual)
        q = pooled(pers)
        row.update({f"persistence_{k}": float(v) for k, v in q.items()})
        union_p = pers["union"] > 0
        row["persistence_mean_tile_iou"] = float((pers["hits"][union_p] / pers["union"][union_p]).mean()) if union_p.any() else float("nan")
        pers_iou_tiles = pooled(pers, tile_index)["iou"]
        row["persistence_iou_ci"] = interval(pers_iou_tiles)
        row["iou_diff"] = row["model_iou"] - row["persistence_iou"]
        row["iou_diff_ci_tiles"] = interval(model_iou_tiles - pers_iou_tiles)
        chunk_model = np.array([pooled({k: v[i] for k, v in model.items()})["iou"] for i in chunk_index])
        chunk_pers = np.array([pooled({k: v[i] for k, v in pers.items()})["iou"] for i in chunk_index])
        row["iou_diff_ci_chunks"] = interval(chunk_model - chunk_pers)
    return row


def load_tiles():
    records = []
    for path in sorted((OUTPUT / "tiles").glob("*.npz")):
        with np.load(path) as data:
            records.append({key: (json.loads(str(data[key])) if key == "meta" else data[key]) for key in data.files})
    return records


def region_of(lat, lon):
    west_east = "West" if lon < -104 else "Central" if lon < -90 else "East"
    return f"{west_east}-{'N' if lat >= 37 else 'S'}"


def nan_out(inputs, channels):
    inputs = inputs.copy()
    for name in channels:
        inputs[:, INPUT_FEATURES.index(name)] = np.nan
    return inputs


# Which detections make "today's fire" in each variant (the label is always Terra on day t+1)
VARIANT_FIRE = {"fire_terra_aqua": ("MODIS", ("Terra", "Aqua")), "fire_viirs_n20": ("VIIRS", ("N20", "N21"))}
CENTRE = 0.0  # footprint_mask scale 0 marks only the pixel holding each detection's centre


class FireMasks:
    """Rebuilds the PrevFireMask channel and the labels of saved tiles from the cached FIRMS points,
    with backend.fire_points.footprint_mask at any scale, the way fire_mask selects points:
    that day's detections from the variant's satellites in the tile's padded query box, and an
    empty mask when none is centred inside the tile (fire_mask then has nothing to use)."""

    def __init__(self):
        self._points = {}

    def points(self, chunk, family, day):
        source = "MODIS_SP" if family == "MODIS" else "VIIRS_NOAA20_SP"
        key = (chunk, source, day)
        if key not in self._points:
            self._points[key] = day_points(chunk_points(source, dt.date.fromisoformat(chunk)), source, day)
        return self._points[key]

    @staticmethod
    def in_box(grid, points, satellites):
        south, west, north, east = grid.bounds
        pad = fire_points.QUERY_PAD_DEG
        return points[
            points.satellite.isin(satellites)
            & points.latitude.between(south - pad, north + pad)
            & points.longitude.between(west - pad, east + pad)
        ]

    def masks(self, case, variant, in_scale, label_scale):
        day = dt.date.fromisoformat(case["day"])
        grid = TileGrid.around(case["lat"], case["lon"], geometry="km" if variant == "geometry_km" else "degree")
        family, satellites = VARIANT_FIRE.get(variant, ("MODIS", ("Terra",)))
        today = self.in_box(grid, self.points(case["chunk"], family, day), satellites)
        rows, _ = grid.cell_of(today.latitude.to_numpy(float), today.longitude.to_numpy(float))
        if (rows >= 0).any():
            fire_today = fire_points.footprint_mask(grid, today, in_scale)
        else:
            fire_today = np.zeros((grid.size, grid.size), np.float32)
        tomorrow = self.in_box(grid, self.points(case["chunk"], "MODIS", day + dt.timedelta(days=1)), ("Terra",))
        return fire_today, fire_points.footprint_mask(grid, tomorrow, label_scale)


def apply_fire_masks(records, builder, in_scale, label_scale, check_stored=False):
    """Overwrite each saved tile's PrevFireMask channels and labels in place. With check_stored
    (centre pixels for both), count tiles whose rebuilt masks differ from what build_inputs saved."""
    fire = INPUT_FEATURES.index("PrevFireMask")
    mismatches = 0
    for record in records:
        case = record["meta"]["case"]
        for key in [k for k in record if k.startswith("inputs_")]:
            variant = key[len("inputs_"):]
            fire_today, label = builder.masks(case, variant, in_scale, label_scale)
            if check_stored:
                stored_label = record["label_km"] if variant == "geometry_km" else record["label"]
                if not (np.array_equal(record[key][fire], fire_today) and np.array_equal(stored_label, label)):
                    mismatches += 1
            record[key][fire] = fire_today
            if variant == "base":
                record["label"] = label
            elif variant == "geometry_km":
                record["label_km"] = label
    return mismatches


def gate_rows(records, title):
    """Baseline (training weather window) and gate (live weather lag) rows for one mask definition."""
    fire = INPUT_FEATURES.index("PrevFireMask")
    out = []
    for key, name in (("base", "weather t-1..t"), ("weather_live_t3_t2", "live lag t-3..t-2")):
        chosen = [r for r in records if f"inputs_{key}" in r]
        inputs = np.stack([r[f"inputs_{key}"] for r in chosen])
        label = np.stack([r["label"] for r in chosen])
        groups = np.array([r["meta"]["case"]["chunk"] for r in chosen])
        out.append(score(f"{title}; {name}", predict(inputs), label, groups, persistence=inputs[:, fire]))
    return out


def run_score(records, out_dir, mask_note, label_rows=None):
    failed = sorted((OUTPUT / "tiles").glob("*.failed.json"))
    if not records:
        raise SystemExit("No built tiles under data/demos/backtest/tiles")
    fire = INPUT_FEATURES.index("PrevFireMask")

    def stack(name, subset=None):
        chosen = [r for r in records if f"inputs_{name}" in r and (subset is None or subset(r))]
        return chosen, np.stack([r[f"inputs_{name}"] for r in chosen])

    def groups_of(chosen):
        return np.array([r["meta"]["case"]["chunk"] for r in chosen])

    base_records, base = stack("base")
    label = np.stack([r["label"] for r in base_records])
    groups = groups_of(base_records)
    base_prob = predict(base)

    rows = []
    rows.append(score("baseline: degree, Terra, weather t-1..t", base_prob, label, groups, persistence=base[:, fire]))

    # Orientation: transform the inputs, predict, transform the forecast back
    orientation_rows = []
    for name, (forward, inverse) in TRANSFORMS.items():
        prob = base_prob if name == "identity" else np.ascontiguousarray(inverse(predict(np.ascontiguousarray(forward(base)))))
        orientation_rows.append(score(f"orientation: {name}", prob, label, groups))

    weather = {
        "weather_t": "weather: day t only",
        "weather_t2_t1": "weather: t-2..t-1 (one day earlier)",
        "weather_live_t3_t2": "weather: t-3..t-2 (live lag)",
    }
    fires = {"fire_terra_aqua": "fire: Terra + Aqua", "fire_viirs_n20": "fire: VIIRS NOAA-20"}
    for key, title in {**weather, **fires}.items():
        chosen, inputs = stack(key)
        # Same Terra day-t+1 label as the baseline, on the same (degree) grid
        lab = np.stack([r["label"] for r in chosen])
        rows.append(score(title, predict(inputs), lab, groups_of(chosen), persistence=inputs[:, fire]))

    # Same-tile comparisons for the km subset
    km_records, km = stack("geometry_km")
    if km_records:
        km_label = np.stack([r["label_km"] for r in km_records])
        deg_inputs = np.stack([r["inputs_base"] for r in km_records])
        deg_label = np.stack([r["label"] for r in km_records])
        rows.append(score(f"geometry: degree (km subset, {len(km_records)} tiles)", predict(deg_inputs), deg_label, groups_of(km_records), persistence=deg_inputs[:, fire]))
        rows.append(score(f"geometry: km (same {len(km_records)} tiles)", predict(km), km_label, groups_of(km_records), persistence=km[:, fire]))

    ablated = nan_out(base, WEATHER_CHANNELS + ("pdsi",))
    rows.append(score("ablation: no weather or drought (all NaN)", predict(ablated), label, groups, persistence=base[:, fire]))

    # The gate: live weather lag and the live fire source (MODIS Terra; fallback never needed
    # because every tile is centred on a Terra pixel)
    gate = next(r for r in rows if r["variant"].startswith("weather: t-3..t-2"))

    # Breakdown of the baseline by region, and tiles with / without fire tomorrow
    breakdown = []
    regions = np.array([region_of(r["meta"]["case"]["lat"], r["meta"]["case"]["lon"]) for r in base_records])
    for region in sorted(set(regions)):
        keep = regions == region
        breakdown.append(score(f"region {region}", base_prob[keep], label[keep], groups[keep], persistence=base[keep][:, fire]))
    has_fire = label.sum((1, 2)) > 0
    breakdown.append(score("tiles with fire on day t+1", base_prob[has_fire], label[has_fire], groups[has_fire], persistence=base[has_fire][:, fire]))

    # Do the rebuilt inputs look like the training inputs? Compare with the train-split stats.
    stats = pd.read_csv(REPO / "data" / "training" / "spread_feature_stats.csv", index_col="feature")
    input_rows = []
    for channel, name in enumerate(INPUT_FEATURES):
        values = base[:, channel]
        finite = values[np.isfinite(values)]
        train = stats.loc[name]
        input_rows.append({
            "channel": name,
            "nan_share": float(1 - finite.size / values.size),
            "backtest_mean": float(finite.mean()) if finite.size else float("nan"),
            "backtest_std": float(finite.std()) if finite.size else float("nan"),
            "train_mean": float(train["mean"]),
            "train_std": float(train["std"]),
            "mean_shift_in_train_std": float((finite.mean() - train["mean"]) / train["std"]) if finite.size else float("nan"),
            "share_outside_train_clip": float(((finite < train["low"]) | (finite > train["high"])).mean()) if finite.size else float("nan"),
        })

    summary = {
        "code_version": code_version(),
        "built_with": (OUTPUT / "built_with.txt").read_text().strip() if (OUTPUT / "built_with.txt").exists() else "unknown",
        "tiles": len(base_records),
        "failed_tiles": len(failed),
        "days": len({r["meta"]["case"]["day"] for r in base_records}),
        "chunks": len(set(groups)),
        "years": sorted({r["meta"]["case"]["day"][:4] for r in base_records}),
        "regions": {k: int(v) for k, v in zip(*np.unique(regions, return_counts=True))},
        "label_fire_px": int(label.sum()),
        "today_fire_px": int(base[:, fire].sum()),
        "tiles_without_fire_tomorrow": int((~has_fire).sum()),
        "variant_failures": {},
        "viirs_tiles_without_detections": sum(
            1 for r in records if r["meta"]["variants"].get("fire_viirs_n20", {}).get("fire_points") == 0
        ),
        "tiles_with_nan_channels": sum(1 for r in base_records if r["meta"]["variants"]["base"]["nan_share"]),
    }
    for r in records:
        for name, message in r["meta"]["failures"].items():
            summary["variant_failures"].setdefault(name, []).append(f"{case_id(r['meta']['case'])}: {message}")
    failed_messages = [json.loads(p.read_text())["failures"] for p in failed]

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "scores.csv", index=False)
    pd.DataFrame(orientation_rows).to_csv(out_dir / "orientation.csv", index=False)
    pd.DataFrame(breakdown).to_csv(out_dir / "breakdown.csv", index=False)
    pd.DataFrame(input_rows).to_csv(out_dir / "inputs_vs_training.csv", index=False)
    pd.DataFrame([r["meta"]["case"] | {"region": region_of(r["meta"]["case"]["lat"], r["meta"]["case"]["lon"])} for r in base_records]).to_csv(out_dir / "cases.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary | {"failed_tile_messages": failed_messages}, indent=1, default=str))
    write_report(out_dir, mask_note, rows, orientation_rows, breakdown, gate, summary, failed_messages, input_rows, label_rows)
    return rows, orientation_rows, breakdown, gate, summary


def pct(value):
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{100 * value:.1f}%"


def ci(pair, scale=1.0, digits=3):
    return f"[{pair[0] * scale:.{digits}f}, {pair[1] * scale:.{digits}f}]"


def write_report(out_dir, mask_note, rows, orientation_rows, breakdown, gate, summary, failed_messages, input_rows, label_rows=None):
    lines = [
        "# Spread forecast backtest",
        "",
        f"Generated by `demos/backtest_spread.py` on {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC"
        f": inputs built with code version {summary['built_with']}, fire masks rebuilt and scored with {summary['code_version']}.",
        "",
        "Inputs are built with `backend.spread_inputs.build_inputs` (the live code path) for past fire days,",
        "the model is `backend.spread_model.predict` with threshold "
        f"{THRESHOLD}, and the label is MODIS Terra detections on day t+1 (FIRMS archive) rasterized on the",
        "same tile grid. FIRMS points carry no cloud mask, so there are no uncertain (-1) pixels: any pixel",
        "without a detection counts as no fire. This is stricter than the training labels, and Terra",
        "sees each place only about twice a day, so absolute scores are not comparable with the test-split",
        "numbers in data/training/spread_results.csv. Model and persistence are compared on the same tiles.",
        "",
        f"**Fire masks in this report:** {mask_note}",
        "",
        "## Cases",
        "",
        f"- {summary['tiles']} tiles on {summary['days']} days in {summary['chunks']} five-day windows, years {', '.join(summary['years'])}, June to October.",
        f"- Regions (West/Central/East split at 104° W and 90° W, N/S at 37° N): "
        + ", ".join(f"{k} {v}" for k, v in summary["regions"].items()),
        f"- Fire pixels: {summary['today_fire_px']} on day t, {summary['label_fire_px']} on day t+1; "
        f"{summary['tiles_without_fire_tomorrow']} tiles have no Terra fire on day t+1.",
        f"- Each tile is centred on a random Terra fire pixel on day t (at most {TILES_PER_DAY} per day, at least {MIN_SEPARATION_KM} km apart).",
        f"- Failed tiles: {summary['failed_tiles']}. Tiles with some NaN input pixels (coast or border): {summary['tiles_with_nan_channels']}.",
        "",
        "Intervals are 95% bootstrap intervals over tiles (1000 resamples). Tiles in the same five-day window",
        "share fires and weather, so the difference is also bootstrapped over whole windows (chunk CI).",
        "",
        "## Gate",
        "",
    ]
    diff_ci = gate["iou_diff_ci_chunks"]
    passed = gate["iou_diff"] > 0 and diff_ci[0] > 0
    lines += [
        f"With the live weather lag (gridMET days t-3 and t-2) and the live fire source (MODIS Terra):",
        f"model IoU {gate['model_iou']:.3f} {ci(gate['model_iou_ci'])} vs persistence {gate['persistence_iou']:.3f} "
        f"{ci(gate['persistence_iou_ci'])}; difference {gate['iou_diff']:+.3f}, tile CI {ci(gate['iou_diff_ci_tiles'])}, "
        f"chunk CI {ci(diff_ci)}.",
        "",
        f"**Verdict: {'PASS' if passed else 'FAIL'}** "
        + (
            "(the model beats persistence and the chunk interval excludes 0)."
            if passed
            else "(the model does not beat persistence with an interval that excludes 0)."
        ),
        "",
        "## Model vs persistence",
        "",
        "| Variant | Tiles | Model P | Model R | Model IoU [95% CI] | PR-AUC | Mean tile IoU | Forecast / burned px | Persist. P | Persist. R | Persist. IoU [95% CI] | IoU diff [chunk CI] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['tiles']} | {pct(r['model_precision'])} | {pct(r['model_recall'])} | "
            f"{r['model_iou']:.3f} {ci(r['model_iou_ci'])} | {r['model_pr_auc']:.3f} | {r['model_mean_tile_iou']:.3f} | "
            f"{r['model_forecast_px']} / {r['label_fire_px']} | "
            f"{pct(r.get('persistence_precision'))} | {pct(r.get('persistence_recall'))} | "
            f"{r['persistence_iou']:.3f} {ci(r['persistence_iou_ci'])} | {r['iou_diff']:+.3f} {ci(r['iou_diff_ci_chunks'])} |"
        )
    lines += [
        "",
        "## Orientation",
        "",
        "Each of the 8 symmetries of the square applied to the baseline input stack, with the forecast turned",
        "back before scoring. `identity` is the orientation TileGrid produces (south-up, as the training tiles).",
        "",
        "| Transform | Model P | Model R | Model IoU [95% CI] | PR-AUC |",
        "|---|---|---|---|---|",
    ]
    for r in orientation_rows:
        lines.append(f"| {r['variant'].split(': ')[1]} | {pct(r['model_precision'])} | {pct(r['model_recall'])} | {r['model_iou']:.3f} {ci(r['model_iou_ci'])} | {r['model_pr_auc']:.3f} |")
    lines += [
        "",
        "## Baseline breakdown",
        "",
        "| Subset | Tiles | Model IoU [95% CI] | Persistence IoU [95% CI] | IoU diff [chunk CI] |",
        "|---|---|---|---|---|",
    ]
    for r in breakdown:
        lines.append(
            f"| {r['variant']} | {r['tiles']} | {r['model_iou']:.3f} {ci(r['model_iou_ci'])} | "
            f"{r['persistence_iou']:.3f} {ci(r['persistence_iou_ci'])} | {r['iou_diff']:+.3f} {ci(r['iou_diff_ci_chunks'])} |"
        )
    if label_rows:
        lines += [
            "",
            "## Fire mask definitions",
            "",
            "The same tiles and inputs, with today's fire (input and persistence) and tomorrow's fire (label)",
            "drawn as centre pixels or as detection footprints (`footprint_mask`, scale x FIRMS scan/track).",
            "",
            "| Inputs; labels; weather | Model P | Model R | Model IoU [95% CI] | Forecast / burned px | Persist. P | Persist. R | Persist. IoU [95% CI] | IoU diff [chunk CI] | Verdict |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in label_rows:
            verdict = "PASS" if r["iou_diff"] > 0 and r["iou_diff_ci_chunks"][0] > 0 else "FAIL"
            lines.append(
                f"| {r['variant']} | {pct(r['model_precision'])} | {pct(r['model_recall'])} | {r['model_iou']:.3f} {ci(r['model_iou_ci'])} | "
                f"{r['model_forecast_px']} / {r['label_fire_px']} | {pct(r['persistence_precision'])} | {pct(r['persistence_recall'])} | "
                f"{r['persistence_iou']:.3f} {ci(r['persistence_iou_ci'])} | {r['iou_diff']:+.3f} {ci(r['iou_diff_ci_chunks'])} | {verdict} |"
            )
    lines += [
        "",
        "## Rebuilt inputs vs training inputs",
        "",
        "Baseline tiles' raw inputs compared with the train-split stats the model was scaled with",
        "(`spread_feature_stats.csv`: clip range, mean, std after clipping). A large mean shift or many",
        "values outside the clip range means the model sees something unlike its training data.",
        "",
        "| Channel | NaN share | Backtest mean | Train mean | Mean shift (train std) | Backtest std | Train std | Outside train clip |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in input_rows:
        lines.append(
            f"| {r['channel']} | {pct(r['nan_share'])} | {r['backtest_mean']:.4g} | {r['train_mean']:.4g} | "
            f"{r['mean_shift_in_train_std']:+.2f} | {r['backtest_std']:.4g} | {r['train_std']:.4g} | {pct(r['share_outside_train_clip'])} |"
        )
    lines += ["", "## Failures", ""]
    if not summary["variant_failures"] and not failed_messages:
        lines.append("None.")
    for name, messages in summary["variant_failures"].items():
        lines.append(f"- {name}: {len(messages)} tiles, e.g. {messages[0]}")
    for message in failed_messages[:10]:
        lines.append(f"- failed tile: {message}")
    lines += [
        "",
        f"VIIRS variant: {summary['viirs_tiles_without_detections']} tiles had no NOAA-20 detection on day t, so their",
        "today's-fire layer is empty (no MODIS fallback in this variant).",
        "",
        "CSVs: `scores.csv`, `orientation.csv`, `breakdown.csv`, `inputs_vs_training.csv`, `cases.csv`; `summary.json`.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def code_version():
    """git HEAD, plus a warning if backend/ has uncommitted changes (they'd shape the numbers)."""
    import subprocess

    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()

    head = git("rev-parse", "--short", "HEAD")
    dirty = git("status", "--porcelain", "--", "backend")
    return head + (" with uncommitted backend changes" if dirty else "")


# ---------------------------------------------------------------- main


def score_all():
    """Score every fire-mask definition. The footprint (scale FOOTPRINT_SCALE) is the baseline:
    it is closest to the MOD14A1 masks the model was trained and tested on."""
    records = load_tiles()
    if not records:
        raise SystemExit("No built tiles under data/demos/backtest/tiles")
    builder = FireMasks()
    fp = fire_points.FOOTPRINT_SCALE
    configs = [
        ("centre pixels; centre pixels", CENTRE, CENTRE),
        (f"footprint x{fp}; centre pixels", fp, CENTRE),
        (f"centre pixels; footprint x{fp}", CENTRE, fp),
        ("footprint x1.0; footprint x1.0", 1.0, 1.0),
        (f"footprint x{fp}; footprint x{fp}", fp, fp),
    ]
    label_rows, mismatches = [], None
    for title, in_scale, label_scale in configs:
        check = in_scale == CENTRE and label_scale == CENTRE
        count = apply_fire_masks(records, builder, in_scale, label_scale, check_stored=check)
        if check:
            mismatches = count
            # Full report for the centre-pixel version (what build_inputs saved before cb082c5)
            run_score(records, OUTPUT / "centre_pixel",
                      "centre pixels for today's fire and for the label (one pixel per FIRMS detection), "
                      f"as built before cb082c5. Rebuilt masks differing from the saved ones: {count} tile-variants.")
        label_rows += gate_rows(records, title)
    # The footprint baseline is applied last, so the records now hold it
    run_score(records, OUTPUT,
              f"detection footprints at scale {fp} (backend.fire_points.footprint_mask, the default since cb082c5) "
              "for today's fire, persistence and the label. The centre-pixel version is in centre_pixel/report.md. "
              f"Check: centre-pixel masks rebuilt from cached FIRMS points differ from the saved ones in {mismatches} tile-variants.",
              label_rows)
    print((OUTPUT / "report.md").read_text().split("## Orientation")[0][-6000:])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help=f"about {QUICK_TILES} tiles, to check the code")
    parser.add_argument("--stage", choices=("all", "build", "score"), default="all")
    parser.add_argument("--max-minutes", type=float, default=1e9, help="stop building after this long (resume later)")
    args = parser.parse_args()

    global OUTPUT
    if args.quick:
        OUTPUT = OUTPUT / "quick"
    gridmet.CACHE_DIR = CACHE / "gridmet"
    terrain.CACHE_DIR = CACHE / "terrain"
    fire_points.CACHE_DIR = CACHE / "fire_points"

    started = time.time()
    print("FIRMS before:", firms_status(), flush=True)
    rng = random.Random(SEED)
    if args.stage in ("all", "build"):
        starts = chunk_starts(QUICK_CHUNKS if args.quick else FULL_CHUNKS, rng)
        cases = sample_cases(starts, rng)
        if args.quick:
            cases = sorted(rng.sample(cases, min(QUICK_TILES, len(cases))), key=lambda c: (c["day"], c["lat"]))
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "planned_cases.json").write_text(json.dumps(cases, indent=1))
        print(f"{len(cases)} cases from {len(starts)} windows", flush=True)
        remaining = run_build(cases, args.max_minutes)
        if remaining and args.stage == "all":
            print("Build not finished; score later with --stage score.", flush=True)
            print("FIRMS after:", firms_status(), flush=True)
            return
    if args.stage in ("all", "score"):
        score_all()
    print("FIRMS after:", firms_status(), flush=True)
    print(f"Took {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
