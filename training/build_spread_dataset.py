"""Build the v2 spread-model training set from archived VIIRS NOAA-20 fires, gridMET and terrain.

Follows docs/spread_retrain_plan.md ("Data") and the channel order and file format in
backend/spread_model_v2.py. Every channel comes from the same verified code the live forecast
uses: TileGrid for geometry, fire_points.footprint_mask for fire, terrain.elevation, and the
gridMET recipe (here from whole yearly files via backend/gridmet_bulk.py, checked equal to the
OPeNDAP path).

Run (low priority, as the plan's Mac safety rules require):
    OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 nice -n 10 python -m training.build_spread_dataset --quick
    OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 nice -n 10 python -m training.build_spread_dataset
The build is resumable: rerunning skips tiles already written.
"""

import argparse
import datetime as dt
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backend import fire_points, terrain
from backend.firms_client import US_BOUNDS, FirmsError, fetch_area
from backend.gridmet_bulk import Bulk
from backend.spread_grid import PIXEL_DEG, TileGrid
from backend.spread_model_v2 import (
    CHANNELS,
    FIRE_FOOTPRINT_SCALE,
    SPLIT_YEARS,
    VIIRS_SATELLITES,
    WEATHER_OFFSETS,
)

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "retrain"
CACHE = REPO / "data" / "cache" / "retrain"
FIRMS_SOURCE = "VIIRS_NOAA20_SP"  # archive covers 2018-04-01 to 2026-07-31; all training years
SEASON = ((4, 1), (11, 30))  # fire days t from April 1 to November 30
TARGETS = {"train": 25000, "val": 4000, "test": 4000}
QUICK_TARGETS = {"train": 300, "val": 100, "test": 100}
DEDUP_KM = 40  # one tile per cluster of day-t fire: centres at least this far apart on a day
MAX_OFFSET_PX = 16  # random shift of the tile centre so the fire isn't always dead centre
QUERY_PAD_DEG = fire_points.QUERY_PAD_DEG  # footprints of detections just outside the tile still count
FIRMS_PAUSE_SECONDS = 5  # each CONUS 5-day archive request costs ~10 FIRMS transactions
SEED = 2026
REGIONS = ((-104.0, "West"), (-90.0, "Central"), (180.0, "East"))  # by longitude, then N/S at 37°N
MIN_FREE_GB = 50.0
TILE_ATTEMPTS = 6
NEEDED_DAYS = (-2, -1, 0, 1)  # fire_t2, fire_t1, fire_t, label
TILE_RETRY_SECONDS = 120


def log(message):
    print(f"[{dt.datetime.now():%H:%M:%S}] {message}", flush=True)


def region_of(lat, lon):
    for limit, name in REGIONS:
        if lon < limit:
            return f"{name}-{'N' if lat >= 37 else 'S'}"
    return "East-N" if lat >= 37 else "East-S"


def split_of(day):
    for split, years in SPLIT_YEARS.items():
        if day.year in years:
            return split
    return None


def season_days(year):
    first = dt.date(year, *SEASON[0])
    last = dt.date(year, *SEASON[1])
    return [first + dt.timedelta(days=i) for i in range((last - first).days + 1)]


# ----------------------------------------------------------------------------- FIRMS


def _normalize(table):
    """Detections in a fixed layout: numbers parsed, satellite normalized, N20 only."""
    if table is None or len(table) == 0:
        return pd.DataFrame(columns=["latitude", "longitude", "acq_date", "satellite", "frp", "scan", "track", "type"])
    out = pd.DataFrame({
        "latitude": pd.to_numeric(table["latitude"], errors="coerce"),
        "longitude": pd.to_numeric(table["longitude"], errors="coerce"),
        "acq_date": table["acq_date"].astype(str),
        "satellite": table["satellite"].map(fire_points.normalize_satellite),
        "frp": pd.to_numeric(table["frp"], errors="coerce"),
        "scan": pd.to_numeric(table["scan"], errors="coerce"),
        "track": pd.to_numeric(table["track"], errors="coerce"),
        "type": pd.to_numeric(table["type"], errors="coerce") if "type" in table else np.nan,
    })
    out = out[out["satellite"].isin(VIIRS_SATELLITES) & out["latitude"].notna() & out["longitude"].notna()]
    return out.reset_index(drop=True)


def load_firms_year(year, cache_dir=CACHE / "firms"):
    """VIIRS NOAA-20 detections for CONUS covering the season of `year` plus 2 days before and 1 after."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    first = dt.date(year, *SEASON[0]) - dt.timedelta(days=2)
    last = dt.date(year, *SEASON[1]) + dt.timedelta(days=1)
    frames, start = [], first
    while start <= last:
        path = cache_dir / f"{FIRMS_SOURCE}_{start.isoformat()}_5.csv.gz"
        if path.exists():
            table = pd.read_csv(path, dtype={"acq_date": str})
        else:
            for attempt in range(10):  # rides out a dropped connection of up to ~25 minutes
                try:
                    table = fetch_area(FIRMS_SOURCE, US_BOUNDS, days=5, date=start, timeout=120)
                    break
                except FirmsError as error:  # message is key-free (firms_client redacts it)
                    log(f"FIRMS {start}: {error}; retrying")
                    time.sleep(min(30 * (attempt + 1), 300))
            else:
                raise RuntimeError(f"FIRMS kept failing for {start}")
            partial = path.with_name(path.name + ".part")
            table.to_csv(partial, index=False, compression="gzip")
            os.replace(partial, path)
            time.sleep(FIRMS_PAUSE_SECONDS)
        frames.append(_normalize(table))
        start += dt.timedelta(days=5)
    table = pd.concat(frames, ignore_index=True)
    return {day: group.reset_index(drop=True) for day, group in table.groupby("acq_date")}


# ----------------------------------------------------------------------------- tile centres


def cluster_centres(points, dedup_km=DEDUP_KM):
    """One centre per cluster of fire: greedy by FRP, dropping detections within dedup_km of a chosen one.

    points: DataFrame with latitude, longitude, frp. Returns a list of (lat, lon).
    """
    if len(points) == 0:
        return []
    order = np.argsort(-np.nan_to_num(points["frp"].to_numpy(dtype=float), nan=0.0), kind="stable")
    lats = points["latitude"].to_numpy(dtype=float)[order]
    lons = points["longitude"].to_numpy(dtype=float)[order]
    alive = np.ones(len(lats), dtype=bool)
    centres = []
    for i in range(len(lats)):
        if not alive[i]:
            continue
        centres.append((float(lats[i]), float(lons[i])))
        dy = (lats - lats[i]) * 111.32
        dx = (lons - lons[i]) * 111.32 * np.cos(np.radians(lats[i]))
        alive &= dy * dy + dx * dx > dedup_km * dedup_km
    return centres


def gridmet_coverage(bulk, year=2020, day_index=180):
    """A function telling whether a point is inside gridMET's data (contiguous US land), from one real plane."""
    plane_file = bulk.year_file("tmmx", year)
    raw = plane_file.raw(day_index, 0, len(plane_file.lats) - 1, 0, len(plane_file.lons) - 1)
    valid = ~np.isin(raw, plane_file.fills)
    lat0, lon0 = plane_file.lats[0], plane_file.lons[0]
    step = abs(plane_file.lats[1] - plane_file.lats[0])

    def covered(lat, lon):
        i = int(round((lat0 - lat) / step))
        j = int(round((lon - lon0) / step))
        return 0 <= i < valid.shape[0] and 0 <= j < valid.shape[1] and bool(valid[i, j])

    return covered


class DayIndex:
    """One day's vegetation detections, binned for fast neighbourhood lookups, as global 1 km cells."""

    BIN_DEG = 0.25

    def __init__(self, points):
        veg = points[points["type"] == 0] if "type" in points and points["type"].notna().any() else points
        self.lat = veg["latitude"].to_numpy(dtype=float)
        self.lon = veg["longitude"].to_numpy(dtype=float)
        self.row = np.floor(self.lat / PIXEL_DEG).astype(np.int64)
        self.col = np.floor(self.lon / PIXEL_DEG).astype(np.int64)
        keys = np.floor(self.lat / self.BIN_DEG).astype(np.int64) * 100000 + np.floor(self.lon / self.BIN_DEG).astype(np.int64)
        order = np.argsort(keys, kind="stable")
        self._keys, self._order = keys[order], order

    def near(self, lat, lon, km):
        """Indices of detections within `km` of (lat, lon)."""
        reach = int(np.ceil(km / 111.32 / self.BIN_DEG / max(np.cos(np.radians(abs(lat) + 1)), 0.2))) + 1
        bi, bj = int(np.floor(lat / self.BIN_DEG)), int(np.floor(lon / self.BIN_DEG))
        found = []
        for di in range(-reach, reach + 1):
            for dj in range(-reach, reach + 1):
                key = (bi + di) * 100000 + (bj + dj)
                lo, hi = np.searchsorted(self._keys, key), np.searchsorted(self._keys, key, side="right")
                found.append(self._order[lo:hi])
        idx = np.concatenate(found) if found else np.zeros(0, dtype=np.int64)
        dy = (self.lat[idx] - lat) * 111.32
        dx = (self.lon[idx] - lon) * 111.32 * np.cos(np.radians(lat))
        return idx[dy * dy + dx * dx <= km * km]


CLUSTER_KM = 20  # the day-t cluster: vegetation detections within this distance of the centre
PERSIST_PX = 5  # day t-1 fire counts as the same fire within this many pixels of a day-t cluster cell


def cluster_features(today, yesterday, lat, lon):
    """(cluster_px_t, persist_px_t1) for a candidate centre.

    cluster_px_t: distinct 1 km cells with day-t vegetation fire within CLUSTER_KM of the centre.
    persist_px_t1: distinct day t-1 cells within PERSIST_PX pixels (Chebyshev) of any of those cells,
    i.e. how much of this fire was already burning the day before.
    """
    idx_t = today.near(lat, lon, CLUSTER_KM)
    if len(idx_t) == 0:
        return 0, 0
    cells_t = np.unique(np.stack([today.row[idx_t], today.col[idx_t]], axis=1), axis=0)
    if yesterday is None:
        return len(cells_t), 0
    idx_y = yesterday.near(lat, lon, CLUSTER_KM + PERSIST_PX * 1.2)
    if len(idx_y) == 0:
        return len(cells_t), 0
    cells_y = np.unique(np.stack([yesterday.row[idx_y], yesterday.col[idx_y]], axis=1), axis=0)
    gap = np.abs(cells_y[:, None, :] - cells_t[None, :, :]).max(axis=2).min(axis=1)
    return len(cells_t), int((gap <= PERSIST_PX).sum())


GROWING_MIN_PX_T = 10  # stratum "growing": the day-t cluster covers at least this many 1 km cells ...
GROWING_MIN_PERSIST = 3  # ... or at least this many day t-1 cells were already burning next to it


def stratum_of(cluster_px_t, persist_px_t1):
    """"growing" for large or ongoing fires, "small" for the rest (see docs/spread_retrain_plan.md)."""
    return "growing" if cluster_px_t >= GROWING_MIN_PX_T or persist_px_t1 >= GROWING_MIN_PERSIST else "small"


def plan_tiles(fires_by_day, targets, rng, covered=None):
    """Candidate tiles per split (clusters of vegetation fire on day t), sampled 50/50 by stratum.

    covered(lat, lon) -> bool drops centres outside gridMET (Mexico, Canada, sea), which would have
    no weather at all. Each tile records its stratum, the stratum's candidate count in its split and
    its sampling probability, so metrics can be re-weighted to the natural (live) mix.
    """
    candidates, indexes = [], {}

    def index(day_text):
        if day_text not in indexes:
            indexes[day_text] = DayIndex(fires_by_day[day_text])
        return indexes[day_text]

    for day_text, points in fires_by_day.items():
        day = dt.date.fromisoformat(day_text)
        split = split_of(day)
        if split is None or not (dt.date(day.year, *SEASON[0]) <= day <= dt.date(day.year, *SEASON[1])):
            continue
        # A fire-season day with no NOAA-20 detections anywhere in CONUS is a data gap (e.g. NOAA-20
        # sent nothing on 2023-07-19..22 while Suomi NPP did), not "no fire": skip tiles whose
        # history or label would fall in one, or the gap would read as fire dying out
        if any((day + dt.timedelta(days=k)).isoformat() not in fires_by_day for k in NEEDED_DAYS):
            continue
        vegetation = points[points["type"] == 0] if points["type"].notna().any() else points
        yesterday = index((day - dt.timedelta(days=1)).isoformat())
        for lat, lon in cluster_centres(vegetation):
            if covered is not None and not covered(lat, lon):
                continue
            px_t, persist = cluster_features(index(day_text), yesterday, lat, lon)
            candidates.append({
                "split": split, "fire_day": day_text, "fire_lat": lat, "fire_lon": lon,
                "cluster_px_t": px_t, "persist_px_t1": persist, "stratum": stratum_of(px_t, persist),
            })
    candidates = pd.DataFrame(candidates)
    counts = candidates.groupby(["split", "stratum"]).size()
    chosen = []
    for split, target in targets.items():
        pool = candidates[candidates["split"] == split]
        growing, small = pool[pool["stratum"] == "growing"], pool[pool["stratum"] == "small"]
        n_growing = min(len(growing), target // 2)
        n_small = min(len(small), target - n_growing)
        n_growing = min(len(growing), target - n_small)  # top up if small runs short
        for stratum, group, n in (("growing", growing, n_growing), ("small", small, n_small)):
            if n == 0:
                continue
            take = group.sample(n=n, random_state=int(rng.integers(1 << 31)))
            take = take.assign(stratum_candidates=len(group), sample_prob=n / len(group))
            chosen.append(take)
    plan = pd.concat(chosen, ignore_index=True)
    offsets = rng.integers(-MAX_OFFSET_PX, MAX_OFFSET_PX + 1, size=(len(plan), 2))
    plan["offset_rows"], plan["offset_cols"] = offsets[:, 0], offsets[:, 1]
    plan["center_lat"] = plan["fire_lat"] + plan["offset_rows"] * PIXEL_DEG
    plan["center_lon"] = plan["fire_lon"] + plan["offset_cols"] * PIXEL_DEG
    plan["region"] = [region_of(a, b) for a, b in zip(plan["center_lat"], plan["center_lon"])]
    # Date order inside each split keeps gridMET reads sequential (each day-block decoded once)
    plan["split_order"] = plan["split"].map({"train": 0, "val": 1, "test": 2})
    plan = plan.sort_values(["split_order", "fire_day", "center_lat"], kind="stable").drop(columns="split_order")
    plan["index"] = plan.groupby("split").cumcount()
    candidate_counts = {
        split: {stratum: int(counts.get((split, stratum), 0)) for stratum in ("growing", "small")}
        for split in targets
    }
    return plan.reset_index(drop=True), candidate_counts


# ----------------------------------------------------------------------------- one tile


def points_near(points, grid):
    """That day's detections inside the tile's box padded by QUERY_PAD_DEG (like fire_points.fire_mask)."""
    if points is None or len(points) == 0:
        return pd.DataFrame(columns=["latitude", "longitude", "frp", "scan", "track"])
    south, west, north, east = grid.bounds
    keep = (
        points["latitude"].between(south - QUERY_PAD_DEG, north + QUERY_PAD_DEG)
        & points["longitude"].between(west - QUERY_PAD_DEG, east + QUERY_PAD_DEG)
    )
    return points[keep]


def frp_layer(grid, points):
    """log1p of summed FRP (MW) of detections whose centre is in each pixel; 0 where there are none."""
    total = np.zeros((grid.size, grid.size), dtype=np.float64)
    if len(points):
        rows, cols = grid.cell_of(points["latitude"].to_numpy(dtype=float), points["longitude"].to_numpy(dtype=float))
        frp = np.nan_to_num(points["frp"].to_numpy(dtype=float), nan=0.0)
        inside = rows >= 0
        np.add.at(total, (rows[inside], cols[inside]), frp[inside])
    return np.log1p(total).astype(np.float32)


def build_tile(row, fires_by_day, bulk):
    """All channels (in CHANNELS order) and the label for one planned tile."""
    grid = TileGrid.around(row.center_lat, row.center_lon)
    t = dt.date.fromisoformat(row.fire_day)

    def day_points(offset):
        return points_near(fires_by_day.get((t + dt.timedelta(days=offset)).isoformat()), grid)

    today = day_points(0)
    layers = {
        "fire_t": fire_points.footprint_mask(grid, today, FIRE_FOOTPRINT_SCALE),
        "fire_t1": fire_points.footprint_mask(grid, day_points(-1), FIRE_FOOTPRINT_SCALE),
        "fire_t2": fire_points.footprint_mask(grid, day_points(-2), FIRE_FOOTPRINT_SCALE),
        "frp_t": frp_layer(grid, today),
        "elevation": terrain.elevation(grid),
    }
    weather_days = [t + dt.timedelta(days=offset) for offset in WEATHER_OFFSETS]
    layers["wind_u"], layers["wind_v"] = bulk.wind_layers(grid, weather_days)
    for var in ("tmmn", "tmmx", "sph", "pr", "erc"):
        layers[var] = bulk.weather_layer(grid, var, weather_days)
    layers["pdsi"], _ = bulk.pdsi_layer(grid, t)
    missing = [name for name in CHANNELS if name not in layers]
    if missing:
        raise RuntimeError(f"channels not built: {missing}")
    inputs = np.stack([layers[name] for name in CHANNELS]).astype(np.float32)
    label = fire_points.footprint_mask(grid, day_points(1), FIRE_FOOTPRINT_SCALE).astype(np.uint8)
    return inputs, label


# ----------------------------------------------------------------------------- stats and checks


MASK_CHANNELS = ("fire_t", "fire_t1", "fire_t2")  # 0/1 masks: range fixed at [0, 1]
SPARSE_CHANNELS = ("frp_t",)  # mostly 0: range 0 .. 99.9th percentile of the non-zero values


def channel_stats(inputs, rng, sample_tiles=4000):
    """Per channel: clip range, then mean and std of the clipped values over all pixels.

    Dense channels clip at their 0.1/99.9 percentiles (on a sample of tiles). Fire masks and FRP
    are almost all 0 (well over 99.9% of pixels), so all-pixel percentiles would give low = high = 0
    and erase them: masks keep [0, 1], and FRP clips at the 99.9th percentile of its non-zero values.
    """
    n = len(inputs)
    picked = np.sort(rng.choice(n, size=min(n, sample_tiles), replace=False))
    rows = []
    for channel, name in enumerate(CHANNELS):
        sample = np.asarray(inputs[picked, channel], dtype=np.float32).ravel()
        sample = sample[np.isfinite(sample)]
        if name in MASK_CHANNELS:
            low, high = 0.0, 1.0
        elif name in SPARSE_CHANNELS:
            nonzero = sample[sample > 0]
            low, high = 0.0, (float(np.percentile(nonzero, 99.9)) if len(nonzero) else np.nan)
        else:
            low, high = (np.percentile(sample, (0.1, 99.9)) if len(sample) else (np.nan, np.nan))
        if not (np.isfinite(low) and np.isfinite(high) and low < high):
            raise ValueError(f"channel {name}: bad clip range low={low} high={high}")
        total = total_sq = count = 0.0
        for start in range(0, n, 1000):
            block = np.asarray(inputs[start : start + 1000, channel], dtype=np.float64)
            block = block[np.isfinite(block)]
            block = np.clip(block, low, high)
            total += block.sum()
            total_sq += np.square(block).sum()
            count += block.size
        mean = total / count if count else np.nan
        std = np.sqrt(max(total_sq / count - mean * mean, 0.0)) if count else np.nan
        rows.append({"feature": name, "low": low, "high": high, "mean": mean, "std": max(std, 1e-6), "nan_share": 1 - count / (n * 64 * 64)})
    return pd.DataFrame(rows).set_index("feature")


def free_gb(path):
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize / 1e9


def peak_memory_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


# ----------------------------------------------------------------------------- main


def terrain_tiles_needed(plan):
    """Distinct Terrarium tiles the plan needs, and how many aren't cached on disk yet."""
    needed = set()
    for row in plan.itertuples():
        needed.update(terrain.tiles_covering(TileGrid.around(row.center_lat, row.center_lon).bounds, terrain.ZOOM))
    cache = CACHE / "terrarium"
    missing = [tile for tile in needed if not (cache / f"{terrain.ZOOM}_{tile[0]}_{tile[1]}.png").exists()]
    return needed, missing


def build(out_dir, targets, seed=SEED, plan_only=False):
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    terrain.CACHE_DIR = CACHE / "terrarium"
    terrain.DOWNLOAD_THREADS = 2  # politeness: at most 2 concurrent tile downloads
    rng = np.random.default_rng(seed)
    bulk = Bulk()

    years = sorted({year for years in SPLIT_YEARS.values() for year in years})
    plan_path = out_dir / "plan.csv"
    fires = {}
    log(f"loading VIIRS N20 detections for {years[0]}-{years[-1]}")
    for year in years:
        fires.update(load_firms_year(year))
        gaps = [d.isoformat() for d in season_days(year) if d.isoformat() not in fires]
        log(f"  {year}: {sum(1 for day in fires if day.startswith(str(year)))} days with detections; "
            f"season days with no NOAA-20 data: {gaps or 'none'}")
    if plan_path.exists():
        plan = pd.read_csv(plan_path, dtype={"fire_day": str})
        log(f"resuming plan with {len(plan)} tiles")
    else:
        plan, candidate_counts = plan_tiles(fires, targets, rng, covered=gridmet_coverage(bulk))
        plan.to_csv(plan_path, index=False)
        (out_dir / "candidates.json").write_text(json.dumps(candidate_counts))
        log(f"planned {plan.groupby(['split', 'stratum']).size().to_dict()} from candidates {candidate_counts}")
    if plan_only:
        needed, missing = terrain_tiles_needed(plan)
        log(f"terrain tiles needed: {len(needed)}, not yet cached: {len(missing)}")
        return plan

    for split in ("train", "val", "test"):
        rows = plan[plan["split"] == split]
        n = len(rows)
        if n == 0:
            continue
        inputs_path, labels_path, done_path = (
            out_dir / f"{split}_inputs.npy", out_dir / f"{split}_labels.npy", out_dir / f"{split}_done.npy"
        )
        need_gb = n * len(CHANNELS) * 64 * 64 * 2 / 1e9
        if not inputs_path.exists() and free_gb(out_dir) - need_gb < MIN_FREE_GB:
            raise SystemExit(f"Stopping: building {split} would leave less than {MIN_FREE_GB} GB free")
        mode = "r+" if inputs_path.exists() else "w+"
        inputs = np.lib.format.open_memmap(inputs_path, mode=mode, dtype=np.float16, shape=(n, len(CHANNELS), 64, 64))
        labels = np.lib.format.open_memmap(labels_path, mode=mode, dtype=np.uint8, shape=(n, 64, 64))
        done = np.load(done_path) if done_path.exists() else np.zeros(n, dtype=bool)
        todo = np.nonzero(~done)[0]
        log(f"{split}: {n} tiles, {len(todo)} to build")
        split_started, built = time.time(), 0

        def checkpoint():
            inputs.flush()
            labels.flush()
            np.save(done_path, done)

        for index in todo:
            row = rows.iloc[index]
            assert row["index"] == index
            for attempt in range(TILE_ATTEMPTS):
                try:
                    tile_inputs, tile_label = build_tile(row, fires, bulk)
                    break
                except Exception as error:  # network trouble (terrain tiles, PDSI): wait, then retry
                    if attempt == TILE_ATTEMPTS - 1:
                        checkpoint()
                        log(f"{split} tile {index} failed {TILE_ATTEMPTS} times ({type(error).__name__}: {error}); stopping, rerun to resume")
                        raise
                    log(f"{split} tile {index}: {type(error).__name__}: {error}; retrying in {TILE_RETRY_SECONDS}s")
                    time.sleep(TILE_RETRY_SECONDS)
            inputs[index] = tile_inputs.astype(np.float16)
            labels[index] = tile_label
            done[index] = True
            built += 1
            if built % 50 == 0 or built == len(todo):
                checkpoint()
                rate = (time.time() - split_started) / built
                log(
                    f"{split}: {done.sum()}/{n} built, {rate:.2f}s/tile, ETA {rate * (len(todo) - built) / 60:.0f} min, "
                    f"peak mem {peak_memory_gb():.2f} GB, free disk {free_gb(out_dir):.0f} GB"
                )
                if free_gb(out_dir) < MIN_FREE_GB:
                    raise SystemExit("Stopping: free disk dropped below the safety margin")
        del inputs, labels

    # Index and stats
    tiles = plan.copy()
    for column in ("fire_t_px", "label_px", "new_fire_px"):
        tiles[column] = 0
    fire_t = CHANNELS.index("fire_t")
    for split in ("train", "val", "test"):
        mask = tiles["split"] == split
        if not mask.any():
            continue
        inputs = np.load(out_dir / f"{split}_inputs.npy", mmap_mode="r")
        labels = np.load(out_dir / f"{split}_labels.npy", mmap_mode="r")
        today = np.asarray(inputs[:, fire_t]) == 1
        tomorrow = np.asarray(labels) == 1
        positions = tiles.loc[mask, "index"].to_numpy()
        tiles.loc[mask, "fire_t_px"] = today.sum((1, 2))[positions]
        tiles.loc[mask, "label_px"] = tomorrow.sum((1, 2))[positions]
        tiles.loc[mask, "new_fire_px"] = (tomorrow & ~today).sum((1, 2))[positions]
    tiles[["split", "index", "fire_day", "center_lat", "center_lon", "fire_t_px", "label_px", "new_fire_px", "region",
           "stratum", "stratum_candidates", "sample_prob", "cluster_px_t", "persist_px_t1",
           "fire_lat", "fire_lon", "offset_rows", "offset_cols"]].to_csv(out_dir / "tiles.csv", index=False)
    train = np.load(out_dir / "train_inputs.npy", mmap_mode="r")
    channel_stats(train, rng).to_csv(out_dir / "channel_stats.csv")
    log(f"done in {(time.time() - started) / 60:.1f} min; peak memory {peak_memory_gb():.2f} GB")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="small sample in data/retrain/quick")
    parser.add_argument("--stats-only", action="store_true", help="only recompute channel_stats.csv from built train arrays")
    parser.add_argument("--plan-only", action="store_true", help="only plan the tiles (plan.csv) and estimate downloads")
    args = parser.parse_args()
    out_dir = OUT_DIR / "quick" if args.quick else OUT_DIR
    if args.stats_only:
        train = np.load(out_dir / "train_inputs.npy", mmap_mode="r")
        channel_stats(train, np.random.default_rng(SEED)).to_csv(out_dir / "channel_stats.csv")
        log(f"wrote {out_dir / 'channel_stats.csv'}")
        return
    targets = QUICK_TARGETS if args.quick else TARGETS
    build(out_dir, targets, plan_only=args.plan_only)


if __name__ == "__main__":
    sys.exit(main())
