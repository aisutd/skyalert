"""Live next-day fire spread forecast with the v2 model (models/spread_v2).

The v2 model was trained on tiles built by training/build_spread_dataset.py. This module builds
the same 13 channels for any place from live data, with the same code, so training and live use
match:
- fire: VIIRS NOAA-20 detections (FIRMS) on days t, t-1 and t-2, footprint masks at scale 1.0,
  plus log1p of summed FRP on day t
- elevation: AWS Terrarium tiles (backend/terrain.py)
- weather: gridMET, mean of days t-2 and t-1, through the exact recipe of backend/gridmet_bulk.py
  (Bulk.weather_layer / wind_layers) with live OPeNDAP requests instead of local yearly files
- drought: gridMET PDSI, latest pentad on or before t

Day t is the latest complete UTC day of NOAA-20 detections; the forecast is for t+1.
Test scores (2024-2025 fire seasons, never used for training or selection) are in
docs/spread_v2_test_results.txt; the headline numbers are in TEST_SCORES.
"""

import datetime as dt
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from backend import fire_points, gridmet, terrain
from backend.firms_client import US_BOUNDS, FirmsError, fetch_area
from backend.gridmet_bulk import Bulk
from backend.spread_grid import TileGrid
from backend.spread_model_v2 import (
    CHANNELS,
    FIRE_FOOTPRINT_SCALE,
    VIIRS_SATELLITES,
    WEATHER_OFFSETS,
    load_run,
    predict,
    select_channels,
)

REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "models" / "spread_v2"

# NOAA-20 passes over the contiguous US around 08:30 and 20:30 UTC, and FIRMS NRT data arrives
# within about 3 hours, so a UTC day is complete by about 03:00 UTC the next day.
DAY_COMPLETE_HOUR_UTC = 4
CACHE_SECONDS = 30 * 60
CACHE_MAX_ENTRIES = 200
# The frontend asks for several fires at once; build at most this many at a time so FIRMS, gridMET
# and the terrain tiles never see a burst of requests (each build makes about 20)
MAX_CONCURRENT_BUILDS = 3
NEAR_PX = 2  # "chance near the fire" = highest chance within this many pixels (~2 km) of the clicked spot  # forecasts kept in memory (each holds ~8k numbers)
DAY_CACHE_SECONDS = 10 * 60  # nationwide NOAA-20 detections of a recent day (NRT can still grow)

# docs/spread_v2_test_results.txt (4,000 test tiles from the 2024-2025 fire seasons, never used for
# training or selection). The test sample is half "growing" fires and half small ones; "all_fires"
# re-weights it to the natural mix of fires (most are small and short-lived), which is what users
# clicking live fires see. IoU = overlap score: pixels both forecast and burning, divided by pixels
# either forecast or burning.
TEST_SCORES = {
    "test_period": "2024-2025 fire seasons",
    "all_fires": {
        "model": {"iou": 0.224, "precision": 0.443, "recall": 0.312, "new_fire_recall": 0.067, "new_fire_precision": 0.324},
        "persistence": {"iou": 0.162, "precision": 0.219, "recall": 0.382},
    },
    "growing_fires": {
        "model": {"iou": 0.318, "precision": 0.461, "recall": 0.507, "new_fire_recall": 0.156, "new_fire_precision": 0.333},
        "persistence": {"iou": 0.258, "precision": 0.364, "recall": 0.470},
    },
}


class ForecastUnavailable(Exception):
    """The forecast can't be made honestly (for example, today's fire couldn't be loaded)."""


class _LiveWeather(Bulk):
    """gridmet_bulk.Bulk's weather recipe, fed by live OPeNDAP requests instead of local files."""

    def __init__(self):  # no local files needed
        pass

    def fetch(self, var, days, south, west, north, east):
        return gridmet.fetch(var, days, south, west, north, east)


_weather = _LiveWeather()
_cache = {}
_cache_lock = threading.Lock()
_day_cache = {}  # (FIRMS source, day) -> (fetched_at, NOAA-20 detections over the contiguous US)
_day_lock = threading.Lock()
_build_slots = threading.BoundedSemaphore(MAX_CONCURRENT_BUILDS)


def model_status():
    """Whether the model can run (weights, stats and manifest in models/spread_v2), for /api/model-status."""
    missing = [str(path.relative_to(REPO)) for path in (MODEL_DIR / name for name in ("weights.pt", "channel_stats.csv", "manifest.json")) if not path.exists()]
    if missing:
        return {"ready": False, "missing": missing, "note": "The spread model files are missing from models/spread_v2."}
    try:
        _, _, manifest = load_run(MODEL_DIR)
    except Exception as error:  # a broken file: say so instead of failing every request
        return {"ready": False, "missing": [], "note": f"Couldn't load the spread model: {type(error).__name__}: {error}"}
    return {"ready": True, "model": "spread_v2", "threshold": float(manifest["threshold"]), "missing_layers": []}


def latest_fire_day(now=None):
    """The latest complete UTC day of NOAA-20 detections."""
    now = now or dt.datetime.now(dt.timezone.utc)
    lag = 1 if now.hour >= DAY_COMPLETE_HOUR_UTC else 2
    return now.date() - dt.timedelta(days=lag)


def _noaa20_source(day):
    """The FIRMS NOAA-20 source covering `day` (archive if it has it, else near-real-time)."""
    noaa20 = [source for source in fire_points.sources_for(day, "VIIRS") if "NOAA20" in source]
    if not noaa20:
        raise ForecastUnavailable(f"No NOAA-20 fire data covers {day.isoformat()} yet.")
    return noaa20[0]


def _noaa20_day(day):
    """Every NOAA-20 detection over the contiguous US on one UTC day, the box the training data used.

    Fetched once per day and shared by every forecast. A day with no NOAA-20 detections anywhere
    in the US is a data outage (it happened on 2023-07-19..22), not a quiet day: training skipped
    such days, so the forecast refuses instead of reading the gap as "no fire".
    """
    source = _noaa20_source(day)
    key = (source, day)
    with _day_lock:
        hit = _day_cache.get(key)
        if hit and (source.endswith("_SP") or time.time() - hit[0] < DAY_CACHE_SECONDS):
            return hit[1]
    west, south, east, north = (float(value) for value in US_BOUNDS.split(","))
    table = fire_points._tidy(fetch_area(source, US_BOUNDS, days=1, date=day), source, day, south, west, north, east)
    table = table[table["satellite"].isin(VIIRS_SATELLITES)]
    if len(table) == 0:
        raise ForecastUnavailable(
            f"NOAA-20 reported no fires anywhere in the US on {day.isoformat()}, which means a data gap, not a quiet day."
        )
    with _day_lock:
        for stale in [k for k, (at, _) in _day_cache.items() if not k[0].endswith("_SP") and time.time() - at > DAY_CACHE_SECONDS]:
            del _day_cache[stale]
        _day_cache[key] = (time.time(), table)
    return table


def _viirs_points(grid, day):
    """NOAA-20 detections on `day` in the tile's box padded like the dataset builder."""
    south, west, north, east = grid.bounds
    pad = fire_points.QUERY_PAD_DEG
    table = _noaa20_day(day)
    keep = table["latitude"].between(south - pad, north + pad) & table["longitude"].between(west - pad, east + pad)
    return table[keep]


def _frp_layer(grid, points):
    """log1p of summed FRP (MW) of detections whose centre is in each pixel (as in the dataset builder)."""
    total = np.zeros((grid.size, grid.size), dtype=np.float64)
    if len(points):
        rows, cols = grid.cell_of(points["latitude"].to_numpy(dtype=float), points["longitude"].to_numpy(dtype=float))
        frp = np.nan_to_num(points["frp"].to_numpy(dtype=float), nan=0.0)
        inside = rows >= 0
        np.add.at(total, (rows[inside], cols[inside]), frp[inside])
    return np.log1p(total).astype(np.float32)


def build_inputs(lat, lon, fire_day, weather_days=None):
    """(grid, inputs (13, 64, 64) in CHANNELS order, info) for the tile around (lat, lon) on fire_day.

    Raises ForecastUnavailable if the fire detections can't be loaded: without them there is no
    honest forecast. Weather or terrain that fails is left NaN (the training average) and reported.
    """
    grid = TileGrid.around(lat, lon)
    fire_days = {offset: fire_day + dt.timedelta(days=offset) for offset in (0, -1, -2)}
    weather_note = None
    if weather_days is None:
        try:
            weather_days = live_weather_days(fire_day)
        except Exception as error:  # gridMET unreachable: try the training days; failures become NaN below
            weather_days = [fire_day + dt.timedelta(days=offset) for offset in WEATHER_OFFSETS]
            weather_note = f"Couldn't check gridMET's latest day ({type(error).__name__}: {error})."

    jobs = {f"fire_{offset}": (lambda day=day: _viirs_points(grid, day)) for offset, day in fire_days.items()}
    jobs["elevation"] = lambda: terrain.elevation(grid)
    jobs["wind"] = lambda: _weather.wind_layers(grid, weather_days)
    for var in ("tmmn", "tmmx", "sph", "pr", "erc"):
        jobs[var] = lambda var=var: _weather.weather_layer(grid, var, weather_days)
    jobs["pdsi"] = lambda: gridmet.pdsi_layer(grid, fire_day)

    results, errors = {}, {}
    # At most 4 requests at a time per forecast, to stay polite to FIRMS, gridMET and AWS
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {name: pool.submit(job) for name, job in jobs.items()}
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as error:
                errors[name] = f"{type(error).__name__}: {error}"

    fire_errors = [errors[key] for key in ("fire_0", "fire_-1", "fire_-2") if key in errors]
    if fire_errors:
        raise ForecastUnavailable("Couldn't load NOAA-20 fire detections: " + "; ".join(fire_errors))

    today = results["fire_0"]
    layers = {
        "fire_t": fire_points.footprint_mask(grid, today, FIRE_FOOTPRINT_SCALE),
        "fire_t1": fire_points.footprint_mask(grid, results["fire_-1"], FIRE_FOOTPRINT_SCALE),
        "fire_t2": fire_points.footprint_mask(grid, results["fire_-2"], FIRE_FOOTPRINT_SCALE),
        "frp_t": _frp_layer(grid, today),
    }
    nan = np.full((grid.size, grid.size), np.nan, dtype=np.float32)
    layers["elevation"] = results.get("elevation", nan)
    layers["wind_u"], layers["wind_v"] = results.get("wind", (nan, nan))
    for var in ("tmmn", "tmmx", "sph", "pr", "erc"):
        layers[var] = results.get(var, nan)
    pdsi = results.get("pdsi")
    layers["pdsi"] = pdsi[0] if pdsi is not None else nan

    inputs = np.stack([np.asarray(layers[name], dtype=np.float32) for name in CHANNELS])
    info = {
        "fire_days": {name: day.isoformat() for name, day in zip(("t", "t-1", "t-2"), fire_days.values())},
        "detections": {name: int(len(results[f"fire_{offset}"])) for name, offset in (("t", 0), ("t-1", -1), ("t-2", -2))},
        "weather_days": [day.isoformat() for day in weather_days],
        "pdsi_pentad": pdsi[1].get("pentad_date") if pdsi is not None else None,
        "missing_layers": {name: message for name, message in errors.items()},
        "weather_note": weather_note,
    }
    return grid, inputs, info


def live_weather_days(fire_day, latest=None):
    """The training weather days (t-2, t-1) if gridMET has them, else its latest two days."""
    wanted = [fire_day + dt.timedelta(days=offset) for offset in WEATHER_OFFSETS]
    latest = latest or gridmet.latest_day()
    if latest >= wanted[-1]:
        return wanted
    return [latest - dt.timedelta(days=1), latest]


def forecast(lat, lon, fire_day=None):
    """The next-day spread forecast around (lat, lon), ready for the API (all lists north-up)."""
    fire_day = fire_day or latest_fire_day()
    grid = TileGrid.around(lat, lon)
    key = (grid.first_lat_index, grid.first_lon_index, fire_day)
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1]

    with _build_slots:
        grid, inputs, info = build_inputs(lat, lon, fire_day)
    model, stats, manifest = load_run(MODEL_DIR)
    channels = tuple(manifest["channels"])
    probability = predict(select_channels(inputs[None], channels), model, stats, channels)[0]
    threshold = float(manifest["threshold"])
    forecast_mask = probability >= threshold
    today = inputs[CHANNELS.index("fire_t")] == 1

    weather_lag = (fire_day - dt.date.fromisoformat(info["weather_days"][-1])).days
    notes = []
    if info["detections"]["t"] == 0:
        notes.append(f"NOAA-20 saw no fire in this area on {fire_day.isoformat()}, so the forecast starts from no fire.")
    if weather_lag > -WEATHER_OFFSETS[-1]:
        notes.append(f"gridMET weather is {weather_lag} days old (the model was trained on 1-day-old weather).")
    if info["weather_note"]:
        notes.append(info["weather_note"])
    if info["missing_layers"]:
        notes.append("Some inputs couldn't be loaded and were filled with training averages: " + ", ".join(info["missing_layers"]) + ".")

    # For a past day, what NOAA-20 actually saw on the forecast day, drawn the same way as the
    # training labels, and how this one forecast scored against it
    actual, check = None, None
    forecast_day = fire_day + dt.timedelta(days=1)
    if forecast_day <= latest_fire_day():
        try:
            burned = fire_points.footprint_mask(grid, _viirs_points(grid, forecast_day), FIRE_FOOTPRINT_SCALE) == 1
        except Exception as error:  # the forecast still stands without the check
            notes.append(f"Couldn't load what burned on {forecast_day.isoformat()} ({type(error).__name__}).")
        else:
            actual = grid.to_north_up(burned).astype(int).tolist()
            new = burned & ~today

            def overlap(guess):
                union = (guess | burned).sum()
                return round(float((guess & burned).sum() / union), 3) if union else None

            check = {
                "burned_pixels": int(burned.sum()),
                "new_fire_pixels": int(new.sum()),
                "new_fire_caught_pixels": int((forecast_mask & new).sum()),
                "model_iou": overlap(forecast_mask),
                "persistence_iou": overlap(today),
            }

    south, west, north, east = grid.bounds
    pixel_km2 = (grid.lat_step * 111.32) * (grid.lon_step * 111.32 * np.cos(np.radians((south + north) / 2)))
    north_up = grid.to_north_up(probability)
    # The clicked spot is pixel [CENTER_INDEX, CENTER_INDEX] in model orientation: find it north-up
    rows, cols = grid.cell_of([lat], [lon])
    spot_row, spot_col = grid.size - 1 - rows[0] if grid.row0_is_south else rows[0], cols[0]
    near = north_up[max(0, spot_row - NEAR_PX) : spot_row + NEAR_PX + 1, max(0, spot_col - NEAR_PX) : spot_col + NEAR_PX + 1]
    result = {
        "status": "ok",
        "fire_day": fire_day.isoformat(),
        "forecast_day": (fire_day + dt.timedelta(days=1)).isoformat(),
        "bounds": {"south": south, "west": west, "north": north, "east": east},
        "size": grid.size,
        # Rows run north to south, columns west to east, for drawing on a map
        "probability": np.round(grid.to_north_up(probability), 3).tolist(),
        # 1 where the model forecasts fire (probability >= threshold), decided before rounding, so
        # the drawn pixels always agree with forecast_pixels
        "likely": grid.to_north_up(forecast_mask).astype(int).tolist(),
        "fire_today": grid.to_north_up(today).astype(int).tolist(),
        # Past days only: what actually burned on forecast_day (north-up), and this forecast's scores
        "actual": actual,
        "check": check,
        "threshold": threshold,
        "forecast_pixels": int(forecast_mask.sum()),
        "fire_today_pixels": int(today.sum()),
        "new_fire_pixels": int((forecast_mask & ~today).sum()),
        # Summary numbers the dashboard's fire list uses (frontend/src/lib/live.js)
        "chance_near": round(float(near.max()), 3),
        "predicted_km2": round(float(forecast_mask.sum() * pixel_km2), 1),
        "fire_today_km2": round(float(today.sum() * pixel_km2), 1),
        "weather_days": info["weather_days"],
        "missing_layers": [],
        "pixel_km": {"north_south": round(grid.lat_step * 111.32, 3),
                     "east_west": round(grid.lon_step * 111.32 * np.cos(np.radians((south + north) / 2)), 3)},
        "inputs": info,
        "model": {"name": "spread_v2", "test_scores": TEST_SCORES},
        "notes": notes,
    }
    with _cache_lock:
        now = time.time()
        for stale in [k for k, (at, _) in _cache.items() if now - at > CACHE_SECONDS]:
            del _cache[stale]
        while len(_cache) >= CACHE_MAX_ENTRIES:
            del _cache[min(_cache, key=lambda k: _cache[k][0])]
        _cache[key] = (now, result)
    return result
