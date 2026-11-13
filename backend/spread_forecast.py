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
from backend.firms_client import FirmsError
from backend.gridmet_bulk import Bulk
from backend.spread_grid import TileGrid
from backend.spread_model_v2 import (
    CHANNELS,
    FIRE_FOOTPRINT_SCALE,
    VIIRS_SATELLITES,
    WEATHER_OFFSETS,
    load_run,
    predict,
)

REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "models" / "spread_v2"

# NOAA-20 passes over the contiguous US around 08:30 and 20:30 UTC, and FIRMS NRT data arrives
# within about 3 hours, so a UTC day is complete by about 03:00 UTC the next day.
DAY_COMPLETE_HOUR_UTC = 4
CACHE_SECONDS = 30 * 60

# docs/spread_v2_test_results.txt, "all" rows (4,000 test tiles, 2024-2025)
TEST_SCORES = {
    "model": {"iou": 0.305, "precision": 0.459, "recall": 0.476, "new_fire_recall": 0.140, "new_fire_precision": 0.332},
    "persistence": {"iou": 0.239, "precision": 0.334, "recall": 0.456},
    "test_period": "2024-2025 fire seasons",
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


def latest_fire_day(now=None):
    """The latest complete UTC day of NOAA-20 detections."""
    now = now or dt.datetime.now(dt.timezone.utc)
    lag = 1 if now.hour >= DAY_COMPLETE_HOUR_UTC else 2
    return now.date() - dt.timedelta(days=lag)


def _viirs_points(grid, day):
    """NOAA-20 detections on `day` in the tile's box padded like the dataset builder."""
    south, west, north, east = grid.bounds
    pad = fire_points.QUERY_PAD_DEG
    table = fire_points.fire_points(south - pad, west - pad, north + pad, east + pad, day, family="VIIRS")
    return table[table["satellite"].isin(VIIRS_SATELLITES)]


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
    if weather_days is None:
        weather_days = live_weather_days(fire_day)

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

    grid, inputs, info = build_inputs(lat, lon, fire_day)
    model, stats, manifest = load_run(MODEL_DIR)
    probability = predict(inputs[None], model, stats, tuple(manifest["channels"]))[0]
    threshold = float(manifest["threshold"])
    forecast_mask = probability >= threshold
    today = inputs[CHANNELS.index("fire_t")] == 1

    weather_lag = (fire_day - dt.date.fromisoformat(info["weather_days"][-1])).days
    notes = []
    if info["detections"]["t"] == 0:
        notes.append(f"NOAA-20 saw no fire in this area on {fire_day.isoformat()}, so the forecast starts from no fire.")
    if weather_lag > -WEATHER_OFFSETS[-1]:
        notes.append(f"gridMET weather is {weather_lag} days old (the model was trained on 1-day-old weather).")
    if info["missing_layers"]:
        notes.append("Some inputs couldn't be loaded and were filled with training averages: " + ", ".join(info["missing_layers"]) + ".")

    south, west, north, east = grid.bounds
    result = {
        "fire_day": fire_day.isoformat(),
        "forecast_day": (fire_day + dt.timedelta(days=1)).isoformat(),
        "bounds": {"south": south, "west": west, "north": north, "east": east},
        "size": grid.size,
        # Rows run north to south, columns west to east, for drawing on a map
        "probability": np.round(grid.to_north_up(probability), 3).tolist(),
        "fire_today": grid.to_north_up(today).astype(int).tolist(),
        "threshold": threshold,
        "forecast_pixels": int(forecast_mask.sum()),
        "fire_today_pixels": int(today.sum()),
        "new_fire_pixels": int((forecast_mask & ~today).sum()),
        "pixel_km": {"north_south": round(grid.lat_step * 111.32, 3),
                     "east_west": round(grid.lon_step * 111.32 * np.cos(np.radians((south + north) / 2)), 3)},
        "inputs": info,
        "model": {"name": "spread_v2", "test_scores": TEST_SCORES},
        "notes": notes,
    }
    with _cache_lock:
        _cache[key] = (time.time(), result)
    return result
