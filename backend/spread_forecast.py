"""Run the trained next-day spread model on live data around one fire.

The model (notebooks/train_spread.ipynb) takes a 64 x 64 km area as a grid of
1 km squares with 12 layers, and returns the chance (0 to 1) that each square
is on fire tomorrow. This module builds those 12 layers from live sources:

  PrevFireMask          NASA FIRMS detections already loaded by the dashboard
                        (VIIRS NOAA-20; training used MODIS, which is coarser)
  vs th tmmn tmmx sph   GRIDMET, latest day, the same source as the training data
  pr erc pdsi           GRIDMET (pdsi updates every 5 days)
  elevation             AWS public terrain tiles
  NDVI population       not filled: they need a NASA Earthdata login. Missing
                        values become the training average, exactly as the
                        training code does for its own missing pixels.

Input scaling copies demos/spread_examples.py. The weights and scaling stats
come from Kaggle (see README, Model Training) and go in data/training/.
"""

import math
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[1]
TRAINING = REPO / "data" / "training"
CACHE_DIR = REPO / "data" / "cache"
WEIGHTS = TRAINING / "spread_unet.pt"
STATS = TRAINING / "spread_feature_stats.csv"
THRESHOLD = 0.621  # chosen on val by the Kaggle run, same as demos/spread_examples.py
GRID = 64
HALF_KM = GRID / 2
NEAR_KM = 2  # "chance near this fire" = highest chance within this distance
THREDDS = "http://thredds.northwestknowledge.net:8080/thredds/ncss"
GRIDMET = {
    "vs": ("agg_met_vs_1979_CurrentYear_CONUS.nc", "daily_mean_wind_speed"),
    "th": ("agg_met_th_1979_CurrentYear_CONUS.nc", "daily_mean_wind_direction"),
    "tmmn": ("agg_met_tmmn_1979_CurrentYear_CONUS.nc", "daily_minimum_temperature"),
    "tmmx": ("agg_met_tmmx_1979_CurrentYear_CONUS.nc", "daily_maximum_temperature"),
    "sph": ("agg_met_sph_1979_CurrentYear_CONUS.nc", "daily_mean_specific_humidity"),
    "pr": ("agg_met_pr_1979_CurrentYear_CONUS.nc", "precipitation_amount"),
    "erc": ("agg_met_erc_1979_CurrentYear_CONUS.nc", "daily_mean_energy_release_component-g"),
    "pdsi": ("agg_met_pdsi_1979_CurrentYear_CONUS.nc", "daily_mean_palmer_drought_severity_index"),
}
MISSING_LAYERS = ("NDVI", "population")
CACHE_SECONDS = 6 * 60 * 60  # GRIDMET is daily, so a forecast stays valid for hours
OUTAGE_SECONDS = 2 * 60  # after a source fails, answer "unavailable" at once for a while instead of waiting again

_model = {"loaded": False, "net": None, "features": None, "ranges": None, "stats": None, "error": None}
_model_lock = threading.Lock()
_cache, _cache_lock = {}, threading.Lock()
_outage = {"until": 0.0, "note": ""}


# ---------- the model ----------

def _notebook_code():
    """The model and feature list straight from train_spread.ipynb, as the demo script does."""
    import nbformat
    import torch

    notebook = nbformat.read(REPO / "notebooks" / "train_spread.ipynb", as_version=4)
    cells = {cell.get("id"): cell.source for cell in notebook.cells}
    namespace = {"torch": torch}
    exec(cells["b2e0d003"].replace('print("Running on"', "#"), namespace)
    exec(cells["b2e0d020"].split("parameters =")[0], namespace)
    return namespace


def model_status():
    """Whether the model can run, and if not, what's missing."""
    missing = [str(p.relative_to(REPO)) for p in (WEIGHTS, STATS) if not p.exists()]
    if missing:
        return {"ready": False, "missing": missing, "note": "Download the weights and stats from Kaggle into data/training (README, Model Training)."}
    _load_model()
    if _model["error"]:
        return {"ready": False, "missing": [], "note": _model["error"]}
    return {"ready": True, "threshold": THRESHOLD, "missing_layers": list(MISSING_LAYERS)}


def _load_model():
    with _model_lock:
        if _model["loaded"]:
            return
        _model["loaded"] = True
        try:
            import pandas as pd
            import torch

            code = _notebook_code()
            features = list(code["INPUT_FEATURES"])
            net = code["UNet"](len(features))
            net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
            net.eval()
            _model.update(net=net, features=features, ranges=code["VALID_RANGES"],
                          stats=pd.read_csv(STATS, index_col="feature"))
        except Exception as err:  # wrong file, changed notebook: say so instead of crashing the API
            _model["error"] = f"The spread model couldn't load: {err}"


def _normalize(inputs):
    """Copy of demos/spread_examples.py normalize(): clip, scale, missing -> training average (0)."""
    inputs = inputs.copy()
    features, stats = _model["features"], _model["stats"]
    for name, (low, high) in _model["ranges"].items():
        plane = inputs[:, features.index(name)]
        plane[(plane < low) | (plane > high)] = np.nan
    for channel, name in enumerate(features):
        low, high, mean, std = stats.loc[name, ["low", "high", "mean", "std"]]
        plane = inputs[:, channel]
        np.clip(plane, low, high, out=plane)
        plane -= mean
        plane /= std
        plane[np.isnan(plane)] = 0.0
    return inputs


# ---------- the 12 live layers ----------

def _box(lat, lon):
    dlat = HALF_KM / 111.32
    dlon = HALF_KM / (111.32 * math.cos(math.radians(lat)))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def _centres(lat, lon):
    s, n, w, e = _box(lat, lon)
    lats = n - (np.arange(GRID) + 0.5) * (n - s) / GRID  # row 0 is the north edge, like an image
    lons = w + (np.arange(GRID) + 0.5) * (e - w) / GRID
    return lats, lons, (s, n, w, e)


def _gridmet(stem, var, box):
    """Latest day of one GRIDMET variable, cut to the box (plus a margin), with its date."""
    import netCDF4

    s, n, w, e = box
    meta = requests.get(f"{THREDDS}/{stem}/dataset.xml", timeout=(10, 30)).text
    day = meta.split("<end>")[1].split("<")[0][:10]
    url = (f"{THREDDS}/{stem}?var={var}&north={n + 0.1}&west={w - 0.1}&east={e + 0.1}&south={s - 0.1}"
           f"&horizStride=1&time_start={day}T00:00:00Z&time_end={day}T00:00:00Z&accept=netcdf")
    r = requests.get(url, timeout=(10, 60))
    r.raise_for_status()
    # Read from a real file: opening this netCDF-3 file from memory makes HDF5 print pages of harmless errors
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        tmp.write(r.content)
        tmp.flush()
        with netCDF4.Dataset(tmp.name) as ds:
            values = np.ma.filled(ds[var][:].astype(np.float32), np.nan).squeeze()
            return np.asarray(ds["lat"][:]), np.asarray(ds["lon"][:]), values, day


def _nearest(grid_lats, grid_lons, values, lats, lons):
    """Nearest GRIDMET cell (about 4 km) for each 1 km square."""
    rows = np.abs(grid_lats[:, None] - lats[None, :]).argmin(0)
    cols = np.abs(grid_lons[:, None] - lons[None, :]).argmin(0)
    return values[np.ix_(rows, cols)]


_tile_lock = threading.Lock()


def _elevation(lats, lons, zoom=9, size=512):
    """Nearest pixel of the AWS terrain tiles (meters), cached on disk."""
    import tifffile

    la, lo = np.meshgrid(lats, lons, indexing="ij")
    count = 2 ** zoom
    fx = (lo + 180) / 360 * count
    fy = (1 - np.log(np.tan(np.radians(la)) + 1 / np.cos(np.radians(la))) / math.pi) / 2 * count
    tx, ty = fx.astype(int), fy.astype(int)
    px = ((fx - tx) * size).astype(int).clip(0, size - 1)
    py = ((fy - ty) * size).astype(int).clip(0, size - 1)
    out = np.empty(la.shape, np.float32)
    for x, y in set(zip(tx.ravel().tolist(), ty.ravel().tolist())):
        path = CACHE_DIR / "elevation" / f"{zoom}_{x}_{y}.tif"
        with _tile_lock:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                r = requests.get(f"https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{zoom}/{x}/{y}.tif", timeout=120)
                r.raise_for_status()
                path.write_bytes(r.content)
        tile = tifffile.imread(path)
        m = (tx == x) & (ty == y)
        out[m] = tile[py[m], px[m]]
    return out


def _fire_mask(fires, box):
    """1 where a FIRMS detection from the newest day falls in the square, else 0."""
    s, n, w, e = box
    mask = np.zeros((GRID, GRID), np.float32)
    if fires is None or not len(fires):
        return mask, None
    newest = fires["acq_date"].max()
    today = fires[(fires["acq_date"] == newest) & fires["latitude"].between(s, n) & fires["longitude"].between(w, e)]
    rows = ((n - today["latitude"]) / (n - s) * GRID).astype(int).clip(0, GRID - 1)
    cols = ((today["longitude"] - w) / (e - w) * GRID).astype(int).clip(0, GRID - 1)
    mask[rows.to_numpy(), cols.to_numpy()] = 1
    return mask, newest


def build_layers(lat, lon, fires):
    """The 12 layers around (lat, lon), raw units, in the model's channel order."""
    lats, lons, box = _centres(lat, lon)
    with ThreadPoolExecutor(max_workers=len(GRIDMET) + 1) as pool:
        jobs = {name: pool.submit(_gridmet, stem, var, box) for name, (stem, var) in GRIDMET.items()}
        elev_job = pool.submit(_elevation, lats, lons)
    layers, days = {}, {}
    for name, job in jobs.items():
        g_lats, g_lons, values, day = job.result()
        layers[name] = _nearest(g_lats, g_lons, values, lats, lons)
        days[name] = day
    layers["elevation"] = elev_job.result()
    layers["PrevFireMask"], fire_day = _fire_mask(fires, box)
    for name in MISSING_LAYERS:
        layers[name] = np.full((GRID, GRID), np.nan, np.float32)
    return layers, lats, lons, box, days, fire_day


# ---------- one forecast ----------

def forecast(lat, lon, fires):
    """Tomorrow's fire chance around one spot, or why it can't be made."""
    status = model_status()
    if not status["ready"]:
        return {"status": "not_ready", **status}
    key = (round(lat, 2), round(lon, 2), date.today().isoformat())
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1]
    import torch

    if time.time() < _outage["until"]:
        return {"status": "inputs_unavailable", "note": _outage["note"]}
    try:
        layers, lats, lons, box, days, fire_day = build_layers(lat, lon, fires)
    except Exception as err:  # a data source is down: say which, don't fail the whole API
        note = f"Couldn't get today's inputs for the model: {type(err).__name__}. GRIDMET (weather) is the usual cause; try again later."
        _outage.update(until=time.time() + OUTAGE_SECONDS, note=note)
        return {"status": "inputs_unavailable", "note": note}
    stack = np.stack([layers[name] for name in _model["features"]])[None].astype(np.float32)  # 1 x 12 x 64 x 64
    with torch.no_grad():
        prob = torch.sigmoid(_model["net"](torch.from_numpy(_normalize(stack)))).numpy()[0, 0]
    # the highest chance within NEAR_KM of the spot itself
    r0 = c0 = GRID // 2
    near = prob[max(0, r0 - NEAR_KM):r0 + NEAR_KM + 1, max(0, c0 - NEAR_KM):c0 + NEAR_KM + 1]
    s, n, w, e = box
    result = {
        "status": "ok",
        "bounds": {"south": s, "north": n, "west": w, "east": e},
        "probability": np.round(prob, 3).tolist(),  # 64 rows, north to south; 64 columns, west to east
        "threshold": THRESHOLD,
        "chance_near": round(float(near.max()), 3),
        "predicted_km2": int((prob > THRESHOLD).sum()),
        "fire_today_km2": int(layers["PrevFireMask"].sum()),
        "fire_day": fire_day,
        "weather_days": days,
        "missing_layers": list(MISSING_LAYERS),
    }
    with _cache_lock:
        _cache[key] = (time.time(), result)
    return result
