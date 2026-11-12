"""Elevation for each pixel of a spread-model tile, from AWS Terrain Tiles (Terrarium).

The training tiles took elevation from SRTM at Earth Engine's 1 km scale. Here each tile pixel
gets the mean of the zoom-10 Terrarium pixels (about 120 m at 38° N) whose centres fall inside
it. Negative values become 0: SRTM has no ocean values, while Terrarium at zoom 10 and below
has sea depths offshore. Clamping also turns Death Valley's -86 m into 0, which doesn't matter
after the model's scaling. See docs/spread_forecast_plan.md, "Elevation tiles (verified)".

Terrarium encoding (verified against Mount Whitney, Lake Tahoe and Badwater in the research
notes): elevation_m = R * 256 + G + B / 256 - 32768. Tiles are 256 x 256 XYZ Web Mercator PNGs
and need no key.
"""

import io
import math
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ZOOM = 10
TILE_PIXELS = 256
TIMEOUT_SECONDS = 20
RETRIES = 3  # attempts per tile
DOWNLOAD_THREADS = 8
# Decoded tiles kept in memory. Each is 256 x 256 float32 = 256 KB, so 256 tiles = 64 MB.
MEMORY_CACHE_TILES = 256
# Set to a Path to also keep the raw PNGs on disk (the backtest uses data/cache/terrarium)
CACHE_DIR = None

_cache = OrderedDict()
_cache_lock = threading.Lock()


class TerrainError(Exception):
    """Raised when elevation tiles can't be fetched or read."""


def decode(rgb):
    """Terrarium RGB (..., 3) to elevation in metres."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0


def global_pixel(lat, lon, zoom=ZOOM):
    """Fractional Web Mercator pixel coordinates (x east, y south) of points at a zoom level."""
    world = TILE_PIXELS * 2**zoom
    lat = np.clip(np.asarray(lat, dtype=float), -85.05112878, 85.05112878)
    x = (np.asarray(lon, dtype=float) + 180.0) / 360.0 * world
    sin_lat = np.sin(np.radians(lat))
    y = (0.5 - np.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * world
    return x, y


def pixel_center(x_index, y_index, zoom=ZOOM):
    """Latitude and longitude of the centre of integer Web Mercator pixels."""
    world = TILE_PIXELS * 2**zoom
    lon = (np.asarray(x_index, dtype=float) + 0.5) / world * 360.0 - 180.0
    mercator_y = math.pi * (1 - 2 * (np.asarray(y_index, dtype=float) + 0.5) / world)
    lat = np.degrees(np.arctan(np.sinh(mercator_y)))
    return lat, lon


def tile_of(lat, lon, zoom=ZOOM):
    """(x, y) of the tile containing a point."""
    x, y = global_pixel(lat, lon, zoom)
    return int(x // TILE_PIXELS), int(y // TILE_PIXELS)


def tiles_covering(bounds, zoom=ZOOM):
    """Every (x, y) tile that overlaps (south, west, north, east)."""
    south, west, north, east = bounds
    west_x, north_y = tile_of(north, west, zoom)
    east_x, south_y = tile_of(south, east, zoom)
    return [(x, y) for y in range(north_y, south_y + 1) for x in range(west_x, east_x + 1)]


def _download(zoom, x, y):
    """Raw PNG bytes for one tile, from the disk cache if CACHE_DIR is set, else the web."""
    path = Path(CACHE_DIR) / f"{zoom}_{x}_{y}.png" if CACHE_DIR else None
    if path is not None and path.exists():
        return path.read_bytes()
    url = TILE_URL.format(z=zoom, x=x, y=y)
    for attempt in range(RETRIES):
        try:
            response = requests.get(url, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as error:
            if attempt == RETRIES - 1:
                raise TerrainError(f"Could not fetch elevation tile {zoom}/{x}/{y}: {error}") from None
            time.sleep(2**attempt)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        partial.write_bytes(response.content)
        partial.replace(path)  # never leave a half-written PNG behind
    return response.content


def fetch_tile_rgb(zoom, x, y):
    """One Terrarium tile as uint8 RGB (256, 256, 3)."""
    try:
        with Image.open(io.BytesIO(_download(zoom, x, y))) as image:
            rgb = np.asarray(image.convert("RGB"))
    except (OSError, ValueError) as error:
        raise TerrainError(f"Elevation tile {zoom}/{x}/{y} is not a readable PNG: {error}") from None
    if rgb.shape != (TILE_PIXELS, TILE_PIXELS, 3):
        raise TerrainError(f"Elevation tile {zoom}/{x}/{y} has shape {rgb.shape}, expected 256 x 256 RGB")
    return rgb


def tile_elevation(zoom, x, y):
    """One tile decoded to metres, float32 (256, 256), cached in memory."""
    key = (zoom, x, y)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    elevation_m = decode(fetch_tile_rgb(zoom, x, y)).astype(np.float32)
    with _cache_lock:
        _cache[key] = elevation_m
        while len(_cache) > MEMORY_CACHE_TILES:
            _cache.popitem(last=False)
    return elevation_m


def clear_cache():
    with _cache_lock:
        _cache.clear()


def elevation(grid, zoom=ZOOM):
    """Elevation in metres for every pixel of a TileGrid: float32 (64, 64), model orientation.

    Each pixel is the mean of the Terrarium pixels whose centres fall inside it; negatives
    become 0. A pixel no Terrarium pixel falls in would be NaN (doesn't happen at zoom 10,
    where about 40 to 60 of them land in every 1 km pixel).
    """
    size = grid.size
    sums = np.zeros(size * size, dtype=np.float64)
    counts = np.zeros(size * size, dtype=np.int64)
    pixel_offsets = np.arange(TILE_PIXELS)
    tiles = tiles_covering(grid.bounds, zoom)  # each tile once
    # Download the tiles at the same time instead of one after another
    with ThreadPoolExecutor(max_workers=min(DOWNLOAD_THREADS, len(tiles))) as pool:
        decoded = list(pool.map(lambda tile: tile_elevation(zoom, *tile), tiles))
    for (x, y), values in zip(tiles, decoded):
        lat, _ = pixel_center(0, y * TILE_PIXELS + pixel_offsets, zoom)  # one latitude per tile row
        _, lon = pixel_center(x * TILE_PIXELS + pixel_offsets, 0, zoom)  # one longitude per tile column
        lats = np.repeat(lat[:, None], TILE_PIXELS, axis=1)
        lons = np.repeat(lon[None, :], TILE_PIXELS, axis=0)
        rows, cols = grid.cell_of(lats.ravel(), lons.ravel())
        inside = rows >= 0
        flat = rows[inside] * size + cols[inside]
        # Clamp every source pixel before averaging: SRTM, behind the training tiles, stores the sea
        # as 0, while Terrarium has sea depths. Clamping after averaging would let depths pull
        # coastline pixels down.
        land = np.maximum(values.ravel()[inside].astype(np.float64), 0.0)
        sums += np.bincount(flat, weights=land, minlength=size * size)
        counts += np.bincount(flat, minlength=size * size)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / counts, np.nan)
    return mean.reshape(size, size).astype(np.float32)
