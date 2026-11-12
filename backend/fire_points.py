"""Fire detections for one UTC day, and today's fire mask for a spread-model tile.

The spread model was trained with "today's fire" from MODIS Terra (MOD14A1 FireMask, one UTC
day, classes 7-9 = fire). FIRMS serves the same MODIS fire detections as points, so we use
Terra points from FIRMS and mark each tile pixel that contains one. When MODIS has no Terra
detection in the tile, or the request fails, VIIRS NOAA-20/NOAA-21 points are used instead and
the result says so. We can't see clouds, so there are no -1 (uncertain) pixels: only 1 and 0.
See docs/spread_forecast_plan.md, "How we rebuild each channel from live data".
"""

import os
import threading
import time
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd

from backend.firms_client import FirmsError, data_availability, fetch_area

# Satellites of each source family, and the FIRMS sources that carry them: the archive (SP)
# first, near real time (NRT) second. For a given day, the first source whose date range
# covers it is used. Suomi NPP is left out on purpose: NASA retired its fire data on 2026-11-02.
FAMILY_SOURCES = {
    "MODIS": [("MODIS_SP", "MODIS_NRT")],
    "VIIRS": [("VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT"), ("VIIRS_NOAA21_SP", "VIIRS_NOAA21_NRT")],
}
FAMILIES = tuple(FAMILY_SOURCES)

# FIRMS spells satellites differently by source: MODIS SP and NRT both send "Terra"/"Aqua"
# (seen 2026-11-11); single letters are accepted in case that changes. VIIRS sends
# "N" (Suomi NPP), "N20" and "N21".
SATELLITE_NAMES = {
    "terra": "Terra", "t": "Terra",
    "aqua": "Aqua", "a": "Aqua",
    "n": "SNPP", "npp": "SNPP", "snpp": "SNPP",
    "n20": "N20", "noaa-20": "N20", "noaa20": "N20", "j1": "N20",
    "n21": "N21", "noaa-21": "N21", "noaa21": "N21", "j2": "N21",
}
VIIRS_SATELLITES = ("N20", "N21")

POINT_COLUMNS = [
    "latitude", "longitude", "acq_date", "acq_time", "satellite",
    "confidence", "frp", "scan", "track", "daynight", "source",
]

# Extra margin around the tile when asking FIRMS, so detections on the tile's edge aren't lost
QUERY_PAD_DEG = 0.05
# Each detection marks a rectangle of FOOTPRINT_SCALE x its FIRMS pixel size (scan km east-west,
# track km north-south), not only the pixel holding its centre. MOD14A1, behind the training fire
# masks, grids the whole swath pixel, and MODIS pixels grow from 1 km at nadir to about 2 x 4.8 km
# at the swath edge. On 19 dated training tiles (demos/reconstruct_spread_tiles.py) agreement with
# the stored MOD14A1 masks (IoU) went from 0.34 with centre pixels to 0.63 with scale 0.5; scale 1
# over-marks (precision 0.60). 0 = centre pixel only.
FOOTPRINT_SCALE = 0.5
KM_PER_DEG_LAT = 111.32
# Near-real-time data for recent days keeps growing as passes are processed, so it's reused
# for a short time only. Archive (SP) data doesn't change.
NRT_CACHE_SECONDS = 10 * 60
# Set to a folder (Path) to keep archive (SP) answers on disk, e.g. for the backtest
CACHE_DIR = None

_cache = {}
_cache_lock = threading.Lock()


def _as_date(day):
    if isinstance(day, Date):
        return day
    return Date.fromisoformat(str(day))


def normalize_satellite(value):
    """Turn FIRMS's spelling of a satellite into "Terra", "Aqua", "N20", "N21" or "SNPP"."""
    text = str(value).strip()
    return SATELLITE_NAMES.get(text.lower(), text)


def sources_for(day, family="MODIS"):
    """The FIRMS sources that cover `day` for a family, one per satellite (archive preferred).

    Raises FirmsError when no source covers the day (for example, MODIS after it is retired).
    """
    if family not in FAMILY_SOURCES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    day = _as_date(day)
    chosen = _covering(data_availability(), day, family)
    if not chosen:
        # The cached date ranges may predate a new UTC day: NRT max_date moves forward daily
        chosen = _covering(data_availability(refresh=True), day, family)
    if not chosen:
        raise FirmsError(f"No FIRMS {family} source covers {day.isoformat()}.")
    return chosen


def _covering(availability, day, family):
    ranges = availability.set_index("data_id")
    chosen = []
    for candidates in FAMILY_SOURCES[family]:
        for source in candidates:
            if source in ranges.index:
                first, last = ranges.loc[source, "min_date"], ranges.loc[source, "max_date"]
                if pd.notna(first) and pd.notna(last) and first <= day <= last:
                    chosen.append(source)
                    break
    return chosen


def _box_text(south, west, north, east):
    # 4 decimals (about 10 m) keeps cache keys stable without moving the box noticeably
    return ",".join(f"{value:.4f}" for value in (west, south, east, north))


def _fetch_day(source, box, day):
    """One source, one box, one UTC day, with the in-memory (and optional on-disk) cache."""
    key = (source, box, day.isoformat())
    archive = source.endswith("_SP")
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and (archive or time.time() - hit[0] < NRT_CACHE_SECONDS):
            return hit[1]

    disk = None
    if archive and CACHE_DIR is not None:
        disk = Path(CACHE_DIR) / f"{source}_{box.replace(',', '_')}_{day.isoformat()}.csv"
        if disk.exists():
            table = pd.read_csv(disk)
            with _cache_lock:
                _cache[key] = (time.time(), table)
            return table

    table = fetch_area(source, area=box, days=1, date=day)
    if disk is not None:
        # Write to a side file and rename, so a concurrent reader never loads a half-written CSV
        disk.parent.mkdir(parents=True, exist_ok=True)
        partial = disk.with_name(f"{disk.name}.{os.getpid()}.{threading.get_ident()}.part")
        table.to_csv(partial, index=False)
        os.replace(partial, disk)
    with _cache_lock:
        _cache[key] = (time.time(), table)
    return table


def _tidy(table, source, day, south, west, north, east):
    """Keep one UTC day inside the box, in POINT_COLUMNS, with a normalized satellite name."""
    if table is None or len(table) == 0:
        return pd.DataFrame(columns=POINT_COLUMNS)
    out = pd.DataFrame({
        "latitude": pd.to_numeric(table["latitude"], errors="coerce"),
        "longitude": pd.to_numeric(table["longitude"], errors="coerce"),
        "acq_date": table["acq_date"].astype(str),
        "acq_time": pd.to_numeric(table["acq_time"], errors="coerce"),
        "satellite": table["satellite"].map(normalize_satellite),
        "confidence": table["confidence"],
        "frp": pd.to_numeric(table.get("frp"), errors="coerce") if "frp" in table else np.nan,
        "scan": pd.to_numeric(table.get("scan"), errors="coerce") if "scan" in table else np.nan,
        "track": pd.to_numeric(table.get("track"), errors="coerce") if "track" in table else np.nan,
        "daynight": table["daynight"] if "daynight" in table else None,
        "source": source,
    })
    keep = (
        (out["acq_date"] == day.isoformat())
        & out["latitude"].between(south, north)
        & out["longitude"].between(west, east)
        & out["acq_time"].notna()
    )
    out = out[keep].copy()
    out["acq_time"] = out["acq_time"].astype(int)
    return out.reset_index(drop=True)[POINT_COLUMNS]


def fire_points(south, west, north, east, day, family="MODIS"):
    """Detections for one UTC day inside a box, from every source of the family that covers it.

    Columns: POINT_COLUMNS. `satellite` is "Terra", "Aqua", "N20" or "N21".
    Raises FirmsError if no source covers the day or a request fails.
    """
    day = _as_date(day)
    box = _box_text(south, west, north, east)
    frames = [
        _tidy(_fetch_day(source, box, day), source, day, south, west, north, east)
        for source in sources_for(day, family)
    ]
    non_empty = [frame for frame in frames if len(frame)]
    if not non_empty:
        return pd.DataFrame(columns=POINT_COLUMNS)
    return pd.concat(non_empty, ignore_index=True)


def footprint_mask(grid, points, scale=FOOTPRINT_SCALE):
    """A 0/1 float32 mask marking each detection's footprint rectangle (and always its centre pixel).

    The rectangle is scale x scan km east-west by scale x track km north-south, centred on the
    detection. It is sampled every third of a pixel and rasterized with grid.rasterize, so a
    pixel is marked when a sample lands in it. Detections just outside the tile can still mark
    edge pixels; samples outside the tile are ignored.
    """
    lats = [points["latitude"].to_numpy(dtype=float)]
    lons = [points["longitude"].to_numpy(dtype=float)]
    if scale > 0 and len(points):
        has_size = points["scan"].notna() & points["track"].notna() if {"scan", "track"} <= set(points.columns) else None
        sized = points[has_size] if has_size is not None else points.iloc[0:0]
        for lat, lon, scan, track in zip(
            sized["latitude"].to_numpy(dtype=float), sized["longitude"].to_numpy(dtype=float),
            sized["scan"].to_numpy(dtype=float), sized["track"].to_numpy(dtype=float),
        ):
            half_ns = scale * track / 2 / KM_PER_DEG_LAT
            half_ew = scale * scan / 2 / (KM_PER_DEG_LAT * np.cos(np.radians(lat)))
            ys = np.arange(lat - half_ns, lat + half_ns + 1e-9, grid.lat_step / 3)
            xs = np.arange(lon - half_ew, lon + half_ew + 1e-9, grid.lon_step / 3)
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            lats.append(yy.ravel())
            lons.append(xx.ravel())
    return grid.rasterize(np.concatenate(lats), np.concatenate(lons))


def _points_on_day(points, day, satellites):
    """The given points that are on `day` and from `satellites` (anywhere in the queried box)."""
    if points is None or len(points) == 0:
        return points.iloc[0:0] if points is not None else pd.DataFrame(columns=POINT_COLUMNS)
    return points[
        (points["acq_date"].astype(str) == day.isoformat())
        & points["satellite"].map(normalize_satellite).isin(satellites)
    ]


def _points_in_tile(grid, points, day, satellites):
    """The given points that are on `day`, from `satellites`, and inside the tile."""
    if points is None or len(points) == 0:
        return points.iloc[0:0] if points is not None else pd.DataFrame(columns=POINT_COLUMNS)
    points = points[
        (points["acq_date"].astype(str) == day.isoformat())
        & points["satellite"].map(normalize_satellite).isin(satellites)
    ]
    rows, _ = grid.cell_of(points["latitude"].to_numpy(dtype=float), points["longitude"].to_numpy(dtype=float))
    return points[rows >= 0]


def fire_mask(
    grid, day, prefer="MODIS", satellites=("Terra",), viirs_satellites=VIIRS_SATELLITES, points=None,
    footprint_scale=FOOTPRINT_SCALE,
):
    """Today's fire for a tile: (float32 (64, 64) mask of 0 and 1, info dict).

    Uses the preferred family's detections from `satellites` (MODIS) or `viirs_satellites`
    (VIIRS). If there are none in the tile, or that request fails, falls back to the other
    family. `points` = {"MODIS": DataFrame, "VIIRS": DataFrame} uses those detections instead
    of asking FIRMS (a missing family is treated as unavailable). Each detection marks its
    footprint (see footprint_mask); footprint_scale=0 marks only the pixel holding its centre.
    """
    if prefer not in FAMILY_SOURCES:
        raise ValueError(f"prefer must be one of {FAMILIES}, got {prefer!r}")
    day = _as_date(day)
    south, west, north, east = grid.bounds
    pad = QUERY_PAD_DEG
    allowed = {"MODIS": tuple(satellites), "VIIRS": tuple(viirs_satellites)}
    order = [prefer] + [family for family in FAMILIES if family != prefer]

    errors, notes = [], []
    for position, family in enumerate(order):
        try:
            if points is not None:
                if family not in points:
                    notes.append(f"{family}: no detections were given")
                    continue
                table = points[family]
                firms_sources = sorted(set(table["source"])) if table is not None and "source" in table and len(table) else []
            else:
                table = fire_points(south - pad, west - pad, north + pad, east + pad, day, family)
                firms_sources = sources_for(day, family)
        except FirmsError as error:
            errors.append(f"{family}: {error}")
            continue
        inside = _points_in_tile(grid, table, day, allowed[family])
        if len(inside) == 0:
            notes.append(f"{family}: no detections from {', '.join(allowed[family])} in the tile")
            continue
        # Footprints of every detection that day in the queried box, so a fire just over the edge
        # still marks the edge pixels it covers
        mask = footprint_mask(grid, _points_on_day(table, day, allowed[family]), footprint_scale)
        info = {
            "source": family,
            "firms_sources": firms_sources,
            "satellites_used": sorted(set(inside["satellite"].map(normalize_satellite))),
            "day": day.isoformat(),
            "n_points": int(len(inside)),
            "n_fire_pixels": int(mask.sum()),
            "footprint_scale": footprint_scale,
            "fallback": position > 0,
            "reason": "; ".join(errors + notes) if position > 0 else None,
            "errors": errors,
        }
        return mask, info

    if errors and len(errors) == len(order):
        # Every request failed: we don't know today's fire, which is not the same as "no fire"
        raise FirmsError("Could not get today's fire detections. " + "; ".join(errors))
    mask = np.zeros((grid.size, grid.size), dtype=np.float32)
    return mask, {
        "source": prefer,
        "firms_sources": [],
        "satellites_used": [],
        "day": day.isoformat(),
        "n_points": 0,
        "n_fire_pixels": 0,
        "fallback": False,
        "reason": "no detections" + (f" ({'; '.join(errors + notes)})" if errors or notes else ""),
        "errors": errors,
    }
