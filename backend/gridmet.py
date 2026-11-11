"""gridMET weather and drought layers for spread-model tiles.

The spread model was trained on Next Day Wildfire Spread, whose weather channels come from
gridMET (IDAHO_EPSCOR/GRIDMET) and whose drought channel comes from gridMET's PDSI
(GRIDMET/DROUGHT). This module reads the same data live from the University of Idaho THREDDS
server and resamples it onto a TileGrid the way the training export did.
See docs/spread_forecast_plan.md, "gridMET facts (verified)" and "Bicubic".

Facts this code relies on (verified against live responses on 2026-11-11):
  - Yearly files MET/{var}/{var}_{YYYY}.nc, PDSI in MET/pdsi/pdsi.nc. OPeNDAP .ascii requests
    with inclusive index ranges: name[d0:1:d1][i0:1:i1][j0:1:j1].
  - Arrays are [day][lat][lon] (the DDS order; the "dimensions" attribute is wrong), latitude
    descending, longitude ascending, 1/24 degree.
  - Values come back as packed UInt16: physical = raw * scale_factor + add_offset, and the
    _FillValue / missing_value (32767) means missing. tmmn and tmmx share the variable name
    air_temperature but have different offsets (210 and 220), so scale and offset are always
    read from each file's .das, never looked up by name.
  - "day" is days since 1900-01-01. The newest day is the last value of the file's day array.
"""

import datetime as dt
import re
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

# EXTERNAL API: gridMET on the University of Idaho THREDDS servers. Free, no key. Tried in order.
BASES = (
    "https://www.reacchpna.org/thredds/dodsC/MET",
    "https://thredds.northwestknowledge.net/thredds/dodsC/MET",
    "https://tds-proxy.nkn.uidaho.edu/thredds/dodsC/MET",
)

# The seven gridMET channels of the spread model, by their Earth Engine band names
VARIABLES = ("th", "vs", "tmmn", "tmmx", "sph", "pr", "erc")
PDSI = "pdsi"

EPOCH = dt.date(1900, 1, 1)  # gridMET "day" is days since this date

# Earth Engine's metres per degree at the equator for EPSG:4326. The training export reprojected
# gridMET onto a 20 km grid (`reproject(...atScale(20000))`) before bicubic resampling, which on
# the WGS84 degree grid means square cells of this many degrees, anchored at (0°, 0°).
NODE_DEG = 20000 / 111319.49079327357

TIMEOUT = 30  # seconds per request
RETRIES = 2  # extra tries per server for timeouts, connection errors and 5xx answers
META_TTL = 3600  # file metadata (the day array grows daily) is re-read after this many seconds
DATA_TTL = 6 * 3600  # recent days can be revised ("early" and "provisional" slices)

# Set to a Path to keep fetched subsets of past years on disk (the backtest uses data/cache/gridmet).
CACHE_DIR = None

_SEPARATOR = "-" * 45
_HEADER = re.compile(r"^([A-Za-z_][\w.\-%]*)((?:\[\d+\])+)\s*$")


class GridmetError(Exception):
    """Raised when gridMET data can't be fetched or doesn't cover the request."""


class _NotFound(GridmetError):
    """The server answered 404: the file doesn't exist (for example next year's file)."""


@dataclass(frozen=True)
class _FileMeta:
    path: str  # e.g. "tmmx/tmmx_2026.nc"
    name: str  # variable name inside the file, e.g. "air_temperature"
    scale: float
    offset: float
    fills: tuple  # raw values that mean missing
    lats: np.ndarray  # as stored: descending
    lons: np.ndarray  # ascending
    days: tuple  # dt.date per day index


_cache_lock = threading.Lock()
_meta_cache = {}  # path -> (time fetched, _FileMeta)
_data_cache = {}  # (path, d0, d1, i0, i1, j0, j1) -> (time fetched, raw array)


# ----------------------------------------------------------------------------- HTTP and parsing


def _fetch_text(path_and_query):
    """GET {base}/{path_and_query} from the first server that answers. Returns (text, base).

    Raises _NotFound on 404 (the file doesn't exist; asking the mirrors won't change that) and
    GridmetError if every server fails.
    """
    problems = []
    for base in BASES:
        for attempt in range(RETRIES + 1):
            try:
                response = requests.get(f"{base}/{path_and_query}", timeout=TIMEOUT)
            except requests.exceptions.RequestException as error:
                problems.append(f"{base}: {type(error).__name__}")
                time.sleep(0.5 * 2**attempt)
                continue
            if response.status_code == 404:
                raise _NotFound(f"gridMET has no {path_and_query.split('.nc')[0]}.nc (HTTP 404)")
            if response.status_code >= 500:
                problems.append(f"{base}: HTTP {response.status_code}")
                time.sleep(0.5 * 2**attempt)
                continue
            if response.status_code != 200:
                raise GridmetError(f"gridMET returned HTTP {response.status_code}: {response.text[:200]}")
            return response.text, base
    raise GridmetError("Every gridMET server failed. " + "; ".join(problems[-3:]))


def parse_ascii(text):
    """Parse an OPeNDAP DAP2 .ascii response into {printed name: ndarray}.

    Grid answers print "name.name[2][3][4]" then rows like "[0][1], v, v, v", then their map
    vectors ("name.day[2]", "name.lat[3]", ...). Plain arrays print "lat[3]" then one line of values.
    """
    if _SEPARATOR not in text:
        raise GridmetError("gridMET did not return OPeNDAP data: " + text[:200])
    body = text.split(_SEPARATOR, 1)[1]
    arrays = {}
    name, shape, values = None, None, []

    def finish():
        if name is None:
            return
        array = np.array(values, dtype=np.float64)
        if array.size != int(np.prod(shape)):
            raise GridmetError(f"gridMET answer for {name} has {array.size} values, expected shape {shape}")
        arrays[name] = array.reshape(shape)

    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        header = _HEADER.match(line)
        if header:
            finish()
            name = header.group(1)
            shape = tuple(int(size) for size in re.findall(r"\[(\d+)\]", header.group(2)))
            values = []
            continue
        if line.startswith("["):  # drop the "[0][1]" row label
            line = line.split(",", 1)[1] if "," in line else ""
        values.extend(float(value) for value in line.split(",") if value.strip())
    finish()
    return arrays


def parse_das(text, name):
    """scale_factor, add_offset and fill values of one variable, from a .das answer."""
    block = re.search(r"^\s*" + re.escape(name) + r"\s*\{(.*?)^\s*\}", text, re.S | re.M)
    if not block:
        raise GridmetError(f"gridMET .das has no attributes for {name}")
    attributes = {}
    for key, value in re.findall(r"^\s*\w+\s+(\w+)\s+(.+?);\s*$", block.group(1), re.M):
        attributes[key] = value.strip().strip('"')
    try:
        scale = float(attributes.get("scale_factor", 1.0))
        offset = float(attributes.get("add_offset", 0.0))
    except ValueError:
        raise GridmetError(f"gridMET .das has an unreadable scale or offset for {name}") from None
    fills = tuple(sorted({float(attributes[key]) for key in ("_FillValue", "missing_value") if key in attributes}))
    return scale, offset, fills


def parse_dds_grids(text):
    """{variable name: (days, lats, lons)} of every [day][lat][lon] grid in a .dds answer."""
    grids = {}
    for name, days, lats, lons in re.findall(
        r"ARRAY:\s*\w+\s+([\w\-]+)\[day = (\d+)\]\[lat = (\d+)\]\[lon = (\d+)\];", text
    ):
        grids[name] = (int(days), int(lats), int(lons))
    if not grids:
        raise GridmetError("gridMET .dds has no [day][lat][lon] grid; the file layout has changed")
    return grids


def _to_date(days_since_1900):
    return EPOCH + dt.timedelta(days=int(round(days_since_1900)))


def _file_path(var, year):
    if var == PDSI:
        return "pdsi/pdsi.nc"
    if var not in VARIABLES:
        raise GridmetError(f"Unknown gridMET variable {var!r}. Use one of {VARIABLES + (PDSI,)}")
    return f"{var}/{var}_{year}.nc"


def _disk_cache_allowed(path):
    """Only past years' files are cached on disk: recent days are still being revised."""
    match = re.search(r"_(\d{4})\.nc$", path)
    return CACHE_DIR is not None and match is not None and int(match.group(1)) < dt.datetime.now(dt.timezone.utc).year


def _meta(path):
    """Variable name, packing and coordinates of one file, read once and cached."""
    with _cache_lock:
        hit = _meta_cache.get(path)
        if hit and time.time() - hit[0] < META_TTL:
            return hit[1]

    disk = Path(CACHE_DIR) / f"meta_{path.replace('/', '_')}.npz" if _disk_cache_allowed(path) else None
    if disk is not None and disk.exists():
        stored = np.load(disk, allow_pickle=False)
        meta = _FileMeta(
            path=path,
            name=str(stored["name"]),
            scale=float(stored["scale"]),
            offset=float(stored["offset"]),
            fills=tuple(stored["fills"].tolist()),
            lats=stored["lats"],
            lons=stored["lons"],
            days=tuple(_to_date(value) for value in stored["days"]),
        )
    else:
        grids = parse_dds_grids(_fetch_text(f"{path}.dds")[0])
        # The PDSI file also holds a "category" grid; the yearly files hold exactly one grid
        names = [PDSI] if path.startswith("pdsi/") else list(grids)
        if len(names) != 1 or names[0] not in grids:
            raise GridmetError(f"Expected one data grid in {path}, found {list(grids)}")
        name = names[0]
        scale, offset, fills = parse_das(_fetch_text(f"{path}.das")[0], name)
        coordinates = parse_ascii(_fetch_text(f"{path}.ascii?lat,lon,day")[0])
        lats, lons, days = coordinates["lat"], coordinates["lon"], coordinates["day"]
        expected = grids[name]
        if (len(days), len(lats), len(lons)) != expected:
            raise GridmetError(f"{path}: coordinate lengths don't match the grid {expected}")
        if not (np.all(np.diff(lats) < 0) and np.all(np.diff(lons) > 0)):
            raise GridmetError(f"{path}: expected latitude descending and longitude ascending")
        meta = _FileMeta(path, name, scale, offset, fills, lats, lons, tuple(_to_date(value) for value in days))
        if disk is not None:
            disk.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                disk, name=name, scale=scale, offset=offset, fills=np.array(fills, dtype=float),
                lats=lats, lons=lons, days=days,
            )

    with _cache_lock:
        _meta_cache[path] = (time.time(), meta)
    return meta


def _raw_subset(meta, d0, d1, i0, i1, j0, j1):
    """Raw packed values meta.name[d0:d1][i0:i1][j0:j1] (inclusive), as stored (lat descending)."""
    key = (meta.path, d0, d1, i0, i1, j0, j1)
    with _cache_lock:
        hit = _data_cache.get(key)
        if hit and time.time() - hit[0] < DATA_TTL:
            return hit[1]

    disk = None
    if _disk_cache_allowed(meta.path):
        disk = Path(CACHE_DIR) / f"{meta.path.replace('/', '_')}_{d0}_{d1}_{i0}_{i1}_{j0}_{j1}.npz"
    if disk is not None and disk.exists():
        raw = np.load(disk)["raw"]
    else:
        query = f"{meta.name}[{d0}:1:{d1}][{i0}:1:{i1}][{j0}:1:{j1}]"
        arrays = parse_ascii(_fetch_text(f"{meta.path}.ascii?{query}")[0])
        raw = arrays.get(f"{meta.name}.{meta.name}")
        if raw is None:
            raise GridmetError(f"gridMET answer for {meta.path} has no {meta.name} values")
        expected = (d1 - d0 + 1, i1 - i0 + 1, j1 - j0 + 1)
        if raw.shape != expected:
            raise GridmetError(f"gridMET answer for {meta.path} has shape {raw.shape}, expected {expected}")
        if disk is not None:
            disk.parent.mkdir(parents=True, exist_ok=True)
            np.savez(disk, raw=raw)

    with _cache_lock:
        _data_cache[key] = (time.time(), raw)
    return raw


def _unpack(raw, meta):
    """Packed integers to physical units, with NaN for the fill value."""
    values = raw.astype(np.float64) * meta.scale + meta.offset
    values[np.isin(raw, meta.fills)] = np.nan
    return values.astype(np.float32)


# ----------------------------------------------------------------------------- public: raw data


def fetch(var, days, south, west, north, east):
    """gridMET values for each day in `days` over a lat/lon box.

    var is one of VARIABLES, or "pdsi" (then each day must be a PDSI pentad date).
    Returns (values[len(days), nlat, nlon] float32 in physical units with NaN for missing,
    lats ascending[nlat], lons ascending[nlon]). Every grid cell whose centre is within one cell
    of the box is included. Raises GridmetError if a day isn't in the files (yet).
    """
    days = list(days)
    if not days:
        raise GridmetError("fetch needs at least one day")
    if not (south < north and west < east):
        raise GridmetError(f"bad box: south={south} north={north} west={west} east={east}")

    by_path = {}
    for position, day in enumerate(days):
        by_path.setdefault(_file_path(var, day.year), []).append((position, day))

    result, lats, lons = None, None, None
    for path, wanted in by_path.items():
        try:
            meta = _meta(path)
        except _NotFound:
            raise GridmetError(
                f"gridMET {var} has no file for {wanted[0][1].year} yet, so {wanted[0][1]} isn't available"
            ) from None
        index_of = {day: i for i, day in enumerate(meta.days)}
        missing = [day for _, day in wanted if day not in index_of]
        if missing:
            raise GridmetError(
                f"gridMET {var} has no data for {missing[0].isoformat()} (its latest day is {meta.days[-1].isoformat()})"
                if var != PDSI
                else f"{missing[0].isoformat()} is not a PDSI pentad date"
            )

        step_lat = abs(meta.lats[1] - meta.lats[0])
        step_lon = abs(meta.lons[1] - meta.lons[0])
        lat_rows = np.nonzero((meta.lats >= south - step_lat) & (meta.lats <= north + step_lat))[0]
        lon_cols = np.nonzero((meta.lons >= west - step_lon) & (meta.lons <= east + step_lon))[0]
        if not len(lat_rows) or not len(lon_cols):
            raise GridmetError(f"The box ({south}, {west}, {north}, {east}) is outside gridMET (contiguous US only)")
        i0, i1 = int(lat_rows[0]), int(lat_rows[-1])
        j0, j1 = int(lon_cols[0]), int(lon_cols[-1])

        day_indices = [index_of[day] for _, day in wanted]
        d0, d1 = min(day_indices), max(day_indices)
        raw = _raw_subset(meta, d0, d1, i0, i1, j0, j1)
        values = _unpack(raw, meta)[[index - d0 for index in day_indices]]
        # Stored latitude descends; flip so it ascends (south to north)
        values = values[:, ::-1, :]
        file_lats = meta.lats[i0 : i1 + 1][::-1]
        file_lons = meta.lons[j0 : j1 + 1]
        if result is None:
            result = np.full((len(days), len(file_lats), len(file_lons)), np.nan, dtype=np.float32)
            lats, lons = file_lats, file_lons
        elif not (np.allclose(lats, file_lats) and np.allclose(lons, file_lons)):
            raise GridmetError(f"{path} uses a different grid from the other years in this request")
        for (position, _), plane in zip(wanted, values):
            result[position] = plane
    return result, lats.astype(np.float64), lons.astype(np.float64)


def available_days(var, year):
    """Every day in {var}_{year}.nc (or every pentad date in the PDSI file), oldest first."""
    return list(_meta(_file_path(var, year)).days)


def latest_day(var="erc", today=None):
    """The newest day gridMET has for `var`. Falls back to last year's file in early January."""
    if var == PDSI:
        return _meta(_file_path(PDSI, 0)).days[-1]
    today = today or dt.datetime.now(dt.timezone.utc).date()
    for year in (today.year, today.year - 1):
        try:
            return _meta(_file_path(var, year)).days[-1]
        except _NotFound:
            continue
    raise GridmetError(f"gridMET has no {var} file for {today.year} or {today.year - 1}")


# ----------------------------------------------------------------------------- resampling


def keys_weights(t, a=-0.5):
    """Weights of the 4 taps at offsets -1, 0, 1, 2 for a point a fraction t (0 <= t < 1) past tap 0."""
    t = np.asarray(t, dtype=np.float64)
    distances = np.stack([1 + t, t, 1 - t, 2 - t], axis=-1)
    near = (a + 2) * distances**3 - (a + 3) * distances**2 + 1  # |x| <= 1
    far = a * distances**3 - 5 * a * distances**2 + 8 * a * distances - 4 * a  # 1 < |x| < 2
    weights = np.where(distances <= 1, near, np.where(distances < 2, far, 0.0))
    return weights


def _axis_taps(coordinates, out):
    """Tap indices (n, 4) and weights (n, 4) along one regular ascending axis."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    steps = np.diff(coordinates)
    if len(coordinates) < 2 or not np.all(steps > 0) or not np.allclose(steps, steps[0], rtol=1e-6, atol=0):
        raise GridmetError("bicubic needs a regular, ascending source grid")
    position = (np.asarray(out, dtype=np.float64) - coordinates[0]) / steps[0]
    base = np.floor(position)
    # Snap points within rounding error of a node onto it, so nodes are reproduced exactly
    near_node = np.isclose(position, np.round(position), rtol=0, atol=1e-9)
    base = np.where(near_node, np.round(position), base)
    fraction = np.where(near_node, 0.0, position - base)
    taps = base.astype(np.int64)[:, None] + np.array([-1, 0, 1, 2])
    return taps, keys_weights(fraction)


def bicubic(values, lats, lons, out_lats, out_lons):
    """Keys (a = -0.5) bicubic interpolation of a regular grid at arbitrary points.

    values[nlat, nlon] with lats and lons ascending. out_lats and out_lons have any shape (the
    same shape); the result has that shape. Taps that are NaN or off the grid are skipped and the
    remaining weights renormalized; if the usable weight is below 0.5 the output is NaN.
    """
    values = np.asarray(values, dtype=np.float64)
    out_lats = np.asarray(out_lats, dtype=np.float64)
    out_lons = np.asarray(out_lons, dtype=np.float64)
    if values.shape != (len(lats), len(lons)):
        raise GridmetError(f"values {values.shape} don't match lats ({len(lats)}) and lons ({len(lons)})")
    shape = out_lats.shape
    row_taps, row_weights = _axis_taps(lats, out_lats.ravel())
    col_taps, col_weights = _axis_taps(lons, out_lons.ravel())

    rows_ok = (row_taps >= 0) & (row_taps < len(lats))
    cols_ok = (col_taps >= 0) & (col_taps < len(lons))
    tap_values = values[np.clip(row_taps, 0, len(lats) - 1)[:, :, None], np.clip(col_taps, 0, len(lons) - 1)[:, None, :]]
    usable = rows_ok[:, :, None] & cols_ok[:, None, :] & np.isfinite(tap_values)
    weights = row_weights[:, :, None] * col_weights[:, None, :]
    weight_sum = np.where(usable, weights, 0.0).sum(axis=(1, 2))
    total = np.where(usable, weights * np.where(usable, tap_values, 0.0), 0.0).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(weight_sum >= 0.5, total / weight_sum, np.nan)
    return result.reshape(shape)


def _node_range(low, high):
    """Node indices whose 4-tap support covers every point in [low, high] (node k centred at (k+0.5)*NODE_DEG)."""
    first = int(np.floor(low / NODE_DEG - 0.5)) - 1
    last = int(np.floor(high / NODE_DEG - 0.5)) + 2
    return first, last


def node_grid(values, lats, lons, lat_nodes, lon_nodes):
    """Block means of a gridMET field on the 20 km node grid.

    values[nlat, nlon] with lats/lons ascending (cell centres). lat_nodes and lon_nodes are
    inclusive (first, last) node index ranges; node k covers [k*NODE_DEG, (k+1)*NODE_DEG).
    Each node is the mean of the valid cells whose centres fall inside it (NaN if none).
    Returns (node values, node centre lats, node centre lons), all ascending.
    """
    values = np.asarray(values, dtype=np.float64)
    lat_index = np.floor(np.asarray(lats) / NODE_DEG).astype(np.int64) - lat_nodes[0]
    lon_index = np.floor(np.asarray(lons) / NODE_DEG).astype(np.int64) - lon_nodes[0]
    n_lat = lat_nodes[1] - lat_nodes[0] + 1
    n_lon = lon_nodes[1] - lon_nodes[0] + 1
    rows, cols = np.meshgrid(lat_index, lon_index, indexing="ij")
    keep = (rows >= 0) & (rows < n_lat) & (cols >= 0) & (cols < n_lon) & np.isfinite(values)
    sums = np.zeros((n_lat, n_lon))
    counts = np.zeros((n_lat, n_lon))
    np.add.at(sums, (rows[keep], cols[keep]), values[keep])
    np.add.at(counts, (rows[keep], cols[keep]), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        nodes = np.where(counts > 0, sums / counts, np.nan)
    node_lats = (np.arange(lat_nodes[0], lat_nodes[1] + 1) + 0.5) * NODE_DEG
    node_lons = (np.arange(lon_nodes[0], lon_nodes[1] + 1) + 0.5) * NODE_DEG
    return nodes, node_lats, node_lons


# ----------------------------------------------------------------------------- public: tile layers


def _mean_over_days(values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN cells stay NaN
        return np.nanmean(values, axis=0)


def weather_layer(grid, var, days):
    """One gridMET channel for a TileGrid, as the training export made it.

    The days are averaged per gridMET cell (the export took the median of two daily images, which
    for two values we take to be their mean), block-averaged onto the 20 km node grid, then
    bicubic-interpolated to every pixel centre. Wind direction is averaged as plain degrees with
    no 0/360 wrap, exactly like the export. Returns (float32 (64, 64) layer in model orientation
    with NaN where gridMET has no data, info dict).
    """
    if var not in VARIABLES:
        raise GridmetError(f"Unknown gridMET weather variable {var!r}. Use one of {VARIABLES}")
    days = sorted(set(days))
    pixel_lats, pixel_lons = grid.lats, grid.lons
    lat_nodes = _node_range(pixel_lats.min(), pixel_lats.max())
    lon_nodes = _node_range(pixel_lons.min(), pixel_lons.max())
    # Fetch exactly the cells that make up the needed nodes (fetch adds one cell of margin)
    south, north = lat_nodes[0] * NODE_DEG, (lat_nodes[1] + 1) * NODE_DEG
    west, east = lon_nodes[0] * NODE_DEG, (lon_nodes[1] + 1) * NODE_DEG
    values, lats, lons = fetch(var, days, south, west, north, east)
    nodes, node_lats, node_lons = node_grid(_mean_over_days(values), lats, lons, lat_nodes, lon_nodes)
    layer = bicubic(nodes, node_lats, node_lons, pixel_lats, pixel_lons).astype(np.float32)
    info = {
        "var": var,
        "days_used": [day.isoformat() for day in days],
        "source": "gridMET (University of Idaho THREDDS)",
        "missing_pixels": int(np.isnan(layer).sum()),
    }
    return layer, info


def pdsi_layer(grid, day):
    """The PDSI channel for a TileGrid: the latest pentad dated on or before `day`.

    Bicubic straight from the native 1/24° grid (the export reprojected PDSI to gridMET's own
    scale, not to 20 km). Returns (float32 (64, 64) layer in model orientation, info dict).
    """
    pentads = available_days(PDSI, 0)
    usable = [pentad for pentad in pentads if pentad <= day]
    if not usable:
        raise GridmetError(f"gridMET PDSI has no pentad on or before {day.isoformat()}")
    pentad = usable[-1]
    pixel_lats, pixel_lons = grid.lats, grid.lons
    step = 1 / 24
    values, lats, lons = fetch(
        PDSI, [pentad],
        pixel_lats.min() - 2.5 * step, pixel_lons.min() - 2.5 * step,
        pixel_lats.max() + 2.5 * step, pixel_lons.max() + 2.5 * step,
    )
    layer = bicubic(values[0], lats, lons, pixel_lats, pixel_lons).astype(np.float32)
    info = {
        "var": PDSI,
        "pentad_date": pentad.isoformat(),
        "days_after_pentad": (day - pentad).days,
        "source": "gridMET drought (University of Idaho THREDDS)",
        "missing_pixels": int(np.isnan(layer).sum()),
    }
    return layer, info


def clear_caches():
    """Forget cached metadata and data (tests, or to force fresh reads)."""
    with _cache_lock:
        _meta_cache.clear()
        _data_cache.clear()
