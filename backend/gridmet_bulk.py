"""gridMET from whole yearly files on disk, for building training data in bulk.

backend/gridmet.py reads small boxes over OPeNDAP, which is right for one live forecast but would
mean hundreds of thousands of requests to build a training set. This module downloads each yearly
file once (`/thredds/fileServer/MET/{var}/{var}_{YYYY}.nc`, HDF5, read with h5py) and produces
**exactly** what gridmet.weather_layer produces for a TileGrid: the same unpacking (scale, offset
and fill read from the file), the same cell selection, the same day averaging, the same 20 km
node grid anchored at gridMET's corner, and the same bicubic. Only the source of the raw numbers
differs. tests/test_gridmet_bulk.py checks the two against each other.

PDSI comes from one 7.5 GB file, so it is never downloaded whole: each needed pentad is fetched
once over OPeNDAP for the whole grid (one request per pentad) and kept on disk.

Wind for the v2 model (backend/spread_model_v2.py) is wind_u = -vs * sin(th), wind_v = -vs * cos(th)
(th = direction the wind blows FROM, degrees clockwise from north), computed per gridMET cell and
day before the days are averaged, then put through the same node + bicubic recipe.
"""

import datetime as dt
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import requests

from backend import gridmet

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "data" / "cache" / "retrain" / "gridmet"
PDSI_CACHE_DIR = REPO / "data" / "cache" / "retrain" / "pdsi"
TIMEOUT = 120
BLOCK_CACHE_PER_VAR = 2  # decoded day-blocks kept per variable (one block ~ 61 days x CONUS x 2 bytes ~ 99 MB)


class BulkError(Exception):
    """A yearly file is missing, damaged, or not laid out as expected."""


# ----------------------------------------------------------------------------- downloading

# The same files are served by three mirrors; the path after the host is identical
MIRRORS = (
    "https://www.reacchpna.org/thredds/fileServer/MET",
    "https://thredds.northwestknowledge.net/thredds/fileServer/MET",
    "https://tds-proxy.nkn.uidaho.edu/thredds/fileServer/MET",
)
DOWNLOAD_URL = MIRRORS[0] + "/{var}/{var}_{year}.nc"
ATTEMPTS = 6  # rounds over all mirrors, with exponential backoff between rounds
BACKOFF_SECONDS = 15
MANIFEST = "manifest.json"  # verified downloads: file name -> size in bytes


def file_path(var, year, cache_dir=None):
    return Path(cache_dir or CACHE_DIR) / f"{var}_{year}.nc"


def _manifest(cache_dir):
    path = Path(cache_dir or CACHE_DIR) / MANIFEST
    return json.loads(path.read_text()) if path.exists() else {}


def _record(cache_dir, name, size):
    folder = Path(cache_dir or CACHE_DIR)
    manifest = _manifest(folder)
    manifest[name] = size
    partial = folder / (MANIFEST + ".part")
    partial.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    os.replace(partial, folder / MANIFEST)


def _with_mirrors(action, what, log=print, sleep=time.sleep):
    """Run action(mirror) on each mirror in turn, for up to ATTEMPTS rounds with exponential backoff.

    Retries timeouts, connection errors and 5xx answers; raises BulkError when every round fails.
    """
    problems = []
    for attempt in range(ATTEMPTS):
        for mirror in MIRRORS:
            try:
                return action(mirror)
            except (requests.exceptions.RequestException, BulkError) as error:
                host = mirror.split("/")[2]
                problems.append(f"{host}: {type(error).__name__}")
        wait = BACKOFF_SECONDS * 2**attempt
        log(f"  {what}: every mirror failed (round {attempt + 1}/{ATTEMPTS}: {', '.join(problems[-3:])}); waiting {wait}s")
        sleep(wait)
    raise BulkError(f"{what}: every mirror failed {ATTEMPTS} times ({'; '.join(problems[-3:])})")


def remote_size(var, year, log=print, sleep=time.sleep):
    def head(mirror):
        response = requests.head(f"{mirror}/{var}/{var}_{year}.nc", timeout=60, allow_redirects=True)
        if response.status_code == 404:
            raise BulkError(f"{var}_{year}.nc: HTTP 404")  # retried on other mirrors, then fails
        if response.status_code != 200:
            raise BulkError(f"{var}_{year}.nc: HTTP {response.status_code}")
        return int(response.headers["Content-Length"])

    return _with_mirrors(head, f"size of {var}_{year}.nc", log, sleep)


def download(var, year, cache_dir=None, log=print, sleep=time.sleep):
    """Download one yearly file. Returns (path, bytes, seconds); seconds is 0 when it was already there.

    Resumable: bytes go to .part and continue with an HTTP Range request (a mirror that ignores the
    Range restarts the file); the size is checked against Content-Length before the rename. Files
    already verified (in the manifest, same size on disk) are skipped without contacting the server.
    """
    if var not in gridmet.VARIABLES:
        raise BulkError(f"Unknown gridMET variable {var!r}")
    target = file_path(var, year, cache_dir)
    known = _manifest(cache_dir).get(target.name)
    if known and target.exists() and target.stat().st_size == known:
        return target, known, 0.0
    size = remote_size(var, year, log, sleep)
    if target.exists() and target.stat().st_size == size:
        _record(cache_dir, target.name, size)
        return target, size, 0.0
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    started = time.time()

    def fetch(mirror):
        have = partial.stat().st_size if partial.exists() else 0
        if have > size:
            partial.unlink()
            have = 0
        if have == size:
            return
        headers = {"Range": f"bytes={have}-"} if have else {}
        with requests.get(f"{mirror}/{var}/{var}_{year}.nc", headers=headers, stream=True, timeout=120) as response:
            if response.status_code not in (200, 206):
                raise BulkError(f"{var}_{year}.nc: HTTP {response.status_code}")
            with open(partial, "ab" if response.status_code == 206 else "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
        if partial.stat().st_size != size:
            raise BulkError(f"{var}_{year}.nc: got {partial.stat().st_size} of {size} bytes")

    _with_mirrors(fetch, f"{var}_{year}.nc", log, sleep)
    os.replace(partial, target)
    _record(cache_dir, target.name, size)
    return target, size, time.time() - started


# ----------------------------------------------------------------------------- reading


class YearFile:
    """One yearly gridMET file: its grid, packing, and raw values by day (decoded in chunk-sized blocks)."""

    def __init__(self, path, var):
        import h5py

        self.path = Path(path)
        self.var = var
        self._h5 = h5py.File(self.path, "r")
        grids = [key for key, item in self._h5.items() if getattr(item, "ndim", 0) == 3]
        if len(grids) != 1:
            raise BulkError(f"{self.path.name}: expected one 3-D grid, found {grids}")
        self.name = grids[0]
        self._data = self._h5[self.name]
        attrs = self._data.attrs
        self.scale = float(np.asarray(attrs.get("scale_factor", [1.0])).ravel()[0])
        self.offset = float(np.asarray(attrs.get("add_offset", [0.0])).ravel()[0])
        self.fills = tuple(
            sorted({float(np.asarray(attrs[key]).ravel()[0]) for key in ("_FillValue", "missing_value") if key in attrs})
        )
        self.lats = self._h5["lat"][:]
        self.lons = self._h5["lon"][:]
        self.days = tuple(gridmet._to_date(value) for value in self._h5["day"][:])
        if self._data.shape != (len(self.days), len(self.lats), len(self.lons)):
            raise BulkError(f"{self.path.name}: grid {self._data.shape} isn't [day][lat][lon]")
        if not (np.all(np.diff(self.lats) < 0) and np.all(np.diff(self.lons) > 0)):
            raise BulkError(f"{self.path.name}: expected latitude descending and longitude ascending")
        self.index_of = {day: i for i, day in enumerate(self.days)}
        self.block = self._data.chunks[0] if self._data.chunks else 32
        self._blocks = OrderedDict()
        self._lock = threading.Lock()

    @property
    def meta(self):
        """The same description gridmet.py builds from OPeNDAP, so gridmet._unpack applies unchanged."""
        return gridmet._FileMeta(
            path=f"{self.var}/{self.var}_{self.days[0].year}.nc",
            name=self.name, scale=self.scale, offset=self.offset, fills=self.fills,
            lats=self.lats, lons=self.lons, days=self.days,
        )

    def raw(self, day_index, i0, i1, j0, j1):
        """Raw packed values for one day, rows i0..i1 and columns j0..j1 inclusive (lat descending)."""
        start = (day_index // self.block) * self.block
        with self._lock:
            block = self._blocks.get(start)
            if block is None:
                block = self._data[start : start + self.block]  # decompresses one row of chunks
                self._blocks[start] = block
                while len(self._blocks) > BLOCK_CACHE_PER_VAR:
                    self._blocks.popitem(last=False)
            else:
                self._blocks.move_to_end(start)
        return block[day_index - start, i0 : i1 + 1, j0 : j1 + 1]

    def close(self):
        self._h5.close()


class Bulk:
    """Weather layers for TileGrids from local yearly files (and cached PDSI pentads)."""

    def __init__(self, cache_dir=None, pdsi_cache_dir=None):
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.pdsi_cache_dir = Path(pdsi_cache_dir or PDSI_CACHE_DIR)
        self._files = {}
        self._lock = threading.Lock()
        self._pdsi_meta = None
        self._pdsi_planes = OrderedDict()  # a few decoded pentads in memory (each ~1.6 MB as int16)

    def year_file(self, var, year):
        key = (var, year)
        with self._lock:
            if key not in self._files:
                path = file_path(var, year, self.cache_dir)
                if not path.exists():
                    raise BulkError(f"{path.name} isn't downloaded (run: python -m backend.gridmet_bulk download)")
                self._files[key] = YearFile(path, var)
            return self._files[key]

    def fetch(self, var, days, south, west, north, east):
        """Same contract and cell selection as gridmet.fetch, from local files."""
        days = list(days)
        if not days:
            raise BulkError("fetch needs at least one day")
        result, lats, lons = None, None, None
        for position, day in enumerate(days):
            year = self.year_file(var, day.year)
            if day not in year.index_of:
                raise BulkError(f"{var}_{day.year}.nc has no {day.isoformat()}")
            step_lat = abs(year.lats[1] - year.lats[0])
            step_lon = abs(year.lons[1] - year.lons[0])
            lat_rows = np.nonzero((year.lats >= south - step_lat) & (year.lats <= north + step_lat))[0]
            lon_cols = np.nonzero((year.lons >= west - step_lon) & (year.lons <= east + step_lon))[0]
            if not len(lat_rows) or not len(lon_cols):
                raise BulkError(f"The box ({south}, {west}, {north}, {east}) is outside gridMET")
            i0, i1, j0, j1 = int(lat_rows[0]), int(lat_rows[-1]), int(lon_cols[0]), int(lon_cols[-1])
            plane = gridmet._unpack(year.raw(year.index_of[day], i0, i1, j0, j1)[None], year.meta)[0]
            plane = plane[::-1, :]  # stored latitude descends; flip so it ascends
            file_lats = year.lats[i0 : i1 + 1][::-1]
            file_lons = year.lons[j0 : j1 + 1]
            if result is None:
                result = np.full((len(days), len(file_lats), len(file_lons)), np.nan, dtype=np.float32)
                lats, lons = file_lats, file_lons
            elif plane.shape != result.shape[1:] or not (np.allclose(lats, file_lats) and np.allclose(lons, file_lons)):
                raise BulkError(f"{var}: the years in this request use different grids")
            result[position] = plane
        return result, lats, lons

    def _node_box(self, grid):
        pixel_lats, pixel_lons = grid.lats, grid.lons
        lat_nodes = gridmet._node_range(pixel_lats.min(), pixel_lats.max(), gridmet.NODE_ORIGIN_LAT)
        lon_nodes = gridmet._node_range(pixel_lons.min(), pixel_lons.max(), gridmet.NODE_ORIGIN_LON)
        box = (
            gridmet.NODE_ORIGIN_LAT + lat_nodes[0] * gridmet.NODE_DEG,
            gridmet.NODE_ORIGIN_LON + lon_nodes[0] * gridmet.NODE_DEG,
            gridmet.NODE_ORIGIN_LAT + (lat_nodes[1] + 1) * gridmet.NODE_DEG,
            gridmet.NODE_ORIGIN_LON + (lon_nodes[1] + 1) * gridmet.NODE_DEG,
        )
        return lat_nodes, lon_nodes, box

    def _recipe(self, grid, per_cell, lats, lons, lat_nodes, lon_nodes):
        nodes, node_lats, node_lons = gridmet.node_grid(per_cell, lats, lons, lat_nodes, lon_nodes)
        return gridmet.bicubic(nodes, node_lats, node_lons, grid.lats, grid.lons).astype(np.float32)

    def weather_layer(self, grid, var, days):
        """Exactly gridmet.weather_layer(grid, var, days)[0], from local files."""
        if var not in gridmet.VARIABLES:
            raise BulkError(f"Unknown gridMET weather variable {var!r}")
        days = sorted(set(days))
        lat_nodes, lon_nodes, (south, west, north, east) = self._node_box(grid)
        values, lats, lons = self.fetch(var, days, south, west, north, east)
        return self._recipe(grid, gridmet._mean_over_days(values), lats, lons, lat_nodes, lon_nodes)

    def wind_layers(self, grid, days):
        """(wind_u, wind_v) in m/s: components per cell and day, averaged over the days, then the recipe."""
        days = sorted(set(days))
        lat_nodes, lon_nodes, (south, west, north, east) = self._node_box(grid)
        th, lats, lons = self.fetch("th", days, south, west, north, east)
        vs, lats_vs, lons_vs = self.fetch("vs", days, south, west, north, east)
        if th.shape != vs.shape or not (np.allclose(lats, lats_vs) and np.allclose(lons, lons_vs)):
            raise BulkError("th and vs cover different cells")
        u, v = wind_components(th, vs)
        return (
            self._recipe(grid, gridmet._mean_over_days(u), lats, lons, lat_nodes, lon_nodes),
            self._recipe(grid, gridmet._mean_over_days(v), lats, lons, lat_nodes, lon_nodes),
        )

    # ------------------------------------------------------------------------- PDSI

    def _pdsi(self):
        if self._pdsi_meta is None:
            self._pdsi_meta = gridmet._meta("pdsi/pdsi.nc")
        return self._pdsi_meta

    def pdsi_pentad(self, day):
        """The latest PDSI pentad dated on or before `day` (same rule as gridmet.pdsi_layer)."""
        usable = [pentad for pentad in self._pdsi().days if pentad <= day]
        if not usable:
            raise BulkError(f"No PDSI pentad on or before {day.isoformat()}")
        return usable[-1]

    def pdsi_plane(self, pentad):
        """Raw packed PDSI for one pentad over the whole grid (lat descending), cached on disk."""
        with self._lock:
            if pentad in self._pdsi_planes:
                self._pdsi_planes.move_to_end(pentad)
                return self._pdsi_planes[pentad]
        meta = self._pdsi()
        cached = self.pdsi_cache_dir / f"pdsi_{pentad.isoformat()}.npz"
        if cached.exists():
            raw = np.load(cached)["raw"]
        else:
            index = meta.days.index(pentad)
            i1, j1 = len(meta.lats) - 1, len(meta.lons) - 1
            raw = gridmet._raw_subset(meta, index, index, 0, i1, 0, j1)[0]
            # gridmet keeps every subset in memory for hours; a whole-grid plane is big, so drop it
            with gridmet._cache_lock:
                gridmet._data_cache.pop((meta.path, index, index, 0, i1, 0, j1), None)
            raw = np.asarray(raw)
            if raw.min() < 0 or raw.max() > 65535:
                raise BulkError(f"PDSI {pentad}: packed values outside UInt16")
            raw = raw.astype(np.uint16)  # gridMET's own packed type; halves the disk cache
            gridmet._save_npz(cached, raw=raw)
        with self._lock:
            self._pdsi_planes[pentad] = raw
            while len(self._pdsi_planes) > 4:
                self._pdsi_planes.popitem(last=False)
        return raw

    def pdsi_layer(self, grid, day):
        """Exactly gridmet.pdsi_layer(grid, day)[0], from a cached whole-grid pentad."""
        meta = self._pdsi()
        pentad = self.pdsi_pentad(day)
        pixel_lats, pixel_lons = grid.lats, grid.lons
        step = 1 / 24
        south, west = pixel_lats.min() - 2.5 * step, pixel_lons.min() - 2.5 * step
        north, east = pixel_lats.max() + 2.5 * step, pixel_lons.max() + 2.5 * step
        # Same cell selection as gridmet.fetch
        step_lat = abs(meta.lats[1] - meta.lats[0])
        step_lon = abs(meta.lons[1] - meta.lons[0])
        lat_rows = np.nonzero((meta.lats >= south - step_lat) & (meta.lats <= north + step_lat))[0]
        lon_cols = np.nonzero((meta.lons >= west - step_lon) & (meta.lons <= east + step_lon))[0]
        i0, i1, j0, j1 = int(lat_rows[0]), int(lat_rows[-1]), int(lon_cols[0]), int(lon_cols[-1])
        raw = self.pdsi_plane(pentad)[i0 : i1 + 1, j0 : j1 + 1]
        values = gridmet._unpack(raw[None], meta)[0][::-1, :]
        lats = meta.lats[i0 : i1 + 1][::-1]
        lons = meta.lons[j0 : j1 + 1]
        return gridmet.bicubic(values, lats, lons, pixel_lats, pixel_lons).astype(np.float32), pentad


def wind_components(th, vs):
    """Eastward (u) and northward (v) wind from direction-from (degrees clockwise from north) and speed.

    Wind FROM the north (th = 0) blows south: v < 0. FROM the west (th = 270) blows east: u > 0.
    NaN in either input gives NaN.
    """
    radians = np.deg2rad(np.asarray(th, dtype=np.float64))
    speed = np.asarray(vs, dtype=np.float64)
    return (-speed * np.sin(radians)).astype(np.float32), (-speed * np.cos(radians)).astype(np.float32)


def check_attributes(var, year, cache_dir=None):
    """Compare the local file's packing with what gridmet.py reads from the OPeNDAP .das."""
    local = YearFile(file_path(var, year, cache_dir), var)
    try:
        remote = gridmet._meta(f"{var}/{var}_{year}.nc")
        return {
            "var": var, "year": year, "name": local.name, "remote_name": remote.name,
            "scale": local.scale, "remote_scale": remote.scale,
            "offset": local.offset, "remote_offset": remote.offset,
            "fills": local.fills, "remote_fills": remote.fills,
            "days": len(local.days), "remote_days": len(remote.days),
            "match": (
                local.name == remote.name and local.scale == remote.scale and local.offset == remote.offset
                and set(local.fills) == set(remote.fills) and local.days == remote.days
                and np.allclose(local.lats, remote.lats) and np.allclose(local.lons, remote.lons)
            ),
        }
    finally:
        local.close()


def download_all(variables, years, min_free_gb=50.0, log=print):
    """Download every (var, year), skipping verified files. One file's failure never stops the rest:
    failures are retried in a second pass. Returns (bytes downloaded, failures)."""
    total, failed = 0, []
    jobs = [(var, year) for year in years for var in variables]
    for pass_number in (1, 2):
        failed = []
        for var, year in jobs:
            try:
                target = file_path(var, year)
                if not (target.exists() and _manifest(None).get(target.name) == target.stat().st_size):
                    stats = os.statvfs(CACHE_DIR.parent if CACHE_DIR.parent.exists() else REPO)
                    if stats.f_bavail * stats.f_frsize / 1e9 - 0.2 < min_free_gb:
                        raise SystemExit(f"Stopping: less than {min_free_gb} GB would remain free")
                path, size, seconds = download(var, year, log=log)
                if seconds:
                    total += size
                    log(f"{path.name}: {size / 1e6:.1f} MB in {seconds:.0f}s")
            except BulkError as error:
                log(f"FAILED {var}_{year}: {error}")
                failed.append((var, year))
        if not failed:
            break
        log(f"pass {pass_number}: {len(failed)} files failed; {'retrying them' if pass_number == 1 else 'giving up for now'}")
        jobs = failed
    return total, failed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download gridMET yearly files for building training data")
    parser.add_argument("command", choices=["download", "check"])
    parser.add_argument("--years", default="2018-2025", help="e.g. 2018-2025")
    parser.add_argument("--vars", default=",".join(gridmet.VARIABLES))
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    args = parser.parse_args()
    first, last = (int(part) for part in args.years.split("-"))
    years, variables = list(range(first, last + 1)), args.vars.split(",")
    started = time.time()
    if args.command == "download":
        logger = lambda message: print(f"[{dt.datetime.now():%H:%M:%S}] {message}", flush=True)
        total, failed = download_all(variables, years, args.min_free_gb, logger)
        logger(f"done: {total / 1e9:.2f} GB downloaded in {time.time() - started:.0f}s; failed: {failed or 'none'}")
        raise SystemExit(1 if failed else 0)
    for year in years:
        for var in variables:
            print(check_attributes(var, year), flush=True)
