"""backend/gridmet_bulk.py must give exactly what backend/gridmet.py gives, from local yearly files.

Offline tests use a small synthetic HDF5 file laid out like gridMET's. Network tests compare the
real downloaded files against the OPeNDAP path (gridmet.weather_layer / pdsi_layer); they need
the yearly files in data/cache/retrain/gridmet (python -m backend.gridmet_bulk download).
"""

import datetime as dt

import h5py
import numpy as np
import pytest

from backend import gridmet, gridmet_bulk
from backend.gridmet_bulk import Bulk, BulkError, YearFile, wind_components
from backend.spread_grid import TileGrid

# ----------------------------------------------------------------------------- offline


def test_wind_components_signs():
    # FROM the north blows south: v < 0, u = 0
    u, v = wind_components(np.array([0.0]), np.array([5.0]))
    assert np.isclose(u[0], 0, atol=1e-6) and np.isclose(v[0], -5)
    # FROM the west (270) blows east: u > 0
    u, v = wind_components(np.array([270.0]), np.array([4.0]))
    assert np.isclose(u[0], 4) and np.isclose(v[0], 0, atol=1e-6)
    # FROM the south-east (135) blows north-west: u < 0, v > 0
    u, v = wind_components(np.array([135.0]), np.array([2.0]))
    assert u[0] < 0 and v[0] > 0 and np.isclose(np.hypot(u[0], v[0]), 2)
    u, v = wind_components(np.array([np.nan, 90.0]), np.array([3.0, np.nan]))
    assert np.isnan(u).all() and np.isnan(v).all()


def make_year_file(path, values, lats, lons, days, scale=0.1, offset=220.0, fill=32767, name="air_temperature"):
    with h5py.File(path, "w") as handle:
        handle["lat"] = lats
        handle["lon"] = lons
        handle["day"] = np.array([(day - gridmet.EPOCH).days for day in days], dtype=float)
        data = handle.create_dataset(name, data=values.astype(np.uint16), chunks=(2, len(lats), len(lons)))
        data.attrs["scale_factor"] = np.array([scale])
        data.attrs["add_offset"] = np.array([offset])
        data.attrs["_FillValue"] = np.array([fill], dtype=np.uint16)
        data.attrs["missing_value"] = np.array([fill], dtype=np.uint16)


@pytest.fixture
def synthetic(tmp_path):
    lats = 41.0 - np.arange(30) / 24  # descending, like gridMET
    lons = -121.0 + np.arange(40) / 24
    days = [dt.date(2020, 8, 1) + dt.timedelta(days=i) for i in range(5)]
    rng = np.random.default_rng(0)
    raw = rng.integers(500, 900, size=(5, 30, 40))
    raw[2, 4, 7] = 32767  # a fill value
    make_year_file(tmp_path / "tmmx_2020.nc", raw, lats, lons, days)
    return tmp_path, raw, lats, lons, days


def test_year_file_reads_packing_and_layout(synthetic):
    folder, raw, lats, lons, days = synthetic
    year = YearFile(folder / "tmmx_2020.nc", "tmmx")
    assert year.name == "air_temperature" and year.scale == 0.1 and year.offset == 220.0
    assert year.fills == (32767.0,)
    assert year.days == tuple(days)
    # Reading a later block then an earlier one gives the right values (block cache)
    assert np.array_equal(year.raw(4, 0, 29, 0, 39), raw[4])
    assert np.array_equal(year.raw(1, 3, 5, 10, 12), raw[1, 3:6, 10:13])


def test_bulk_fetch_matches_manual_unpacking(synthetic):
    folder, raw, lats, lons, days = synthetic
    bulk = Bulk(cache_dir=folder)
    values, out_lats, out_lons = bulk.fetch("tmmx", [days[2], days[0]], 40.2, -120.5, 40.6, -120.0)
    assert np.all(np.diff(out_lats) > 0), "lat must come back ascending"
    rows = [int(np.nonzero(np.isclose(lats, lat))[0][0]) for lat in out_lats]
    cols = [int(np.nonzero(np.isclose(lons, lon))[0][0]) for lon in out_lons]
    for position, day_index in enumerate([2, 0]):
        expected = raw[day_index][np.ix_(rows, cols)].astype(np.float64) * 0.1 + 220.0
        expected[raw[day_index][np.ix_(rows, cols)] == 32767] = np.nan
        np.testing.assert_allclose(values[position], expected.astype(np.float32), rtol=0, atol=1e-4)
    # The selection rule is gridmet.fetch's: every cell within one cell of the box
    step = 1 / 24
    assert out_lats.min() >= 40.2 - step - 1e-9 and out_lats.max() <= 40.6 + step + 1e-9


def test_missing_file_and_day_are_errors(synthetic):
    folder, *_ = synthetic
    bulk = Bulk(cache_dir=folder)
    with pytest.raises(BulkError):
        bulk.fetch("tmmx", [dt.date(2021, 8, 1)], 40.2, -120.5, 40.6, -120.0)
    with pytest.raises(BulkError):
        bulk.fetch("tmmx", [dt.date(2020, 9, 1)], 40.2, -120.5, 40.6, -120.0)


def test_rejects_wrong_layout(tmp_path):
    lats = 41.0 + np.arange(5) / 24  # ascending: not gridMET's layout
    lons = -121.0 + np.arange(6) / 24
    make_year_file(tmp_path / "tmmx_2020.nc", np.zeros((2, 5, 6)), lats, lons, [dt.date(2020, 1, 1), dt.date(2020, 1, 2)])
    with pytest.raises(BulkError):
        YearFile(tmp_path / "tmmx_2020.nc", "tmmx")


# ----------------------------------------------------------------------------- network: equal to OPeNDAP

TILES = [(38.5, -120.5), (34.2, -117.2), (44.9, -114.6)]
DAYS = [dt.date(2020, 8, 19), dt.date(2020, 8, 20)]


def _need(var, year):
    if not gridmet_bulk.file_path(var, year).exists():
        pytest.skip(f"needs {var}_{year}.nc downloaded")


@pytest.fixture
def fresh_gridmet(monkeypatch):
    monkeypatch.setattr(gridmet, "CACHE_DIR", None)
    gridmet.clear_caches()
    yield
    gridmet.clear_caches()


@pytest.mark.network
@pytest.mark.parametrize("var", ["tmmx", "erc", "pr"])
@pytest.mark.parametrize("tile", TILES)
def test_weather_layer_equals_opendap(fresh_gridmet, tile, var):
    _need(var, 2020)
    grid = TileGrid.around(*tile)
    ours = Bulk().weather_layer(grid, var, DAYS)
    theirs, _ = gridmet.weather_layer(grid, var, DAYS)
    assert ours.shape == theirs.shape == (64, 64)
    assert np.array_equal(np.isnan(ours), np.isnan(theirs))
    np.testing.assert_allclose(ours, theirs, rtol=1e-4, atol=1e-6, equal_nan=True)


@pytest.mark.network
def test_wind_layers_equal_opendap_recipe(fresh_gridmet):
    _need("th", 2020)
    _need("vs", 2020)
    grid = TileGrid.around(*TILES[0])
    u, v = Bulk().wind_layers(grid, DAYS)
    # Independent path: OPeNDAP th and vs, components per cell and day, then the same recipe
    lat_nodes = gridmet._node_range(grid.lats.min(), grid.lats.max(), gridmet.NODE_ORIGIN_LAT)
    lon_nodes = gridmet._node_range(grid.lons.min(), grid.lons.max(), gridmet.NODE_ORIGIN_LON)
    box = (
        gridmet.NODE_ORIGIN_LAT + lat_nodes[0] * gridmet.NODE_DEG, gridmet.NODE_ORIGIN_LON + lon_nodes[0] * gridmet.NODE_DEG,
        gridmet.NODE_ORIGIN_LAT + (lat_nodes[1] + 1) * gridmet.NODE_DEG, gridmet.NODE_ORIGIN_LON + (lon_nodes[1] + 1) * gridmet.NODE_DEG,
    )
    th, lats, lons = gridmet.fetch("th", DAYS, *box)
    vs, _, _ = gridmet.fetch("vs", DAYS, *box)
    for ours, component in zip((u, v), wind_components(th, vs)):
        nodes, node_lats, node_lons = gridmet.node_grid(np.nanmean(component, axis=0), lats, lons, lat_nodes, lon_nodes)
        expected = gridmet.bicubic(nodes, node_lats, node_lons, grid.lats, grid.lons)
        np.testing.assert_allclose(ours, expected, rtol=1e-4, atol=1e-5, equal_nan=True)
    # Sanity: wind speed from the components is within the daily speeds' range
    assert np.nanmax(np.hypot(u, v)) <= np.nanmax(vs) + 0.5


@pytest.mark.network
def test_pdsi_layer_equals_opendap(fresh_gridmet, tmp_path):
    grid = TileGrid.around(*TILES[1])
    ours, pentad = Bulk(pdsi_cache_dir=tmp_path).pdsi_layer(grid, DAYS[-1])
    theirs, info = gridmet.pdsi_layer(grid, DAYS[-1])
    assert pentad.isoformat() == info["pentad_date"]
    np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-5, equal_nan=True)


@pytest.mark.network
@pytest.mark.parametrize("var", gridmet.VARIABLES)
def test_local_file_packing_matches_das(fresh_gridmet, var):
    _need(var, 2020)
    result = gridmet_bulk.check_attributes(var, 2020)
    assert result["match"], result


# ----------------------------------------------------------------------------- downloader resilience


class FakeResponse:
    def __init__(self, status, body=b"", headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def iter_content(self, chunk_size):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_server(monkeypatch, tmp_path):
    """A stub of the three mirrors. `down` lists hosts that time out; `ignore_range` hosts send 200."""
    import requests

    content = bytes(range(256)) * 40
    state = {"down": set(), "ignore_range": set(), "calls": [], "cut": None}

    def host(url):
        return url.split("/")[2]

    def head(url, timeout, allow_redirects):
        state["calls"].append(("HEAD", host(url)))
        if host(url) in state["down"]:
            raise requests.exceptions.ConnectTimeout("down")
        return FakeResponse(200, headers={"Content-Length": str(len(content))})

    def get(url, headers=None, stream=False, timeout=None):
        state["calls"].append(("GET", host(url), (headers or {}).get("Range")))
        if host(url) in state["down"]:
            raise requests.exceptions.ConnectionError("down")
        start = int(headers["Range"].split("=")[1].rstrip("-")) if headers and host(url) not in state["ignore_range"] else 0
        body = content[start:]
        if state["cut"]:  # simulate a dropped connection part-way through, once
            body, state["cut"] = body[: state["cut"]], None
        return FakeResponse(206 if start else 200, body)

    monkeypatch.setattr(gridmet_bulk.requests, "head", head)
    monkeypatch.setattr(gridmet_bulk.requests, "get", get)
    monkeypatch.setattr(gridmet_bulk, "CACHE_DIR", tmp_path)
    return state, content, tmp_path


def test_falls_back_to_the_next_mirror(fake_server):
    state, content, folder = fake_server
    state["down"].add("www.reacchpna.org")
    path, size, _ = gridmet_bulk.download("erc", 2020, folder, log=lambda m: None, sleep=lambda s: None)
    assert path.read_bytes() == content and size == len(content)
    assert ("GET", "thredds.northwestknowledge.net", None) in state["calls"]


def test_resumes_a_cut_download_with_range(fake_server):
    state, content, folder = fake_server
    state["cut"] = 1000
    path, _, _ = gridmet_bulk.download("erc", 2020, folder, log=lambda m: None, sleep=lambda s: None)
    assert path.read_bytes() == content
    assert any(call[0] == "GET" and call[2] == "bytes=1000-" for call in state["calls"])
    assert not path.with_name(path.name + ".part").exists()


def test_restarts_when_a_mirror_ignores_range(fake_server):
    state, content, folder = fake_server
    (folder / "erc_2020.nc.part").write_bytes(content[:500])
    state["ignore_range"].add("www.reacchpna.org")
    path, _, _ = gridmet_bulk.download("erc", 2020, folder, log=lambda m: None, sleep=lambda s: None)
    assert path.read_bytes() == content


def test_verified_files_are_skipped_without_the_network(fake_server):
    state, content, folder = fake_server
    gridmet_bulk.download("erc", 2020, folder, log=lambda m: None, sleep=lambda s: None)
    state["calls"].clear()
    state["down"].update({"www.reacchpna.org", "thredds.northwestknowledge.net", "tds-proxy.nkn.uidaho.edu"})
    path, size, seconds = gridmet_bulk.download("erc", 2020, folder, log=lambda m: None, sleep=lambda s: None)
    assert seconds == 0 and state["calls"] == []


def test_all_mirrors_down_raises_after_backoff(fake_server):
    state, content, folder = fake_server
    state["down"].update({"www.reacchpna.org", "thredds.northwestknowledge.net", "tds-proxy.nkn.uidaho.edu"})
    waits = []
    with pytest.raises(BulkError):
        gridmet_bulk.download("erc", 2020, folder, log=lambda m: None, sleep=waits.append)
    assert waits == [gridmet_bulk.BACKOFF_SECONDS * 2**i for i in range(gridmet_bulk.ATTEMPTS)]


def test_one_failed_file_does_not_stop_the_rest(fake_server, monkeypatch):
    state, content, folder = fake_server
    real = gridmet_bulk.download

    def flaky(var, year, cache_dir=None, log=print, sleep=None):
        if var == "vs":
            raise BulkError("vs is broken")
        return real(var, year, cache_dir, log=log, sleep=lambda s: None)

    monkeypatch.setattr(gridmet_bulk, "download", flaky)
    total, failed = gridmet_bulk.download_all(["erc", "vs", "pr"], [2020], min_free_gb=0, log=lambda m: None)
    assert failed == [("vs", 2020)]
    assert (folder / "erc_2020.nc").exists() and (folder / "pr_2020.nc").exists()
