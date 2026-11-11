"""Tests for backend/gridmet.py.

Offline tests serve small fake gridMET files whose .dds, .das and .ascii answers copy the
format of the real server (captured on 2026-11-11). Network tests (run with --network) check
real values the research probe verified by hand.
"""

import datetime as dt
import re

import numpy as np
import pytest

from backend import gridmet
from backend.spread_grid import TileGrid

STEP = 1 / 24

# ----------------------------------------------------------------------------- a fake THREDDS server


class FakeFile:
    """One gridMET-style file: raw UInt16 cube [day][lat desc][lon asc] plus its attributes."""

    def __init__(self, name, raw, lats_desc, lons, days, scale, offset, extra_grid=None):
        self.name, self.raw, self.lats, self.lons, self.days = name, raw, lats_desc, lons, days
        self.scale, self.offset, self.extra_grid = scale, offset, extra_grid

    def dds(self, path):
        nd, nl, nn = self.raw.shape

        def grid(name):
            return (
                "    Grid {\n     ARRAY:\n"
                f"        UInt16 {name}[day = {nd}][lat = {nl}][lon = {nn}];\n"
                "     MAPS:\n"
                f"        Float64 day[day = {nd}];\n        Float64 lat[lat = {nl}];\n        Float64 lon[lon = {nn}];\n"
                f"    }} {name};\n"
            )

        body = grid(self.name) + (grid(self.extra_grid) if self.extra_grid else "")
        return (
            "Dataset {\n" + body + f"    Float64 lat[lat = {nl}];\n    Float64 day[day = {nd}];\n"
            f"    UInt16 crs[crs = 1];\n    Float64 lon[lon = {nn}];\n}} MET/{path};\n"
        )

    def das(self):
        block = (
            f"    {self.name} {{\n"
            "        Int16 _FillValue 32767;\n"
            '        String units "K";\n'
            "        Int16 missing_value 32767;\n"
            '        String dimensions "lon lat time";\n'
            f"        Float64 scale_factor {self.scale};\n"
            f"        Float64 add_offset {self.offset};\n"
            '        String _Unsigned "true";\n'
            "    }\n"
        )
        lat = '    lat {\n        String units "degrees_north";\n    }\n'
        return "Attributes {\n" + block + lat + "}\n"

    def ascii(self, path, query):
        header = f"Dataset {{\n    ...\n}} MET/{path};\n" + gridmet._SEPARATOR + "\n"
        if query == "lat,lon,day":
            def line(values):
                return ", ".join(repr(float(v)) for v in values)
            return header + (
                f"lat[{len(self.lats)}]\n{line(self.lats)}\n\n"
                f"day[{len(self.days)}]\n{line(self.days)}\n\n"
                f"lon[{len(self.lons)}]\n{line(self.lons)}\n\n"
            )
        match = re.fullmatch(re.escape(self.name) + r"\[(\d+):1:(\d+)\]\[(\d+):1:(\d+)\]\[(\d+):1:(\d+)\]", query)
        assert match, query
        d0, d1, i0, i1, j0, j1 = map(int, match.groups())
        cube = self.raw[d0 : d1 + 1, i0 : i1 + 1, j0 : j1 + 1]
        nd, nl, nn = cube.shape
        rows = [f"[{d}][{i}], " + ", ".join(str(int(v)) for v in cube[d, i]) for d in range(nd) for i in range(nl)]
        name = self.name
        return header + (
            f"{name}.{name}[{nd}][{nl}][{nn}]\n" + "\n".join(rows) + "\n\n"
            f"{name}.day[{nd}]\n" + ", ".join(repr(float(v)) for v in self.days[d0 : d1 + 1]) + "\n\n"
            f"{name}.lat[{nl}]\n" + ", ".join(repr(float(v)) for v in self.lats[i0 : i1 + 1]) + "\n\n"
            f"{name}.lon[{nn}]\n" + ", ".join(repr(float(v)) for v in self.lons[j0 : j1 + 1]) + "\n"
        )


class FakeServer:
    def __init__(self):
        self.files = {}
        self.requests = []

    def add(self, path, fake):
        self.files[path] = fake

    def __call__(self, path_and_query):
        self.requests.append(path_and_query)
        path, _, rest = path_and_query.partition(".nc")
        path += ".nc"
        if path not in self.files:
            raise gridmet._NotFound(f"no {path}")
        fake = self.files[path]
        if rest == ".dds":
            return fake.dds(path), "fake"
        if rest == ".das":
            return fake.das(), "fake"
        assert rest.startswith(".ascii?")
        return fake.ascii(path, rest[len(".ascii?") :]), "fake"


def day_numbers(first, count):
    return np.array([(first - gridmet.EPOCH).days + k for k in range(count)], dtype=float)


def grid_axes(lat_top, n_lat, lon_left, n_lon):
    lats = lat_top - STEP * np.arange(n_lat)  # descending, like gridMET
    lons = lon_left + STEP * np.arange(n_lon)
    return lats, lons


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(gridmet, "_fetch_text", fake)
    monkeypatch.setattr(gridmet, "CACHE_DIR", None)
    gridmet.clear_caches()
    yield fake
    gridmet.clear_caches()


def raw_from_physical(values, scale, offset):
    raw = np.round((values - offset) / scale)
    return np.where(np.isnan(values), 32767, raw).astype(np.uint16)


# ----------------------------------------------------------------------------- parsing

REAL_DAS = """Attributes {
    air_temperature {
        Int16 _FillValue 32767;
        String units "K";
        String description "Daily Maximum Temperature (2m)";
        Int16 missing_value 32767;
        String dimensions "lon lat time";
        Float64 scale_factor 0.1;
        Float64 add_offset 220.0;
        String coordinates "lon lat time";
        String _Unsigned "true";
        Int32 _ChunkSizes 53, 117, 278;
    }
    lat {
        String units "degrees_north";
    }
}
"""

REAL_ASCII = """Dataset {
    Grid {
     ARRAY:
        UInt16 air_temperature[day = 2][lat = 2][lon = 3];
     MAPS:
        Float64 day[day = 2];
        Float64 lat[lat = 2];
        Float64 lon[lon = 3];
    } air_temperature;
} MET/tmmx/tmmx_2026.nc;
---------------------------------------------
air_temperature.air_temperature[2][2][3]
[0][0], 737, 738, 732
[0][1], 734, 731, 32767
[1][0], 720, 724, 721
[1][1], 716, 716, 715

air_temperature.day[2]
46318.0, 46319.0

air_temperature.lat[2]
38.35833333333334, 38.31666666666667

air_temperature.lon[3]
-120.55833330000002, -120.51666663333334, -120.47499996666667
"""


def test_parse_das_reads_scale_offset_and_fill():
    scale, offset, fills = gridmet.parse_das(REAL_DAS, "air_temperature")
    assert (scale, offset, fills) == (0.1, 220.0, (32767.0,))


def test_parse_das_handles_dash_in_name():
    das = REAL_DAS.replace("air_temperature", "energy_release_component-g").replace("0.1;", "1.0;").replace("220.0", "0.0")
    assert gridmet.parse_das(das, "energy_release_component-g") == (1.0, 0.0, (32767.0,))


def test_parse_ascii_real_answer():
    arrays = gridmet.parse_ascii(REAL_ASCII)
    cube = arrays["air_temperature.air_temperature"]
    assert cube.shape == (2, 2, 3)
    assert cube[0, 0, 1] == 738 and cube[0, 1, 2] == 32767 and cube[1, 1, 0] == 716
    assert np.all(np.diff(arrays["air_temperature.lat"]) < 0)
    assert gridmet._to_date(46318) == dt.date(2026, 10, 25)


def test_parse_ascii_rejects_error_text():
    with pytest.raises(gridmet.GridmetError):
        gridmet.parse_ascii('Error { code = 400; message = "Bad Projection Request"; }')


def test_tmmn_and_tmmx_use_their_own_offsets(server):
    """Same variable name, different add_offset: the offset must come from each file's .das."""
    lats, lons = grid_axes(40.0, 4, -110.0, 4)
    days = day_numbers(dt.date(2026, 10, 25), 1)
    raw = np.full((1, 4, 4), 717, dtype=np.uint16)
    server.add("tmmn/tmmn_2026.nc", FakeFile("air_temperature", raw, lats, lons, days, 0.1, 210.0))
    server.add("tmmx/tmmx_2026.nc", FakeFile("air_temperature", raw, lats, lons, days, 0.1, 220.0))
    box = (39.9, -109.95, 39.95, -109.9)
    tmmn, _, _ = gridmet.fetch("tmmn", [dt.date(2026, 10, 25)], *box)
    tmmx, _, _ = gridmet.fetch("tmmx", [dt.date(2026, 10, 25)], *box)
    assert np.allclose(tmmn, 281.7, atol=1e-4)
    assert np.allclose(tmmx, 291.7, atol=1e-4)


def test_fetch_unpacks_masks_fill_and_flips_latitude(server):
    lats, lons = grid_axes(40.0, 6, -110.0, 5)
    days = day_numbers(dt.date(2026, 7, 1), 3)
    physical = np.empty((3, 6, 5))
    for d in range(3):
        physical[d] = 10 * d + lats[:, None] - 40 + 0 * lons[None, :]  # depends on day and latitude
    physical[1, 2, 3] = np.nan
    server.add("vs/vs_2026.nc", FakeFile("wind_speed", raw_from_physical(physical + 5, 0.1, 0.0), lats, lons, days, 0.1, 0.0))

    values, out_lats, out_lons = gridmet.fetch(
        "vs", [dt.date(2026, 7, 3), dt.date(2026, 7, 2)], lats[-1], lons[0], lats[0], lons[-1]
    )
    assert np.all(np.diff(out_lats) > 0), "latitude must come back ascending"
    assert np.allclose(out_lats, lats[::-1]) and np.allclose(out_lons, lons)
    # Days come back in the order asked for
    assert np.allclose(values[0], (physical[2] + 5)[::-1], atol=0.051)
    expected_day2 = (physical[1] + 5)[::-1]
    assert np.isnan(values[1]).sum() == 1 and np.isnan(values[1][6 - 1 - 2, 3])
    assert np.allclose(values[1][~np.isnan(expected_day2)], expected_day2[~np.isnan(expected_day2)], atol=0.051)
    assert values.dtype == np.float32


def test_fetch_spans_two_years(server):
    lats, lons = grid_axes(40.0, 3, -110.0, 3)
    for year, value in ((2025, 100), (2026, 200)):
        first = dt.date(year, 1, 1)
        count = (dt.date(year + 1, 1, 1) - first).days if year == 2025 else 10
        raw = np.full((count, 3, 3), value, dtype=np.uint16)
        server.add(f"erc/erc_{year}.nc", FakeFile("energy_release_component-g", raw, lats, lons, day_numbers(first, count), 1.0, 0.0))
    values, _, _ = gridmet.fetch("erc", [dt.date(2025, 12, 31), dt.date(2026, 1, 1)], lats[-1], lons[0], lats[0], lons[-1])
    assert np.all(values[0] == 100) and np.all(values[1] == 200)


def test_fetch_refuses_days_not_published_yet(server):
    lats, lons = grid_axes(40.0, 3, -110.0, 3)
    raw = np.zeros((5, 3, 3), dtype=np.uint16)
    server.add("pr/pr_2026.nc", FakeFile("precipitation_amount", raw, lats, lons, day_numbers(dt.date(2026, 1, 1), 5), 0.1, 0.0))
    with pytest.raises(gridmet.GridmetError, match="latest day is 2026-01-05"):
        gridmet.fetch("pr", [dt.date(2026, 1, 6)], lats[-1], lons[0], lats[0], lons[-1])
    with pytest.raises(gridmet.GridmetError, match="no file for 2027"):
        gridmet.fetch("pr", [dt.date(2027, 1, 1)], lats[-1], lons[0], lats[0], lons[-1])


def test_latest_day_falls_back_to_last_year(server):
    lats, lons = grid_axes(40.0, 2, -110.0, 2)
    raw = np.zeros((365, 2, 2), dtype=np.uint16)
    server.add("erc/erc_2026.nc", FakeFile("energy_release_component-g", raw, lats, lons, day_numbers(dt.date(2026, 1, 1), 365), 1.0, 0.0))
    assert gridmet.latest_day("erc", today=dt.date(2027, 1, 2)) == dt.date(2026, 12, 31)


def test_unknown_variable():
    with pytest.raises(gridmet.GridmetError):
        gridmet.fetch("wind", [dt.date(2026, 1, 1)], 39, -110, 40, -109)


# ----------------------------------------------------------------------------- resampling


def test_keys_weights_sum_to_one_and_hit_nodes():
    t = np.linspace(0, 0.999, 50)
    assert np.allclose(gridmet.keys_weights(t).sum(axis=-1), 1)
    assert np.allclose(gridmet.keys_weights(np.array([0.0]))[0], [0, 1, 0, 0])


def test_bicubic_reproduces_linear_and_quadratic_fields():
    lats = np.arange(30.0, 31.0, 0.1)
    lons = np.arange(-110.0, -109.0, 0.1)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    out_lats = np.random.default_rng(0).uniform(30.15, 30.75, 200)
    out_lons = np.random.default_rng(1).uniform(-109.85, -109.25, 200)
    for field in (lambda a, b: 3 * a - 2 * b + 7, lambda a, b: (a - 30) ** 2 + 0.5 * (b + 110) ** 2):
        result = gridmet.bicubic(field(la, lo), lats, lons, out_lats, out_lons)
        assert np.allclose(result, field(out_lats, out_lons), atol=1e-9)


def test_bicubic_returns_node_values_at_nodes():
    lats = np.arange(30.0, 31.0, 0.1)
    lons = np.arange(-110.0, -109.0, 0.1)
    values = np.random.default_rng(2).normal(size=(len(lats), len(lons)))
    la, lo = np.meshgrid(lats[2:-2], lons[2:-2], indexing="ij")
    result = gridmet.bicubic(values, lats, lons, la, lo)
    assert np.allclose(result, values[2:-2, 2:-2], atol=1e-12)


def test_bicubic_nan_rule():
    lats = np.arange(0.0, 1.0, 0.1)
    lons = np.arange(0.0, 1.0, 0.1)
    values = np.ones((10, 10))
    # One NaN tap with a small weight (about 0.03 at 0.42): the rest are renormalized, so a
    # constant field stays exact
    values[5, 5] = np.nan
    assert np.isclose(gridmet.bicubic(values, lats, lons, np.array([0.42]), np.array([0.42]))[0], 1.0)
    # Near the NaN node (about 0.7 of the weight at 0.47) less than half the weight is usable: NaN
    assert np.isnan(gridmet.bicubic(values, lats, lons, np.array([0.47]), np.array([0.47]))[0])
    # The point sits on the NaN node: its weight is ~all the weight, so the answer is NaN
    assert np.isnan(gridmet.bicubic(values, lats, lons, np.array([0.5]), np.array([0.5]))[0])
    # All taps NaN
    assert np.isnan(gridmet.bicubic(np.full((10, 10), np.nan), lats, lons, np.array([0.45]), np.array([0.45]))[0])
    # Off the grid edge: missing taps renormalized, far outside gives NaN
    assert np.isclose(gridmet.bicubic(np.ones((10, 10)), lats, lons, np.array([0.02]), np.array([0.5]))[0], 1.0)
    assert np.isnan(gridmet.bicubic(np.ones((10, 10)), lats, lons, np.array([5.0]), np.array([0.5]))[0])


def test_bicubic_rejects_irregular_grid():
    with pytest.raises(gridmet.GridmetError):
        gridmet.bicubic(np.ones((3, 3)), np.array([0, 0.1, 0.3]), np.array([0, 0.1, 0.2]), np.array([0.1]), np.array([0.1]))


def test_node_grid_block_means():
    nd = gridmet.NODE_DEG
    # Cells centred inside node (k_lat=200, k_lon=-600): average of their values; NaN cells skipped
    lats = 200 * nd + np.array([0.01, 0.05, 0.09, 0.13, 0.17, 0.19])  # last one is in node 201
    lons = -600 * nd + np.array([0.02, 0.06, 0.10])
    values = np.arange(18, dtype=float).reshape(6, 3)
    values[0, 0] = np.nan
    nodes, node_lats, node_lons = gridmet.node_grid(values, lats, lons, (200, 201), (-600, -600))
    assert nodes.shape == (2, 1)
    assert np.isclose(nodes[0, 0], np.nanmean(values[:5]))
    assert np.isclose(nodes[1, 0], values[5].mean())
    assert np.allclose(node_lats, [(200.5) * nd, (201.5) * nd]) and np.isclose(node_lons[0], -599.5 * nd)


def test_node_grid_empty_node_is_nan():
    nd = gridmet.NODE_DEG
    nodes, _, _ = gridmet.node_grid(np.ones((1, 1)), np.array([10 * nd + 0.01]), np.array([10 * nd + 0.01]), (10, 11), (10, 10))
    assert nodes[0, 0] == 1 and np.isnan(nodes[1, 0])


# ----------------------------------------------------------------------------- layers on a TileGrid


def add_ramp_file(server, var, name, center_lat, center_lon, days, value_of, scale=0.1, offset=0.0):
    """A gridMET-like file around a point, wide enough for weather_layer, with values from value_of(lat, lon)."""
    lat_top = np.floor(center_lat * 24) / 24 + 2.0 + STEP / 2
    lon_left = np.floor(center_lon * 24) / 24 - 2.0 + STEP / 2
    lats, lons = grid_axes(lat_top, 96, lon_left, 96)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    physical = np.repeat(value_of(la, lo)[None], len(days), axis=0)
    server.add(f"{var}/{var}_{days[0].year}.nc", FakeFile(name, raw_from_physical(physical, scale, offset), lats, lons,
                                                          day_numbers(days[0], len(days)), scale, offset))


@pytest.mark.parametrize("row0_is_south", [True, False])
def test_weather_layer_follows_grid_orientation(server, row0_is_south):
    days = [dt.date(2026, 8, 1), dt.date(2026, 8, 2)]
    add_ramp_file(server, "tmmx", "air_temperature", 38.5, -120.5, days, lambda la, lo: 300 + 2 * (la - 38.5), offset=220.0)
    grid = TileGrid.around(38.5, -120.5, row0_is_south=row0_is_south)
    layer, info = gridmet.weather_layer(grid, "tmmx", days)
    assert layer.shape == (64, 64) and layer.dtype == np.float32
    assert not np.isnan(layer).any()
    # Warmer to the north: the layer must increase along grid.lats, whichever way rows run
    along_lats = np.corrcoef(layer.ravel(), grid.lats.ravel())[0, 1]
    assert along_lats > 0.99
    # A linear field survives block means and bicubic almost exactly
    assert np.allclose(layer, 300 + 2 * (grid.lats - 38.5), atol=0.06)
    assert info["days_used"] == ["2026-08-01", "2026-08-02"]


def test_weather_layer_averages_the_days(server):
    days = [dt.date(2026, 8, 1), dt.date(2026, 8, 2)]
    lat_top = np.floor(38.5 * 24) / 24 + 2.0 + STEP / 2
    lon_left = np.floor(-120.5 * 24) / 24 - 2.0 + STEP / 2
    lats, lons = grid_axes(lat_top, 96, lon_left, 96)
    physical = np.stack([np.full((96, 96), 2.0), np.full((96, 96), 4.0)])
    server.add("vs/vs_2026.nc", FakeFile("wind_speed", raw_from_physical(physical, 0.1, 0.0), lats, lons,
                                         day_numbers(days[0], 2), 0.1, 0.0))
    layer, _ = gridmet.weather_layer(TileGrid.around(38.5, -120.5), "vs", days)
    assert np.allclose(layer, 3.0, atol=1e-5)


def test_pdsi_layer_uses_latest_pentad_on_or_before_day(server):
    pentads = [dt.date(2026, 10, 27), dt.date(2026, 11, 1), dt.date(2026, 11, 6)]
    lat_top = np.floor(38.5 * 24) / 24 + 1.0 + STEP / 2
    lon_left = np.floor(-120.5 * 24) / 24 - 1.0 + STEP / 2
    lats, lons = grid_axes(lat_top, 48, lon_left, 48)
    physical = np.stack([np.full((48, 48), value) for value in (-1.0, -2.0, -3.0)])
    days = np.array([(p - gridmet.EPOCH).days for p in pentads], dtype=float)
    server.add("pdsi/pdsi.nc", FakeFile("pdsi", raw_from_physical(physical, 0.01, -15.0), lats, lons, days, 0.01, -15.0,
                                        extra_grid="category"))
    layer, info = gridmet.pdsi_layer(TileGrid.around(38.5, -120.5), dt.date(2026, 11, 5))
    assert np.allclose(layer, -2.0, atol=1e-4)
    assert info["pentad_date"] == "2026-11-01" and info["days_after_pentad"] == 4
    with pytest.raises(gridmet.GridmetError):
        gridmet.pdsi_layer(TileGrid.around(38.5, -120.5), dt.date(2026, 10, 1))


def test_caches_avoid_repeat_requests(server):
    days = [dt.date(2026, 8, 1)]
    add_ramp_file(server, "erc", "energy_release_component-g", 38.5, -120.5, days, lambda la, lo: 50 + 0 * la, scale=1.0)
    grid = TileGrid.around(38.5, -120.5)
    gridmet.weather_layer(grid, "erc", days)
    count = len(server.requests)
    gridmet.weather_layer(grid, "erc", days)
    assert len(server.requests) == count


# ----------------------------------------------------------------------------- live server (--network)


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setattr(gridmet, "CACHE_DIR", None)
    gridmet.clear_caches()


@pytest.mark.network
def test_live_values_match_the_hand_checked_probe(live):
    """The probe read raw 719 (tmmx), 717 (tmmn) and 35 (vs) at (38.483, -120.517) on 2026-10-25."""
    day = [dt.date(2026, 10, 25)]
    box = (38.45, -120.55, 38.52, -120.48)
    expected = {"tmmx": 291.9, "tmmn": 281.7, "vs": 3.5}
    for var, value in expected.items():
        values, lats, lons = gridmet.fetch(var, day, *box)
        i = int(np.argmin(np.abs(lats - 38.48333333333334)))
        j = int(np.argmin(np.abs(lons - (-120.51666663333334))))
        assert abs(lats[i] - 38.48333) < 1e-4 and abs(lons[j] + 120.51667) < 1e-4
        assert values[0, i, j] == pytest.approx(value, abs=1e-3), var


@pytest.mark.network
def test_live_latest_day_is_the_same_for_every_variable(live):
    latest = {var: gridmet.latest_day(var) for var in gridmet.VARIABLES}
    print(latest)
    assert len(set(latest.values())) == 1
    assert next(iter(latest.values())) >= dt.date(2026, 11, 8)


@pytest.mark.network
def test_live_weather_and_pdsi_layers_for_a_real_tile(live):
    import time

    grid = TileGrid.around(38.5, -120.5)
    latest = gridmet.latest_day("erc")
    days = [latest - dt.timedelta(days=1), latest]
    plausible = {
        "th": (0, 360), "vs": (0, 30), "tmmn": (220, 320), "tmmx": (230, 330),
        "sph": (0, 0.03), "pr": (0, 300), "erc": (0, 150),
    }
    started = time.time()
    for var in gridmet.VARIABLES:
        layer, info = gridmet.weather_layer(grid, var, days)
        assert np.isfinite(layer).all(), var
        low, high = plausible[var]
        print(f"{var:5s} min {layer.min():.5g} max {layer.max():.5g} days {info['days_used']}")
        assert low <= layer.min() and layer.max() <= high, var
    pdsi, info = gridmet.pdsi_layer(grid, latest)
    print(f"pdsi  min {pdsi.min():.3f} max {pdsi.max():.3f} pentad {info['pentad_date']}")
    print(f"cold layers for 7 vars + pdsi: {time.time() - started:.1f}s")
    assert np.isfinite(pdsi).all() and -10 <= pdsi.min() and pdsi.max() <= 10


@pytest.mark.network
def test_live_backtest_year_2020(live):
    grid = TileGrid.around(37.2, -119.3)
    for var in ("tmmx", "erc"):
        layer, _ = gridmet.weather_layer(grid, var, [dt.date(2020, 8, 19), dt.date(2020, 8, 20)])
        assert np.isfinite(layer).all()
    pdsi, info = gridmet.pdsi_layer(grid, dt.date(2020, 8, 20))
    assert np.isfinite(pdsi).all() and info["pentad_date"] <= "2020-08-20"
