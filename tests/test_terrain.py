import math
import time

import numpy as np
import pytest
import requests

from backend import terrain
from backend.spread_grid import CENTER_INDEX, TileGrid


def encode(elevation_m):
    """Terrarium RGB for elevations (inverse of terrain.decode, to 1/256 m)."""
    total = np.asarray(elevation_m, dtype=np.float64) + 32768.0
    red = np.floor(total / 256)
    green = np.floor(total - red * 256)
    blue = np.round((total - red * 256 - green) * 256)
    # Rounding blue up to 256 carries into green
    carry = blue >= 256
    blue = np.where(carry, 0, blue)
    green = green + carry
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


@pytest.fixture
def stub_tiles(monkeypatch):
    """Replace the web fetch with synthetic tiles: value_fn(global_x, global_y) -> metres."""
    fetched = []

    def install(value_fn):
        def fake_fetch(zoom, x, y):
            fetched.append((zoom, x, y))
            gx, gy = np.meshgrid(x * 256 + np.arange(256), y * 256 + np.arange(256))
            return encode(value_fn(gx, gy))

        monkeypatch.setattr(terrain, "fetch_tile_rgb", fake_fetch)
        terrain.clear_cache()
        return fetched

    yield install
    terrain.clear_cache()


def test_decode_matches_research_values():
    assert terrain.decode(np.array([128, 0, 0])) == 0.0
    # Mount Whitney pixel from the research notes: RGB (145, 60, 95) -> 4412.37 m
    assert math.isclose(terrain.decode(np.array([145, 60, 95])), 4412.37109375)
    values = np.array([-86.0, 0.0, 1897.5, 4412.37109375, -2197.15])
    assert np.allclose(terrain.decode(encode(values)), values, atol=1 / 512)


def test_known_tile_coordinates():
    # From the research notes: Mount Whitney is in z12 tile 702/1600
    assert terrain.tile_of(36.5785, -118.2923, 12) == (702, 1600)
    # (0, 0) is the top-left corner of tile (1, 1) at zoom 1
    assert terrain.tile_of(-0.0001, 0.0001, 1) == (1, 1)
    assert terrain.tile_of(0.0001, -0.0001, 1) == (0, 0)


def test_pixel_centre_round_trip():
    x_index = np.array([0, 1000, 175_000, 262_143])
    y_index = np.array([0, 98_765, 101_000, 262_143])
    lat, lon = terrain.pixel_center(x_index, y_index, 10)
    x, y = terrain.global_pixel(lat, lon, 10)
    assert np.allclose(x, x_index + 0.5, atol=1e-6)
    assert np.allclose(y, y_index + 0.5, atol=1e-6)
    # Top-left pixel of the world at zoom 0: half a pixel in from (-180, ~85.05)
    lat0, lon0 = terrain.pixel_center(0, 0, 0)
    assert math.isclose(lon0, -180 + 360 / 512)
    assert 84.9 < lat0 < 85.06


def test_tiles_covering_spans_the_tile():
    grid = TileGrid.around(38.5, -120.5)
    tiles = terrain.tiles_covering(grid.bounds)
    south, west, north, east = grid.bounds
    for lat, lon in [(south, west), (south, east), (north, west), (north, east), grid.center]:
        assert terrain.tile_of(lat, lon) in tiles
    assert len(tiles) == len(set(tiles)) <= 9


@pytest.mark.parametrize("geometry", ["degree", "km"])
@pytest.mark.parametrize("point", [(38.5, -120.5), (47.5, -122.0), (25.5, -80.5)])
def test_every_pixel_gets_data_and_each_tile_is_fetched_once(stub_tiles, point, geometry):
    fetched = stub_tiles(lambda gx, gy: np.full(gx.shape, 1234.0))
    out = terrain.elevation(TileGrid.around(*point, geometry=geometry))
    assert out.shape == (64, 64) and out.dtype == np.float32
    assert not np.isnan(out).any()
    assert np.allclose(out, 1234.0)
    assert len(fetched) == len(set(fetched))


def test_tiles_are_cached_between_calls(stub_tiles):
    fetched = stub_tiles(lambda gx, gy: np.full(gx.shape, 10.0))
    grid = TileGrid.around(38.5, -120.5)
    terrain.elevation(grid)
    first = len(fetched)
    terrain.elevation(grid)
    assert len(fetched) == first


def test_negative_values_become_zero(stub_tiles):
    stub_tiles(lambda gx, gy: np.full(gx.shape, -50.0))
    assert np.all(terrain.elevation(TileGrid.around(37.7, -123.2)) == 0)


def test_cell_value_is_the_mean_of_the_pixels_inside_it(stub_tiles):
    # Checkerboard of 100 m and 300 m pixels with a gentle ramp, so every cell has its own mean
    def value(gx, gy):
        return np.where((gx + gy) % 2 == 0, 100.0, 300.0) + (gx % 97) * 0.5

    stub_tiles(value)
    grid = TileGrid.around(38.5, -120.5)
    out = terrain.elevation(grid)
    south, west, _, _ = grid.bounds
    for row, col in [(0, 0), (31, 31), (63, 63), (5, 50), (50, 5)]:
        # Independent brute force: every zoom-10 pixel whose centre lies inside this cell's edges
        lat_c, lon_c = grid.lats[row, col], grid.lons[row, col]
        lat_lo, lat_hi = lat_c - grid.lat_step / 2, lat_c + grid.lat_step / 2
        lon_lo, lon_hi = lon_c - grid.lon_step / 2, lon_c + grid.lon_step / 2
        x0, y0 = terrain.global_pixel(lat_hi, lon_lo, 10)
        x1, y1 = terrain.global_pixel(lat_lo, lon_hi, 10)
        values = []
        for gy in range(int(y0) - 1, int(y1) + 2):
            for gx in range(int(x0) - 1, int(x1) + 2):
                plat, plon = terrain.pixel_center(gx, gy, 10)
                if lat_lo <= plat < lat_hi and lon_lo <= plon < lon_hi:
                    values.append(value(np.array(gx), np.array(gy)))
        assert len(values) > 20
        assert math.isclose(out[row, col], np.mean(values), abs_tol=0.01), (row, col)


@pytest.mark.parametrize("row0_is_south", [True, False])
def test_orientation_follows_the_grid(stub_tiles, row0_is_south):
    # Higher to the north: elevation falls as Web Mercator y (south) grows
    stub_tiles(lambda gx, gy: (262_144 - gy) / 10.0)
    grid = TileGrid.around(40.0, -105.0, row0_is_south=row0_is_south)
    out = terrain.elevation(grid)
    assert np.corrcoef(out.ravel(), grid.lats.ravel())[0, 1] > 0.999
    step = np.diff(out, axis=0)
    assert np.all(step > 0) if row0_is_south else np.all(step < 0)


def test_fetch_failure_raises_terrain_error(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.exceptions.ConnectionError("offline")

    monkeypatch.setattr(terrain.requests, "get", fail)
    monkeypatch.setattr(terrain.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(terrain, "CACHE_DIR", None)
    terrain.clear_cache()
    with pytest.raises(terrain.TerrainError):
        terrain.elevation(TileGrid.around(38.5, -120.5))


def test_disk_cache_round_trip(monkeypatch, tmp_path):
    png = (tmp_path / "source.png")
    from PIL import Image

    Image.fromarray(encode(np.full((256, 256), 777.0))).save(png)
    calls = []

    class Response:
        content = png.read_bytes()

        def raise_for_status(self):
            pass

    def fake_get(url, timeout):
        calls.append(url)
        return Response()

    monkeypatch.setattr(terrain.requests, "get", fake_get)
    monkeypatch.setattr(terrain, "CACHE_DIR", tmp_path / "cache")
    assert np.allclose(terrain.fetch_tile_rgb(10, 1, 2), encode(np.full((256, 256), 777.0)))
    assert (tmp_path / "cache" / "10_1_2.png").exists()
    terrain.fetch_tile_rgb(10, 1, 2)
    assert len(calls) == 1


# ---- network: real Terrarium tiles ----


@pytest.fixture
def real_tiles():
    terrain.clear_cache()
    yield
    terrain.clear_cache()


@pytest.mark.network
def test_lake_tahoe_surface(real_tiles):
    grid = TileGrid.around(39.09, -120.04)
    started = time.time()
    out = terrain.elevation(grid)
    print(f"cold elevation(grid) call: {time.time() - started:.2f} s")
    assert not np.isnan(out).any()
    assert abs(out[CENTER_INDEX, CENTER_INDEX] - 1897) <= 10


@pytest.mark.network
def test_mount_whitney_area_is_high(real_tiles):
    out = terrain.elevation(TileGrid.around(36.5785, -118.2923))
    assert out.max() > 3800


@pytest.mark.network
def test_offshore_is_zero(real_tiles):
    out = terrain.elevation(TileGrid.around(37.7, -123.2))
    assert out[CENTER_INDEX, CENTER_INDEX] == 0


@pytest.mark.network
def test_matches_open_meteo_points(real_tiles):
    grid = TileGrid.around(38.5, -120.5)
    out = terrain.elevation(grid)
    cells = [(0, 0), (31, 31), (63, 63), (10, 50), (50, 10)]
    lats = [float(grid.lats[r, c]) for r, c in cells]
    lons = [float(grid.lons[r, c]) for r, c in cells]
    response = requests.get(
        "https://api.open-meteo.com/v1/elevation",
        params={"latitude": ",".join(f"{v:.6f}" for v in lats), "longitude": ",".join(f"{v:.6f}" for v in lons)},
        timeout=20,
    )
    response.raise_for_status()
    reference = response.json()["elevation"]
    for (row, col), ours, theirs in zip(cells, (out[r, c] for r, c in cells), reference):
        print(f"cell [{row},{col}] ours {ours:.0f} m, Open-Meteo point {theirs:.0f} m, diff {ours - theirs:+.0f} m")
        # A 1 km mean against a 90 m point sample in the Sierra foothills: allow terrain spread
        assert abs(ours - theirs) <= 250


def test_sea_depths_are_clamped_before_averaging(stub_tiles):
    # Alternating source columns at -100 m (sea) and +100 m (land). SRTM stores the sea as 0, so each
    # cell is 100 m times its share of land columns: 3/7 or 4/7 here (a cell spans an odd number of
    # source columns). Clamping after averaging would give 0 to 14 m instead.
    stub_tiles(lambda gx, gy: np.where(gx % 2 == 0, -100.0, 100.0))
    out = terrain.elevation(TileGrid.around(38.5, -120.5))
    assert out.min() > 40 and out.max() < 60
