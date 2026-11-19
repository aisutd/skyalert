import math

import numpy as np
import pytest

from backend.spread_grid import CENTER_INDEX, PIXEL_DEG, TILE_SIZE, TileGrid


@pytest.mark.parametrize("row0_is_south", [True, False])
@pytest.mark.parametrize("geometry", ["degree", "km"])
@pytest.mark.parametrize("point", [(38.5, -120.5), (34.0001, -118.9999), (47.9, -67.2), (25.1, -80.4)])
def test_center_pixel_contains_point(point, geometry, row0_is_south):
    grid = TileGrid.around(*point, geometry=geometry, row0_is_south=row0_is_south)
    rows, cols = grid.cell_of([point[0]], [point[1]])
    assert (rows[0], cols[0]) == (CENTER_INDEX, CENTER_INDEX)
    lat, lon = grid.center
    assert abs(lat - point[0]) <= grid.lat_step / 2 + 1e-12
    assert abs(lon - point[1]) <= grid.lon_step / 2 + 1e-12


def test_south_up_orientation():
    grid = TileGrid.around(40.0, -105.0, row0_is_south=True)
    assert grid.lats[0, 0] < grid.lats[-1, 0], "row 0 must be the south edge"
    assert grid.lons[0, 0] < grid.lons[0, -1], "column 0 must be the west edge"
    assert np.allclose(np.diff(grid.lats[:, 0]), PIXEL_DEG)


def test_north_up_orientation():
    grid = TileGrid.around(40.0, -105.0, row0_is_south=False)
    assert grid.lats[0, 0] > grid.lats[-1, 0]
    assert np.allclose(np.diff(grid.lats[:, 0]), -PIXEL_DEG)


def test_degree_geometry_is_square_in_degrees():
    grid = TileGrid.around(40.0, -105.0, geometry="degree")
    south, west, north, east = grid.bounds
    assert math.isclose(north - south, TILE_SIZE * PIXEL_DEG)
    assert math.isclose(east - west, TILE_SIZE * PIXEL_DEG)
    # About 64 km tall and about 49 km wide at 40 N
    assert 63 < (north - south) * 111.32 < 65
    assert 48 < (east - west) * 111.32 * math.cos(math.radians(40)) < 50


def test_km_geometry_is_about_64_km_wide():
    grid = TileGrid.around(40.0, -105.0, geometry="km")
    south, west, north, east = grid.bounds
    assert 63 < (east - west) * 111.32 * math.cos(math.radians(40)) < 65


def test_pixels_sit_on_the_global_grid():
    a = TileGrid.around(40.0, -105.0)
    b = TileGrid.around(40.2, -104.8)
    # Pixel edges are multiples of PIXEL_DEG from (0, 0), so nearby tiles share edges
    for value in (a.bounds[0], a.bounds[1], b.bounds[0], b.bounds[1]):
        assert math.isclose(value / PIXEL_DEG, round(value / PIXEL_DEG), abs_tol=1e-9)


@pytest.mark.parametrize("row0_is_south", [True, False])
def test_every_pixel_centre_maps_back_to_its_own_cell(row0_is_south):
    grid = TileGrid.around(36.7, -119.4, row0_is_south=row0_is_south)
    rows, cols = grid.cell_of(grid.lats.ravel(), grid.lons.ravel())
    expected_rows, expected_cols = np.indices((TILE_SIZE, TILE_SIZE))
    assert np.array_equal(rows, expected_rows.ravel())
    assert np.array_equal(cols, expected_cols.ravel())


def test_rasterize_marks_points_and_ignores_outside():
    grid = TileGrid.around(36.7, -119.4)
    lats = [grid.lats[5, 7], grid.lats[5, 7] + 1e-6, grid.lats[60, 2], 10.0]
    lons = [grid.lons[5, 7], grid.lons[5, 7], grid.lons[60, 2], 10.0]
    mask = grid.rasterize(lats, lons)
    assert mask.dtype == np.float32
    assert mask.sum() == 2 and mask[5, 7] == 1 and mask[60, 2] == 1


def test_to_north_up_puts_north_first():
    grid = TileGrid.around(36.7, -119.4, row0_is_south=True)
    north_up = grid.to_north_up(grid.lats)
    assert north_up[0, 0] > north_up[-1, 0]


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        TileGrid.around(95, 0)
    with pytest.raises(ValueError):
        TileGrid.around(40, -100, geometry="metres")
