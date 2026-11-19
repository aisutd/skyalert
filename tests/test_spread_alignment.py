"""Our tiles line up with the training tiles.

Three NDWS tiles were located on Earth by matching their elevation channel against a real DEM
(NCC 0.99+, only under a vertical flip of the stored array). Rebuilding their elevation with
TileGrid + terrain.elevation, with no manual flipping, must reproduce the stored channel.
This pins down orientation (row 0 = south), degree-based pixels, and the centre pixel.
"""

from pathlib import Path

import numpy as np
import pytest

from backend.spread_grid import TileGrid
from backend.terrain import elevation

REPO = Path(__file__).resolve().parents[1]
NDWS = REPO / "data" / "ndws"

# (shard, record position, centre lat, centre lon of stored pixel [31, 31])
LOCATED_TILES = [
    ("next_day_wildfire_spread_test_01.tfrecord", 677, 36.35089, -118.71629),
    ("next_day_wildfire_spread_eval_00.tfrecord", 939, 36.572, -118.761),
    ("next_day_wildfire_spread_train_00.tfrecord", 828, 34.075, -116.785),
]


def stored_elevation(shard, position):
    import nbformat

    cells = {c.get("id"): c.source for c in nbformat.read(REPO / "notebooks" / "train_spread.ipynb", as_version=4).cells}
    namespace = {}
    exec(cells["b2e0d003"].split("ON_KAGGLE = ")[0], namespace)
    exec(cells["b2e0d007"].split("data, rows = {}, []")[0], namespace)
    for index, grids in enumerate(namespace["read_records"](NDWS / shard)):
        if index == position:
            return grids["elevation"]
    raise IndexError(position)


@pytest.mark.network
@pytest.mark.parametrize("shard,position,lat,lon", LOCATED_TILES)
def test_rebuilt_elevation_matches_training_tile(shard, position, lat, lon):
    if not (NDWS / shard).exists():
        pytest.skip(f"needs data/ndws/{shard}")
    stored = stored_elevation(shard, position)
    ours = elevation(TileGrid.around(lat, lon))
    correlation = np.corrcoef(ours.ravel(), stored.ravel())[0, 1]
    flipped = np.corrcoef(ours[::-1].ravel(), stored.ravel())[0, 1]
    assert correlation > 0.99, correlation
    assert correlation - flipped > 0.3, (correlation, flipped)
    # Terrarium vs SRTM and 1 km averaging differ by tens of metres, not hundreds
    assert np.abs(ours - stored).mean() < 80


def _knot_phases(fields_and_grids):
    """Where bicubic 'knots' (|4th difference|) fall on the global pixel grid, mod 20 (= one 20 km node)."""
    lat_mass, lon_mass = np.zeros(20), np.zeros(20)
    for field, grid in fields_and_grids:
        field = (field - field.mean()) / (field.std() + 1e-9)
        for k, value in enumerate(np.abs(np.diff(field, n=4, axis=0)).mean(axis=1)):
            lat_mass[(grid.first_lat_index + k + 2) % 20] += value  # rows are south-up
        for k, value in enumerate(np.abs(np.diff(field, n=4, axis=1)).mean(axis=0)):
            lon_mass[(grid.first_lon_index + k + 2) % 20] += value
    return lat_mass / lat_mass.sum(), lon_mass / lon_mass.sum()


def test_weather_node_grid_lines_up_with_training_knots():
    """The 20 km node grid sits where the training export put it (gridMET's corner, not 0°, 0°).

    Bicubic interpolation leaves faint 'knots' at node centres. Their phase in the stored training
    weather pins down the node grid. Our node_grid + bicubic on random gridMET-like fields must put
    the knots at the same phases.
    """
    from backend import gridmet

    shards = [tile for tile in LOCATED_TILES if (NDWS / tile[0]).exists()]
    if not shards:
        pytest.skip("needs data/ndws shards")
    import nbformat

    cells = {c.get("id"): c.source for c in nbformat.read(REPO / "notebooks" / "train_spread.ipynb", as_version=4).cells}
    namespace = {}
    exec(cells["b2e0d003"].split("ON_KAGGLE = ")[0], namespace)
    exec(cells["b2e0d007"].split("data, rows = {}, []")[0], namespace)

    stored, ours = [], []
    rng = np.random.default_rng(0)
    for shard, position, lat, lon in shards:
        record = next(g for i, g in enumerate(namespace["read_records"](NDWS / shard)) if i == position)
        grid = TileGrid.around(lat, lon)
        stored += [(record[name].astype(float), grid) for name in ("tmmx", "tmmn", "erc", "sph", "vs")]
        lat_nodes = gridmet._node_range(grid.lats.min(), grid.lats.max(), gridmet.NODE_ORIGIN_LAT)
        lon_nodes = gridmet._node_range(grid.lons.min(), grid.lons.max(), gridmet.NODE_ORIGIN_LON)
        south = gridmet.NODE_ORIGIN_LAT + lat_nodes[0] * gridmet.NODE_DEG - 0.1
        north = gridmet.NODE_ORIGIN_LAT + (lat_nodes[1] + 1) * gridmet.NODE_DEG + 0.1
        west = gridmet.NODE_ORIGIN_LON + lon_nodes[0] * gridmet.NODE_DEG - 0.1
        east = gridmet.NODE_ORIGIN_LON + (lon_nodes[1] + 1) * gridmet.NODE_DEG + 0.1
        # gridMET cell centres as the THREDDS files list them
        lats = np.sort(49.4 - np.arange(585) / 24)
        lats = lats[(lats > south) & (lats < north)]
        lons = -124.76666663333334 + np.arange(1386) / 24
        lons = lons[(lons > west) & (lons < east)]
        for _ in range(20):
            field = rng.normal(size=(len(lats), len(lons))).cumsum(0).cumsum(1)
            nodes, node_lats, node_lons = gridmet.node_grid(field, lats, lons, lat_nodes, lon_nodes)
            ours.append((gridmet.bicubic(nodes, node_lats, node_lons, grid.lats, grid.lons), grid))

    stored_lat, stored_lon = _knot_phases(stored)
    our_lat, our_lon = _knot_phases(ours)
    # Measured: about 0.07 (lat) and 0.03 (lon) with gridMET's corner; 1.0 and 2.0 with a (0°, 0°) origin
    assert np.abs(our_lat - stored_lat).sum() < 0.2
    assert np.abs(our_lon - stored_lon).sum() < 0.2
