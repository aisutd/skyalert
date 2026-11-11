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
