"""backend/spread_model.py must match train_spread.ipynb exactly.

Runs the notebook's own model and scaling code next to the backend module on the stored
NDWS test tiles. Needs data/ndws/*test*.tfrecord and the Kaggle outputs in data/training,
so it is skipped where those aren't downloaded.
"""

from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

from backend import spread_model

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "train_spread.ipynb"
TEST_SHARDS = sorted((REPO / "data" / "ndws").glob("next_day_wildfire_spread_test_*.tfrecord"))
LOG = REPO / "data" / "training" / "skyalert-fire-spread-unet.log"

needs_data = pytest.mark.skipif(
    not (TEST_SHARDS and spread_model.WEIGHTS_PATH.exists() and spread_model.STATS_PATH.exists()),
    reason="needs data/ndws test shard and data/training Kaggle outputs",
)


@pytest.fixture(scope="module")
def notebook(tmp_path_factory):
    """Definitions from the notebook's settings, reader, scaling and model cells, without running training."""
    cells = {cell.get("id"): cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells}
    namespace = {"torch": torch}
    settings = cells["b2e0d003"]
    # Keep the notebook from making folders: only the constants are needed
    settings = settings.split("ON_KAGGLE = ")[0]
    exec(settings, namespace)
    namespace["DATA_ROOT"] = REPO / "data" / "ndws"
    exec(cells["b2e0d007"].split("data, rows = {}, []")[0], namespace)
    exec(cells["b2e0d013"].split("for split in SPLITS:")[0], namespace)
    exec(cells["b2e0d020"].split("parameters =")[0], namespace)
    return namespace


def test_constants_match_notebook(notebook):
    assert tuple(notebook["INPUT_FEATURES"]) == spread_model.INPUT_FEATURES
    assert notebook["VALID_RANGES"] == spread_model.VALID_RANGES
    assert notebook["TILE_SIZE"] == spread_model.TILE_SIZE


def test_model_layers_match_notebook(notebook):
    ours = spread_model.UNet(len(spread_model.INPUT_FEATURES)).state_dict()
    theirs = notebook["UNet"](len(notebook["INPUT_FEATURES"])).state_dict()
    assert list(ours) == list(theirs)
    assert all(ours[key].shape == theirs[key].shape for key in ours)


@needs_data
def test_threshold_matches_training_log():
    assert "Fire threshold from val: 0.621" in LOG.read_text()
    assert spread_model.THRESHOLD == 0.621


@needs_data
def test_stats_file_matches_module():
    stats = pd.read_csv(spread_model.STATS_PATH, index_col="feature")
    assert list(stats.index) == list(spread_model.INPUT_FEATURES)


@needs_data
def test_predictions_match_notebook(notebook):
    inputs, labels, _ = notebook["load_split"]("test")
    stats = spread_model.load_stats()

    # Notebook path: mark_missing then normalize, in place, then the notebook's U-Net
    theirs_inputs = inputs.copy()
    notebook["mark_missing"](theirs_inputs)
    notebook["normalize"](theirs_inputs, pd.read_csv(spread_model.STATS_PATH).set_index("feature"))
    theirs_model = notebook["UNet"](len(notebook["INPUT_FEATURES"]))
    theirs_model.load_state_dict(torch.load(spread_model.WEIGHTS_PATH, map_location="cpu", weights_only=True))
    theirs_model.eval()
    with torch.no_grad():
        theirs = torch.sigmoid(theirs_model(torch.from_numpy(theirs_inputs)))[:, 0].numpy()

    # Backend path, on the raw inputs
    ours_inputs = inputs.copy()
    ours = spread_model.predict(ours_inputs, model=spread_model.load_model(), stats=stats)

    assert np.array_equal(ours_inputs, inputs, equal_nan=True), "predict must not change its input"
    assert np.array_equal(spread_model.normalize(inputs, stats), theirs_inputs)
    assert ours.shape == labels.shape
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-5)
