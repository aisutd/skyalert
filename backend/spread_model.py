"""The trained next-day fire spread model, ready to run inside the backend.

Training happens in notebooks/train_spread.ipynb on Kaggle. That notebook has to stand
alone on Kaggle, so the model class and input scaling are repeated here. They must
stay identical to the notebook: tests/test_spread_model.py checks that this module
gives exactly the same probabilities as the notebook code on the stored test tiles.

Input: a float32 array (tiles, 12, 64, 64) in INPUT_FEATURES order, in raw units.
Missing pixels can be NaN. They become the train mean, as in training.
Output: the probability (0 to 1) that each 1 km pixel is burning the next day.
"""

import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = REPO / "data" / "training" / "spread_unet.pt"
STATS_PATH = REPO / "data" / "training" / "spread_feature_stats.csv"

TILE_SIZE = 64  # pixels per side, 1 km each

# Input channel order, the same as INPUT_FEATURES in train_spread.ipynb
INPUT_FEATURES = (
    "elevation",  # meters
    "th",  # wind direction, degrees clockwise from north
    "vs",  # wind speed, meters per second
    "tmmn",  # minimum temperature, kelvin
    "tmmx",  # maximum temperature, kelvin
    "sph",  # specific humidity
    "pr",  # precipitation, millimeters
    "pdsi",  # Palmer Drought Severity Index
    "NDVI",  # vegetation index, stored scaled rather than -1 to 1
    "population",  # population density
    "erc",  # energy release component, BTU per square foot
    "PrevFireMask",  # today's fire: 1 fire, 0 no fire, -1 uncertain
)

# Values outside these ranges are fill values in the source data, so they count as missing.
# Same as VALID_RANGES in train_spread.ipynb.
VALID_RANGES = {"th": (0, 360), "tmmn": (200, 330), "tmmx": (200, 330)}

# A pixel is forecast as fire when its probability is at least this. Chosen as the best F1
# on the val split by the Kaggle run (data/training/skyalert-fire-spread-unet.log).
THRESHOLD = 0.621

# Test-split scores from data/training/spread_results.csv, shown to people next to forecasts
TEST_SCORES = {
    "model": {"precision": 0.245, "recall": 0.525, "iou": 0.200},
    "persistence": {"precision": 0.357, "recall": 0.273, "iou": 0.183},
}


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """U-Net Light (Pereira et al. 2021), copied from train_spread.ipynb."""

    def __init__(self, in_channels, base=16, depth=4, dropout=0.1):
        super().__init__()
        widths = [base * 2**level for level in range(depth)]
        self.down = nn.ModuleList()
        channels = in_channels
        for width in widths:
            self.down.append(conv_block(channels, width))
            channels = width
        self.bottom = conv_block(widths[-1], widths[-1] * 2)
        self.dropout = nn.Dropout2d(dropout)
        self.up = nn.ModuleList(nn.ConvTranspose2d(width * 2, width, 2, stride=2) for width in reversed(widths))
        self.merge = nn.ModuleList(conv_block(width * 2, width) for width in reversed(widths))
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.dropout(self.bottom(x))
        for up, merge, skip in zip(self.up, self.merge, reversed(skips)):
            x = merge(torch.cat([up(x), skip], dim=1))
        return self.head(x)


_model = None
_stats = None
_load_lock = threading.Lock()


def load_stats(path=STATS_PATH):
    """The train-split clip range, mean and std of each input, written by train_spread.ipynb."""
    stats = pd.read_csv(path, index_col="feature")
    missing = [name for name in INPUT_FEATURES if name not in stats.index]
    if missing:
        raise ValueError(f"{path} has no stats for {missing}")
    return stats.loc[list(INPUT_FEATURES), ["low", "high", "mean", "std"]]


def load_model(weights_path=WEIGHTS_PATH):
    """Build the U-Net and load the trained weights. strict=True fails on any mismatch."""
    model = UNet(len(INPUT_FEATURES))
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def _loaded():
    """Load the weights and stats once, the first time a forecast is asked for."""
    global _model, _stats
    with _load_lock:
        if _model is None:
            for path in (WEIGHTS_PATH, STATS_PATH):
                if not path.exists():
                    raise FileNotFoundError(
                        f"{path} is missing. Download the Kaggle outputs into data/training "
                        "(see README, Model Training)."
                    )
            _stats = load_stats()
            _model = load_model()
        return _model, _stats


def normalize(inputs, stats):
    """Scale raw inputs the way train_spread.ipynb does. Returns a new array.

    1. Pixels outside VALID_RANGES are fill values, so they become missing (NaN).
    2. Each feature is clipped to its train 0.1-99.9 percentile range and standardized
       with the train mean and std.
    3. Missing pixels become 0, the train mean after standardizing.
    """
    inputs = np.array(inputs, dtype=np.float32, copy=True)
    if inputs.ndim != 4 or inputs.shape[1:] != (len(INPUT_FEATURES), TILE_SIZE, TILE_SIZE):
        raise ValueError(f"inputs must be (tiles, {len(INPUT_FEATURES)}, {TILE_SIZE}, {TILE_SIZE}), got {inputs.shape}")
    for name, (low, high) in VALID_RANGES.items():
        plane = inputs[:, INPUT_FEATURES.index(name)]
        plane[(plane < low) | (plane > high)] = np.nan
    for channel, name in enumerate(INPUT_FEATURES):
        low, high, mean, std = stats.loc[name, ["low", "high", "mean", "std"]]
        plane = inputs[:, channel]
        np.clip(plane, low, high, out=plane)
        plane -= mean
        plane /= std
        plane[np.isnan(plane)] = 0.0
    return inputs


@torch.no_grad()
def predict(inputs, model=None, stats=None, batch_size=256):
    """Fire probability for tomorrow, shape (tiles, 64, 64), from raw inputs (tiles, 12, 64, 64)."""
    if model is None or stats is None:
        loaded_model, loaded_stats = _loaded()
        model = model or loaded_model
        stats = stats if stats is not None else loaded_stats
    scaled = normalize(inputs, stats)
    probabilities = []
    for start in range(0, len(scaled), batch_size):
        batch = torch.from_numpy(scaled[start : start + batch_size])
        probabilities.append(torch.sigmoid(model(batch))[:, 0].numpy())
    return np.concatenate(probabilities)
