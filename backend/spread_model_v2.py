"""The live-trained next-day spread model (v2): channels and dataset format.

See docs/spread_retrain_plan.md. This module is the single source of truth for the channel
order, shared by the dataset builder (training/build_spread_dataset.py) and the training code.

Inputs are float32 arrays (tiles, channels, 64, 64) in raw units, NaN = missing, in the TileGrid
orientation: rows increase northward (south-up), columns increase eastward. A trained run lives in
one folder (data/retrain/runs/<name>/) holding weights.pt, channel_stats.csv and manifest.json
(channel list, threshold, training settings, val scores); load_run() restores all of it.
"""

import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backend.spread_model import UNet  # same U-Net Light architecture; only in_channels differs

# Input channels, in order. All are (64, 64) on backend.spread_grid.TileGrid (south-up rows,
# degree pixels), raw units, NaN = missing.
CHANNELS = (
    "fire_t",  # VIIRS NOAA-20 fire on day t: 0/1 footprint mask (fire_points.footprint_mask, scale 1.0)
    "fire_t1",  # same, day t-1
    "fire_t2",  # same, day t-2
    "frp_t",  # log1p(summed FRP in MW of day-t detections whose centre is in the pixel), 0 where none
    "elevation",  # metres, terrain.elevation
    "wind_u",  # eastward wind (m/s): -vs*sin(th), per gridMET cell and day, then the weather recipe
    "wind_v",  # northward wind (m/s): -vs*cos(th), same
    "tmmn",  # K, gridMET recipe (weather_layer) over the weather days
    "tmmx",  # K
    "sph",  # kg/kg
    "pr",  # mm
    "erc",  # NFDRS energy release component
    "pdsi",  # gridMET drought, latest pentad on or before t
)
LABEL = "fire_t+1"  # VIIRS NOAA-20 fire on day t+1, same mask recipe as fire_t

# Weather days relative to the fire day t: the mean of t-2 and t-1, what live data has
# (gridMET runs about two days behind).
WEATHER_OFFSETS = (-2, -1)
FIRE_FOOTPRINT_SCALE = 1.0  # VIIRS pixels are 0.4 to 0.8 km, so mostly one 1 km cell
VIIRS_SATELLITES = ("N20",)

# Dataset files under data/retrain/ (written by training/build_spread_dataset.py):
#   {split}_inputs.npy  float16 (n, len(CHANNELS), 64, 64), NaN kept for missing (opened with mmap)
#   {split}_labels.npy  uint8 (n, 64, 64), 1 = fire on t+1
#   tiles.csv           one row per tile: split, index (row in the split arrays), fire_day,
#                       center_lat, center_lon, fire_t_px, label_px, new_fire_px, region
#   channel_stats.csv   per-channel low/high (0.1/99.9 percentiles), mean, std from train only
SPLIT_YEARS = {"train": range(2018, 2023), "val": (2023,), "test": (2024, 2025)}

TILE_SIZE = 64
FIRE_CHANNELS = ("fire_t", "fire_t1", "fire_t2")  # 0/1 masks: passed through unscaled, NaN -> 0
WIND_CHANNELS = ("wind_u", "wind_v")

REPO = Path(__file__).resolve().parents[1]
RETRAIN_DIR = REPO / "data" / "retrain"
STATS_PATH = RETRAIN_DIR / "channel_stats.csv"
RUNS_DIR = RETRAIN_DIR / "runs"


# ----------------------------------------------------------------------------- normalization


def load_stats(path=STATS_PATH, channels=CHANNELS):
    """Per-channel low/high/mean/std (train split only), indexed by channel name, in `channels` order."""
    stats = pd.read_csv(path, index_col=0)
    stats.index = stats.index.astype(str)
    missing = [name for name in channels if name not in stats.index and name not in FIRE_CHANNELS]
    if missing:
        raise ValueError(f"{path} has no stats for {missing}")
    rows = []
    for name in channels:
        if name in FIRE_CHANNELS:
            rows.append({"low": 0.0, "high": 1.0, "mean": 0.0, "std": 1.0})
        else:
            low, high, mean, std = (float(stats.loc[name, key]) for key in ("low", "high", "mean", "std"))
            if not high > low:
                # A collapsed clip range would silently turn the whole channel into one constant
                # (e.g. a sparse channel like frp_t whose 99.9th percentile over all pixels is 0)
                raise ValueError(f"{path}: channel {name!r} has low {low} >= high {high}; its stats need fixing")
            rows.append({"low": low, "high": high, "mean": mean, "std": max(std, 1e-6)})
    return pd.DataFrame(rows, index=list(channels))


def scaling_arrays(stats, channels):
    """(low, high, mean, std, is_fire) as float32 arrays shaped (C, 1, 1) for broadcasting."""
    table = stats.loc[list(channels)]
    shape = (len(channels), 1, 1)
    arrays = [table[key].to_numpy(np.float32).reshape(shape) for key in ("low", "high", "mean", "std")]
    is_fire = np.array([name in FIRE_CHANNELS for name in channels]).reshape(shape)
    return (*arrays, is_fire)


def normalize(inputs, stats, channels=CHANNELS):
    """Scale raw inputs (..., C, 64, 64) for the model. Returns a new float32 array.

    Fire masks pass through as 0/1. Every other channel is clipped to its train 0.1-99.9 percentile
    range and standardized with the train mean and std. Missing values (NaN) become 0, which is the
    train mean for scaled channels and "no fire" for fire masks.
    """
    x = np.array(inputs, dtype=np.float32, copy=True)
    if x.shape[-3] != len(channels) or x.shape[-2:] != (TILE_SIZE, TILE_SIZE):
        raise ValueError(f"inputs must be (..., {len(channels)}, {TILE_SIZE}, {TILE_SIZE}), got {x.shape}")
    low, high, mean, std, is_fire = scaling_arrays(stats, channels)
    scaled = (np.clip(x, low, high) - mean) / std
    x = np.where(is_fire, x, scaled)
    x[np.isnan(x)] = 0.0
    return x.astype(np.float32, copy=False)


# ----------------------------------------------------------------------------- augmentation
# The 8 symmetries of the square: k in 0..7, bit 0 = transpose rows/cols, bit 1 = flip rows,
# bit 2 = flip columns, applied in that order. Rows increase northward and columns eastward, so:
#   transpose swaps the north and east axes   -> (u, v) becomes (v, u)
#   flip rows mirrors north-south              -> v becomes -v
#   flip columns mirrors east-west             -> u becomes -u
# The wind keeps pointing the way the fire would spread in the transformed picture.

N_TRANSFORMS = 8


def transform(inputs, label=None, k=0, channels=CHANNELS):
    """Apply symmetry k to one tile's inputs (C, H, W) and label (H, W). Returns copies."""
    if not 0 <= k < N_TRANSFORMS:
        raise ValueError(f"k must be 0..7, got {k}")
    x = np.asarray(inputs)
    y = None if label is None else np.asarray(label)
    has_wind = all(name in channels for name in WIND_CHANNELS)
    u_index, v_index = (channels.index("wind_u"), channels.index("wind_v")) if has_wind else (None, None)
    if k & 1:
        x = np.swapaxes(x, -1, -2)
        y = None if y is None else np.swapaxes(y, -1, -2)
    if k & 2:
        x = x[..., ::-1, :]
        y = None if y is None else y[..., ::-1, :]
    if k & 4:
        x = x[..., :, ::-1]
        y = None if y is None else y[..., :, ::-1]
    x = np.array(x, copy=True)
    if has_wind:
        u, v = x[u_index].copy(), x[v_index].copy()
        if k & 1:
            u, v = v, u
        if k & 2:
            v = -v
        if k & 4:
            u = -u
        x[u_index], x[v_index] = u, v
    return (x, None if y is None else np.array(y, copy=True))


def inverse_transform_map(probabilities, k):
    """Undo symmetry k on a (..., H, W) map (e.g. to average test-time-augmented forecasts)."""
    p = np.asarray(probabilities)
    if k & 4:
        p = p[..., :, ::-1]
    if k & 2:
        p = p[..., ::-1, :]
    if k & 1:
        p = np.swapaxes(p, -1, -2)
    return np.array(p, copy=True)


# ----------------------------------------------------------------------------- model and runs


def build_model(channels=CHANNELS):
    return UNet(len(channels))


@torch.no_grad()
def predict(inputs, model, stats, channels=CHANNELS, batch_size=64, device="cpu"):
    """Probability of fire on t+1, (tiles, 64, 64), from raw inputs (tiles, C, 64, 64)."""
    model = model.to(device).eval()
    out = np.empty((len(inputs), TILE_SIZE, TILE_SIZE), dtype=np.float32)
    for start in range(0, len(inputs), batch_size):
        batch = normalize(np.asarray(inputs[start : start + batch_size], dtype=np.float32), stats, channels)
        logits = model(torch.from_numpy(batch).to(device))
        out[start : start + len(batch)] = torch.sigmoid(logits)[:, 0].float().cpu().numpy()
    return out


def save_run(folder, model, stats_path, manifest):
    """Write weights, the stats used and a JSON manifest (must include channels and threshold)."""
    for key in ("channels", "threshold"):
        if key not in manifest:
            raise ValueError(f"manifest needs {key!r}")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, folder / "weights.pt")
    if Path(stats_path).resolve() != (folder / "channel_stats.csv").resolve():
        shutil.copyfile(stats_path, folder / "channel_stats.csv")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return folder


_runs = {}
_runs_lock = threading.Lock()


def load_run(folder):
    """(model in eval mode on CPU, stats, manifest) for a saved run. Cached per folder."""
    folder = Path(folder).resolve()
    with _runs_lock:
        if folder not in _runs:
            manifest = json.loads((folder / "manifest.json").read_text())
            channels = tuple(manifest["channels"])
            unknown = [name for name in channels if name not in CHANNELS]
            if unknown:
                raise ValueError(f"run uses unknown channels {unknown}")
            model = build_model(channels)
            state = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            stats = load_stats(folder / "channel_stats.csv", channels)
            _runs[folder] = (model.eval(), stats, manifest)
        return _runs[folder]


def select_channels(inputs, channels):
    """Pick `channels` (in that order) out of a full-CHANNELS array (..., len(CHANNELS), H, W)."""
    index = [CHANNELS.index(name) for name in channels]
    return np.asarray(inputs)[..., index, :, :]
