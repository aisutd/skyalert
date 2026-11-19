"""Read manifest rows into validated, aligned arrays.

Every layer of a sample (radar, label, validity, permanent water) must share CRS, transform and
shape; matching shapes alone are not enough. Invalid pixels are kept and marked, never dropped.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .contract import IGNORE, INPUT_SCHEMA, SIGMA0_MAX

LABEL_VALUES = {0, 1, 2, IGNORE}


class ContractError(ValueError):
    """An input does not match the flood model's input contract."""


def _open(path):
    path = Path(path)
    if not path.exists():
        raise ContractError(f"missing file {path}")
    return rasterio.open(path)


def check_grid(reference, other, name):
    """Raise unless `other` sits on exactly the same grid as `reference`."""
    if other.crs != reference.crs:
        raise ContractError(f"{name}: CRS {other.crs} != {reference.crs}")
    if other.shape != reference.shape:
        raise ContractError(f"{name}: shape {other.shape} != {reference.shape}")
    if not other.transform.almost_equals(reference.transform, precision=1e-6):
        raise ContractError(f"{name}: transform {tuple(other.transform)[:6]} != {tuple(reference.transform)[:6]}")


def check_radar(src):
    """Band names, projected CRS and units of a post-event S1 GRD file."""
    names = [d for d in src.descriptions]
    if src.count != len(INPUT_SCHEMA["bands"]):
        raise ContractError(f"expected {len(INPUT_SCHEMA['bands'])} bands, got {src.count}")
    if any(names) and names != INPUT_SCHEMA["bands"]:
        raise ContractError(f"band order {names} != {INPUT_SCHEMA['bands']}")
    if src.crs is None or not src.crs.is_projected:
        raise ContractError(f"need a projected CRS in metres, got {src.crs}")


def read_radar(path, validity=None):
    """Radar in dB [2, H, W] with NaN where invalid, plus the open dataset's profile."""
    with _open(path) as src:
        check_radar(src)
        x = src.read().astype("float32")
        profile = src.profile.copy()
        tags = src.tags()
    finite = x[np.isfinite(x)]
    if finite.size and (finite.min() < 0 or np.percentile(finite, 99) > SIGMA0_MAX):
        raise ContractError("radar values are not linear sigma0 (negative or very large); was it already in dB?")
    bad = (x <= 0).any(0) | ~np.isfinite(x).all(0)
    if validity is not None:
        bad |= ~validity
    with np.errstate(divide="ignore", invalid="ignore"):
        x = 10 * np.log10(x)
    x[:, bad] = np.nan
    return x, profile, tags


def read_sample(row, root):
    """Aligned arrays for one manifest row.

    Returns dict with x (dB, NaN invalid), y (labels, IGNORE where input or label is invalid),
    permwater (bool), valid (bool), profile.
    """
    root = Path(root)
    with _open(root / row["radar_path"]) as ref:
        grid = ref
        with _open(root / row["label_path"]) as f:
            check_grid(grid, f, "label")
            y = f.read(1)
        with _open(root / row["validity_path"]) as f:
            check_grid(grid, f, "validity")
            validity = f.read(1) == 1
        with _open(root / row["permwater_path"]) as f:
            check_grid(grid, f, "permwater")
            permwater = f.read(1) == 1
    unexpected = set(np.unique(y).tolist()) - LABEL_VALUES
    if unexpected:
        raise ContractError(f"{row['sample_id']}: label values {sorted(unexpected)} outside {sorted(LABEL_VALUES)}")
    x, profile, _ = read_radar(root / row["radar_path"], validity)
    valid = np.isfinite(x).all(0) & (y != IGNORE)
    y = np.where(valid, y, IGNORE).astype("uint8")
    return {"x": x, "y": y, "permwater": permwater, "valid": valid, "profile": profile}


def iter_split(manifest, split, root):
    """One sample at a time, so a whole split never sits in memory as float32."""
    for _, r in manifest[manifest.split == split].iterrows():
        yield dict(read_sample(r, root), sample_id=r["sample_id"], event_id=r["event_id"])


def model_input(x, mean, std, permwater=None):
    """Scaled radar, plus the permanent-water reference as a 0/1 channel when the model takes one."""
    z = normalize(x, mean, std)
    if permwater is not None:
        z = np.concatenate([z, permwater[None].astype("float16")])
    return z


def load_scaled(manifest, split, root, mean, std, permwater_input=False):
    """Compact samples for training/evaluation: model input z as float16 plus labels and masks."""
    out = []
    for s in iter_split(manifest, split, root):
        s["z"] = model_input(s.pop("x"), mean, std, s["permwater"] if permwater_input else None)
        out.append(s)
    return out


def fit_band_stats(samples):
    """Per-band mean and std over valid pixels of training samples only (any iterable of samples)."""
    n = s = ss = 0
    for smp in samples:
        v = np.isfinite(smp["x"]).all(0)
        px = smp["x"][:, v].astype("float64")
        n += px.shape[1]; s += px.sum(1); ss += (px * px).sum(1)
    if n == 0:
        raise ValueError("no valid training pixels to fit normalization")
    mean = s / n
    return mean.astype("float32"), np.sqrt(ss / n - mean**2).astype("float32")


def normalize(x, mean, std):
    """Scaled radar with invalid pixels as 0, float16 to halve memory."""
    z = (x - np.asarray(mean)[:, None, None]) / np.asarray(std)[:, None, None]
    return np.nan_to_num(z, nan=0.0).astype("float16")


def read_manifest(path):
    m = pd.read_csv(path, dtype={"sample_id": str, "event_id": str, "split_group": str})
    required = {"sample_id", "event_id", "split", "split_group", "radar_path", "label_path", "validity_path", "permwater_path"}
    missing = required - set(m.columns)
    if missing:
        raise ContractError(f"manifest missing columns {sorted(missing)}")
    leaks = m.groupby("split_group").split.nunique()
    if (leaks > 1).any():
        raise ContractError(f"split groups in more than one split: {leaks[leaks > 1].index.tolist()[:5]}")
    events = m[m.split != "heldout"].groupby("event_id").split.nunique()
    if (events > 1).any():
        raise ContractError(f"events in more than one split: {events[events > 1].index.tolist()[:5]}")
    return m
