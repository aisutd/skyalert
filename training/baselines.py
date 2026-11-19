"""Simple rules the v2 model has to beat, all on v2 tiles (backend/spread_model_v2.CHANNELS).

Orientation: rows increase northward (south-up tiles), columns increase eastward; wind_u is the
eastward and wind_v the northward wind component (m/s), i.e. (u, v) points DOWNWIND.

- persistence: tomorrow's fire = today's fire (fire_t).
- dilated r: today's fire grown by r pixels in every direction (8-neighbour, so a (2r+1) square).
- downwind r: today's fire plus copies of it shifted 1..r pixels along the tile's downwind unit
  vector, rounded to the nearest pixel. The tile's wind is the mean (u, v) over today's fire
  pixels; below MIN_WIND_SPEED m/s there is no clear direction and it stays persistence.
- old NDWS model: backend/spread_model fed v2 inputs mapped to its 12 channels (reference only;
  the mapping is approximate, see old_model_inputs).
"""

import warnings

import numpy as np

from backend import spread_model_v2 as v2

MIN_WIND_SPEED = 0.5  # m/s
CHUNK = 512  # tiles per chunk when reading memory-mapped inputs


def _channel(inputs, name):
    return np.asarray(inputs[:, v2.CHANNELS.index(name)], dtype=np.float32)


def shift(mask, rows, cols):
    """Move a (..., H, W) boolean mask by (rows, cols) pixels, filling with False (no wrap)."""
    out = np.zeros_like(mask)
    h, w = mask.shape[-2:]
    if abs(rows) >= h or abs(cols) >= w:
        return out
    src_r = slice(max(0, -rows), h - max(0, rows))
    dst_r = slice(max(0, rows), h - max(0, -rows))
    src_c = slice(max(0, -cols), w - max(0, cols))
    dst_c = slice(max(0, cols), w - max(0, -cols))
    out[..., dst_r, dst_c] = mask[..., src_r, src_c]
    return out


def persistence(fire_t):
    return np.asarray(fire_t) > 0.5


def dilate(mask, radius):
    """Grow a boolean mask by `radius` pixels with the 8-neighbourhood (a (2r+1) x (2r+1) square)."""
    out = np.asarray(mask, dtype=bool)
    for _ in range(radius):
        grown = out.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    grown |= shift(out, dr, dc)
        out = grown
    return out


def downwind_vectors(fire_t, wind_u, wind_v):
    """Per tile: downwind unit vector (east, north) from the mean wind over today's fire pixels,
    or (0, 0) when there is no fire, no wind data, or the wind is below MIN_WIND_SPEED."""
    fire = persistence(fire_t)
    n = fire.sum(axis=(1, 2))
    u = np.where(fire, wind_u, np.nan)
    v = np.where(fire, wind_v, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # tiles without fire: all-NaN mean
        mean_u = np.nanmean(u, axis=(1, 2))
        mean_v = np.nanmean(v, axis=(1, 2))
    speed = np.hypot(mean_u, mean_v)
    ok = (n > 0) & np.isfinite(speed) & (speed >= MIN_WIND_SPEED)
    east = np.where(ok, mean_u / np.where(ok, speed, 1), 0.0)
    north = np.where(ok, mean_v / np.where(ok, speed, 1), 0.0)
    return east, north


def downwind(fire_t, wind_u, wind_v, steps):
    """Today's fire plus copies shifted 1..steps pixels downwind (see module docstring)."""
    fire = persistence(fire_t)
    east, north = downwind_vectors(fire_t, wind_u, wind_v)
    out = fire.copy()
    for i in range(len(fire)):
        if east[i] == 0 and north[i] == 0:
            continue
        for k in range(1, steps + 1):
            # rows increase northward, columns eastward
            out[i] |= shift(fire[i], int(np.rint(k * north[i])), int(np.rint(k * east[i])))
    return out


def old_model_inputs(inputs):
    """Map v2 tiles (n, len(CHANNELS), 64, 64) to the old NDWS model's 12 channels.

    Approximate, for reference only: wind direction and speed come from the averaged (u, v)
    vector (the NDWS export averaged degrees and speeds as plain numbers), the weather days are
    t-2..t-1 (NDWS: t-1..t), fire is VIIRS NOAA-20 (NDWS: MODIS Terra), and NDVI and population
    are missing (NaN = train mean).
    """
    from backend.spread_model import INPUT_FEATURES

    x = np.asarray(inputs, dtype=np.float32)
    u, v = x[:, v2.CHANNELS.index("wind_u")], x[:, v2.CHANNELS.index("wind_v")]
    mapped = {
        "elevation": x[:, v2.CHANNELS.index("elevation")],
        # direction the wind blows FROM, degrees clockwise from north
        "th": (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0,
        "vs": np.hypot(u, v),
        "NDVI": np.full(u.shape, np.nan, dtype=np.float32),
        "population": np.full(u.shape, np.nan, dtype=np.float32),
        "PrevFireMask": x[:, v2.CHANNELS.index("fire_t")],
    }
    for name in ("tmmn", "tmmx", "sph", "pr", "pdsi", "erc"):
        mapped[name] = x[:, v2.CHANNELS.index(name)]
    return np.stack([mapped[name] for name in INPUT_FEATURES], axis=1).astype(np.float32)


def old_model_forecast(inputs):
    """The old NDWS model's forecast at its own threshold (0.621), or None if its files are missing."""
    from backend import spread_model

    if not (spread_model.WEIGHTS_PATH.exists() and spread_model.STATS_PATH.exists()):
        return None
    out = np.empty((len(inputs), v2.TILE_SIZE, v2.TILE_SIZE), dtype=bool)
    for start in range(0, len(inputs), CHUNK):
        chunk = old_model_inputs(np.asarray(inputs[start : start + CHUNK], dtype=np.float32))
        out[start : start + len(chunk)] = spread_model.predict(chunk) >= spread_model.THRESHOLD
    return out


def all_baselines(inputs, include_old_model=True):
    """Every baseline's forecast (bool, (n, 64, 64)) for v2 inputs, reading in chunks."""
    names = ("persistence", "dilated 1", "dilated 2", "downwind 1", "downwind 2")
    n = len(inputs)
    out = {name: np.empty((n, v2.TILE_SIZE, v2.TILE_SIZE), dtype=bool) for name in names}
    for start in range(0, n, CHUNK):
        chunk = np.asarray(inputs[start : start + CHUNK], dtype=np.float32)
        fire_t = _channel(chunk, "fire_t")
        u, v = _channel(chunk, "wind_u"), _channel(chunk, "wind_v")
        end = start + len(chunk)
        out["persistence"][start:end] = persistence(fire_t)
        out["dilated 1"][start:end] = dilate(fire_t > 0.5, 1)
        out["dilated 2"][start:end] = dilate(fire_t > 0.5, 2)
        out["downwind 1"][start:end] = downwind(fire_t, u, v, 1)
        out["downwind 2"][start:end] = downwind(fire_t, u, v, 2)
    if include_old_model:
        old = old_model_forecast(inputs)
        if old is not None:
            out["old NDWS model (approx. inputs)"] = old
    return out
