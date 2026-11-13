"""The live v2 forecast (backend/spread_forecast.py)."""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend import spread_forecast as sf
from backend.spread_grid import TileGrid
from backend.spread_model_v2 import CHANNELS

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "retrain"


def test_latest_fire_day_waits_for_the_day_to_complete():
    late = dt.datetime(2026, 11, 13, 16, 0, tzinfo=dt.timezone.utc)
    early = dt.datetime(2026, 11, 13, 2, 0, tzinfo=dt.timezone.utc)
    assert sf.latest_fire_day(late) == dt.date(2026, 11, 12)
    assert sf.latest_fire_day(early) == dt.date(2026, 11, 11)


def test_live_weather_days_match_training_when_available():
    t = dt.date(2026, 11, 12)
    assert sf.live_weather_days(t, latest=dt.date(2026, 11, 11)) == [dt.date(2026, 11, 10), dt.date(2026, 11, 11)]
    # gridMET a day further behind: its latest two days
    assert sf.live_weather_days(t, latest=dt.date(2026, 11, 10)) == [dt.date(2026, 11, 9), dt.date(2026, 11, 10)]


@pytest.fixture
def stubbed(monkeypatch):
    """Every source stubbed: one detection at the tile centre on day t, constant weather."""
    lat, lon = 37.0, -119.0

    def points(south, west, north, east, day, family="VIIRS"):
        rows = [] if day != dt.date(2026, 11, 12) else [{"latitude": lat, "longitude": lon, "satellite": "N20", "frp": 50.0, "scan": 0.4, "track": 0.4}]
        rows.append({"latitude": lat, "longitude": lon, "satellite": "N21", "frp": 99.0, "scan": 0.4, "track": 0.4})  # must be ignored
        return pd.DataFrame(rows)

    flat = lambda value: np.full((64, 64), value, dtype=np.float32)
    monkeypatch.setattr(sf.fire_points, "fire_points", points)
    monkeypatch.setattr(sf.terrain, "elevation", lambda grid: flat(1200.0))
    monkeypatch.setattr(sf._weather, "wind_layers", lambda grid, days: (flat(2.0), flat(-1.0)))
    monkeypatch.setattr(sf._weather, "weather_layer", lambda grid, var, days: flat({"tmmn": 280.0, "tmmx": 295.0, "sph": 0.004, "pr": 0.0, "erc": 60.0}[var]))
    monkeypatch.setattr(sf.gridmet, "pdsi_layer", lambda grid, day: (flat(-2.0), {"pentad_date": "2026-11-06"}))
    monkeypatch.setattr(sf.gridmet, "latest_day", lambda *a, **k: dt.date(2026, 11, 11))
    sf._cache.clear()
    return lat, lon


def test_inputs_in_channel_order_n20_only(stubbed):
    lat, lon = stubbed
    grid, inputs, info = sf.build_inputs(lat, lon, dt.date(2026, 11, 12))
    assert inputs.shape == (len(CHANNELS), 64, 64)
    fire_t = inputs[CHANNELS.index("fire_t")]
    assert fire_t[31, 31] == 1 and inputs[CHANNELS.index("fire_t1")].sum() == 0
    # Only the N20 detection's FRP (50 MW), not the N21 one
    assert np.isclose(inputs[CHANNELS.index("frp_t")][31, 31], np.log1p(50.0))
    assert (inputs[CHANNELS.index("tmmx")] == 295.0).all() and (inputs[CHANNELS.index("wind_v")] == -1.0).all()
    assert info["weather_days"] == ["2026-11-10", "2026-11-11"] and info["detections"]["t"] == 1


def test_forecast_is_north_up_and_complete(stubbed):
    lat, lon = stubbed
    result = sf.forecast(lat, lon, fire_day=dt.date(2026, 11, 12))
    grid = TileGrid.around(lat, lon)
    today = np.array(result["fire_today"])
    # The tile centre pixel [31, 31] (south-up) is row 63 - 31 = 32 when drawn north-up. The 0.4 km
    # footprint can also touch a neighbouring pixel, as in training.
    assert today[32, 31] == 1 and today.sum() <= 4
    rows, cols = np.nonzero(today)
    assert np.abs(rows - 32).max() <= 1 and np.abs(cols - 31).max() <= 1
    assert np.array(result["probability"]).shape == (64, 64)
    assert result["forecast_day"] == "2026-11-13" and result["fire_today_pixels"] == int(today.sum())
    assert result["bounds"]["north"] > result["bounds"]["south"]
    assert np.isclose(result["bounds"]["south"], grid.bounds[0])


def test_failed_fire_detections_refuse_the_forecast(stubbed, monkeypatch):
    def down(*args, **kwargs):
        raise sf.FirmsError("FIRMS is down")

    monkeypatch.setattr(sf.fire_points, "fire_points", down)
    with pytest.raises(sf.ForecastUnavailable, match="FIRMS is down"):
        sf.forecast(37.0, -119.0, fire_day=dt.date(2026, 11, 12))


@pytest.mark.network
def test_live_inputs_match_the_training_tiles():
    """Train/serve parity: live-built inputs reproduce stored dataset tiles (float16 rounding aside)."""
    if not (DATA / "tiles.csv").exists():
        pytest.skip("needs data/retrain")
    tiles = pd.read_csv(DATA / "tiles.csv")
    stored_inputs = {split: np.load(DATA / f"{split}_inputs.npy", mmap_mode="r") for split in ("val", "test")}
    sample = tiles[tiles.split.isin(["val", "test"]) & (tiles.stratum == "growing")].sample(2, random_state=7)
    # float16 storage steps: ~0.125 at 280 K, 1 m at ~1500 m
    tolerance = {"elevation": 2.0, "tmmn": 0.25, "tmmx": 0.25, "erc": 0.07, "frp_t": 0.01}
    for row in sample.itertuples():
        t = dt.date.fromisoformat(row.fire_day)
        _, live, _ = sf.build_inputs(row.center_lat, row.center_lon, t, weather_days=[t - dt.timedelta(days=2), t - dt.timedelta(days=1)])
        stored = np.asarray(stored_inputs[row.split][row.index], dtype=np.float32)
        for channel, name in enumerate(CHANNELS):
            assert np.array_equal(np.isnan(live[channel]), np.isnan(stored[channel])), name
            finite = np.isfinite(stored[channel])
            assert np.abs(live[channel][finite] - stored[channel][finite]).max() <= tolerance.get(name, 0.005), name
