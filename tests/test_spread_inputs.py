import datetime as dt

import numpy as np
import pytest

from backend import spread_inputs
from backend.spread_model import INPUT_FEATURES

DAY = dt.date(2020, 8, 20)


@pytest.fixture
def stub_layers(monkeypatch):
    """Each source returns a constant layer equal to its channel number, so order is checkable."""
    calls = {}

    def constant(name):
        return np.full((64, 64), INPUT_FEATURES.index(name), dtype=np.float32)

    def weather_layer(grid, var, days):
        calls.setdefault("weather_days", []).append(list(days))
        return constant(var), {"var": var, "days_used": list(days), "source": "gridMET"}

    monkeypatch.setattr(spread_inputs.terrain, "elevation", lambda grid: constant("elevation"))
    monkeypatch.setattr(spread_inputs.gridmet, "weather_layer", weather_layer)
    monkeypatch.setattr(spread_inputs.gridmet, "pdsi_layer", lambda grid, day: (constant("pdsi"), {"pentad_date": "2020-08-17"}))
    monkeypatch.setattr(
        spread_inputs.fire_points,
        "fire_mask",
        lambda grid, day, **kwargs: (np.ones((64, 64), np.float32), {"source": "MODIS", "n_points": 3, "fallback": False}),
    )
    monkeypatch.setattr(spread_inputs.gridmet, "latest_day", lambda *a, **k: dt.date(2026, 11, 9))
    return calls


def test_channels_in_model_order_and_filled_channels_are_nan(stub_layers):
    built = spread_inputs.build_inputs(38.5, -120.5, DAY)
    assert built.inputs.shape == (12, 64, 64) and built.inputs.dtype == np.float32
    for channel, name in enumerate(INPUT_FEATURES):
        if name in spread_inputs.FILLED_CHANNELS:
            assert np.isnan(built.inputs[channel]).all()
            assert built.layers[name]["status"] == "filled"
        elif name == "PrevFireMask":
            assert (built.inputs[channel] == 1).all()
        else:
            assert (built.inputs[channel] == channel).all(), name
    assert built.batch.shape == (1, 12, 64, 64)
    assert built.forecast_day == DAY + dt.timedelta(days=1)


def test_training_window_is_live_when_available(stub_layers):
    built = spread_inputs.build_inputs(38.5, -120.5, DAY)
    assert stub_layers["weather_days"][0] == [dt.date(2020, 8, 19), DAY]
    assert built.layers["tmmx"]["status"] == "live"
    assert built.layers["tmmx"]["lag_days"] == 0


def test_lagged_weather_when_gridmet_is_behind(stub_layers):
    built = spread_inputs.build_inputs(38.5, -120.5, dt.date(2026, 11, 11))
    assert stub_layers["weather_days"][0] == [dt.date(2026, 11, 8), dt.date(2026, 11, 9)]
    assert built.layers["vs"]["status"] == "lagged"
    assert built.layers["vs"]["lag_days"] == 2


def test_fire_layer_description(stub_layers):
    built = spread_inputs.build_inputs(38.5, -120.5, DAY)
    fire = built.layers["PrevFireMask"]
    assert fire["source"] == "NASA FIRMS MODIS" and fire["family"] == "MODIS" and fire["n_points"] == 3


def test_failed_layer_is_missing_not_invented(stub_layers, monkeypatch):
    def broken(grid, var, days):
        raise RuntimeError("server down")

    monkeypatch.setattr(spread_inputs.gridmet, "weather_layer", broken)
    built = spread_inputs.build_inputs(38.5, -120.5, DAY)
    assert np.isnan(built.inputs[INPUT_FEATURES.index("tmmx")]).all()
    assert built.layers["tmmx"]["status"] == "missing"
    assert "server down" in built.layers["tmmx"]["error"]
    with pytest.raises(spread_inputs.SpreadInputError):
        spread_inputs.build_inputs(38.5, -120.5, DAY, strict=True)


def test_fire_layer_missing_when_firms_unreachable(stub_layers, monkeypatch):
    def unreachable(grid, day, **kwargs):
        raise spread_inputs.fire_points.FirmsError("Could not get today's fire detections.")

    monkeypatch.setattr(spread_inputs.fire_points, "fire_mask", unreachable)
    built = spread_inputs.build_inputs(38.5, -120.5, DAY)
    assert built.layers["PrevFireMask"]["status"] == "missing"
    assert np.isnan(built.inputs[INPUT_FEATURES.index("PrevFireMask")]).all()
    with pytest.raises(spread_inputs.SpreadInputError):
        spread_inputs.build_inputs(38.5, -120.5, DAY, strict=True)
