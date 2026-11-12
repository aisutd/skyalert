import datetime as dt

import numpy as np
import pandas as pd
import pytest

from backend.spread_grid import PIXEL_DEG, TileGrid
from backend.spread_model_v2 import CHANNELS
from training import build_spread_dataset as builder


def points(rows):
    return pd.DataFrame(rows, columns=["latitude", "longitude", "frp", "scan", "track", "type"])


def with_neighbours(fires):
    """Make t-2..t+1 exist for every day (an offshore type-3 detection: present, but never a centre)."""
    out = dict(fires)
    for day in list(fires):
        t = dt.date.fromisoformat(day)
        for k in (-2, -1, 1):
            out.setdefault((t + dt.timedelta(days=k)).isoformat(), points([(30.0, -130.0, 1, 0.5, 0.5, 3)]))
    return out


def test_cluster_centres_keep_strongest_and_dedup():
    table = points([
        (38.00, -120.00, 5.0, 0.5, 0.5, 0),
        (38.10, -120.00, 50.0, 0.5, 0.5, 0),  # 11 km away, stronger: chosen, the first dropped
        (38.60, -120.00, 1.0, 0.5, 0.5, 0),  # 56 km from the strongest: its own cluster
        (np.nan, -120.0, 1.0, 0.5, 0.5, 0),
    ]).dropna(subset=["latitude"])
    centres = builder.cluster_centres(table, dedup_km=40)
    assert centres[0] == (38.10, -120.00)
    assert len(centres) == 2 and centres[1] == (38.60, -120.00)
    assert builder.cluster_centres(points([])) == []


def test_split_by_year_and_season():
    assert builder.split_of(dt.date(2018, 6, 1)) == "train"
    assert builder.split_of(dt.date(2022, 10, 1)) == "train"
    assert builder.split_of(dt.date(2023, 7, 1)) == "val"
    assert builder.split_of(dt.date(2024, 7, 1)) == "test"
    assert builder.split_of(dt.date(2025, 7, 1)) == "test"
    assert builder.split_of(dt.date(2017, 7, 1)) is None
    days = builder.season_days(2020)
    assert days[0] == dt.date(2020, 4, 1) and days[-1] == dt.date(2020, 11, 30)


def test_plan_offsets_regions_and_targets():
    fires = {
        "2020-07-01": points([(38.0, -120.0, 10, 0.5, 0.5, 0), (39.0, -100.0, 10, 0.5, 0.5, 0), (33.0, -85.0, 3, 0.5, 0.5, 2)]),
        "2023-07-01": points([(40.0, -110.0, 10, 0.5, 0.5, 0)]),
        "2024-07-01": points([(35.0, -95.0, 10, 0.5, 0.5, 0)]),
        "2020-01-15": points([(38.0, -120.0, 10, 0.5, 0.5, 0)]),  # out of season
    }
    plan, counts = builder.plan_tiles(with_neighbours(fires), {"train": 10, "val": 10, "test": 10}, np.random.default_rng(0))
    # the type-2 (static source) point is not a centre; single small fires are "small"
    assert counts == {"train": {"growing": 0, "small": 2}, "val": {"growing": 0, "small": 1}, "test": {"growing": 0, "small": 1}}
    assert set(plan["split"]) == {"train", "val", "test"}
    assert plan.groupby("split")["index"].apply(lambda s: list(s) == list(range(len(s)))).all()
    assert plan[["offset_rows", "offset_cols"]].abs().max().max() <= builder.MAX_OFFSET_PX
    np.testing.assert_allclose(plan["center_lat"], plan["fire_lat"] + plan["offset_rows"] * PIXEL_DEG)
    # The clicked fire stays inside the tile, within 16 px of the centre pixel
    for row in plan.itertuples():
        grid = TileGrid.around(row.center_lat, row.center_lon)
        rows, cols = grid.cell_of([row.fire_lat], [row.fire_lon])
        assert abs(rows[0] - 31) <= 17 and abs(cols[0] - 31) <= 17
    assert dict(zip(plan["fire_day"], plan["region"]))["2020-07-01"] in {"West-N", "Central-N"}


def test_plan_samples_down_to_target():
    fires = {f"2020-07-{d:02d}": points([(30 + d * 0.5, -120.0, 1, 0.5, 0.5, 0)]) for d in range(1, 21)}
    plan, counts = builder.plan_tiles(with_neighbours(fires), {"train": 5, "val": 5, "test": 5}, np.random.default_rng(1))
    assert sum(counts["train"].values()) == 20 and (plan["split"] == "train").sum() == 5


def test_frp_layer_sums_per_pixel():
    grid = TileGrid.around(38.5, -120.5)
    lat, lon = grid.lats[10, 20], grid.lons[10, 20]
    table = points([(lat, lon, 3.0, 0.5, 0.5, 0), (lat, lon, 4.0, 0.5, 0.5, 0), (grid.lats[5, 5], grid.lons[5, 5], np.nan, 0.5, 0.5, 0)])
    layer = builder.frp_layer(grid, table)
    assert np.isclose(layer[10, 20], np.log1p(7.0)) and layer[5, 5] == 0 and layer.sum() == pytest.approx(np.log1p(7.0))


class FakeBulk:
    """Weather layers equal to each channel's position, so channel order is checkable."""

    def weather_layer(self, grid, var, days):
        assert days == [dt.date(2020, 6, 28), dt.date(2020, 6, 29)], days  # t-2, t-1
        return np.full((64, 64), CHANNELS.index(var), np.float32)

    def wind_layers(self, grid, days):
        return np.full((64, 64), CHANNELS.index("wind_u"), np.float32), np.full((64, 64), CHANNELS.index("wind_v"), np.float32)

    def pdsi_layer(self, grid, day):
        return np.full((64, 64), CHANNELS.index("pdsi"), np.float32), day


def test_build_tile_channels_in_order(monkeypatch):
    monkeypatch.setattr(builder.terrain, "elevation", lambda grid: np.full((64, 64), CHANNELS.index("elevation"), np.float32))
    grid = TileGrid.around(38.5, -120.5)
    spot = (grid.lats[31, 31], grid.lons[31, 31])
    fires = {
        "2020-06-30": points([(*spot, 9.0, 0.5, 0.5, 0)]),
        "2020-06-29": points([(grid.lats[30, 31], grid.lons[30, 31], 1.0, 0.5, 0.5, 0)]),
        "2020-07-01": points([(grid.lats[33, 31], grid.lons[33, 31], 1.0, 0.5, 0.5, 0)]),
    }
    row = pd.Series({"center_lat": 38.5, "center_lon": -120.5, "fire_day": "2020-06-30"})
    inputs, label = builder.build_tile(row, fires, FakeBulk())
    assert inputs.shape == (len(CHANNELS), 64, 64) and label.shape == (64, 64) and label.dtype == np.uint8
    for channel, name in enumerate(CHANNELS):
        if name not in ("fire_t", "fire_t1", "fire_t2", "frp_t"):
            assert (inputs[channel] == channel).all(), name
    assert inputs[CHANNELS.index("fire_t"), 31, 31] == 1
    assert inputs[CHANNELS.index("fire_t1"), 30, 31] == 1 and inputs[CHANNELS.index("fire_t2")].sum() == 0
    assert label[33, 31] == 1 and inputs[CHANNELS.index("frp_t"), 31, 31] == pytest.approx(np.log1p(9.0))


def test_tiny_build_writes_the_documented_format(monkeypatch, tmp_path):
    ramp = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    monkeypatch.setattr(builder.terrain, "elevation", lambda grid: 400.0 + 200.0 * ramp)
    monkeypatch.setattr(builder, "Bulk", FakeBulkAnyDay)
    fires = {f"2020-06-{d:02d}": points([(38.0 + d * 0.6, -120.0, 5, 0.5, 0.5, 0)]) for d in range(1, 30)}
    fires.update({f"2023-06-{d:02d}": points([(38.0 + d * 0.6, -110.0, 5, 0.5, 0.5, 0)]) for d in range(1, 10)})
    fires.update({f"2024-06-{d:02d}": points([(38.0 + d * 0.6, -100.0, 5, 0.5, 0.5, 0)]) for d in range(1, 10)})
    monkeypatch.setattr(builder, "load_firms_year", lambda year: {k: v for k, v in fires.items() if k.startswith(str(year))})
    builder.build(tmp_path, {"train": 6, "val": 3, "test": 3})
    for split, n in (("train", 6), ("val", 3), ("test", 3)):
        inputs = np.load(tmp_path / f"{split}_inputs.npy", mmap_mode="r")
        labels = np.load(tmp_path / f"{split}_labels.npy", mmap_mode="r")
        assert inputs.shape == (n, len(CHANNELS), 64, 64) and inputs.dtype == np.float16
        assert labels.shape == (n, 64, 64) and labels.dtype == np.uint8
        assert (np.asarray(inputs[:, CHANNELS.index("fire_t")]).sum((1, 2)) >= 1).all(), "every tile has today's fire"
    tiles = pd.read_csv(tmp_path / "tiles.csv")
    assert list(tiles.columns[:9]) == ["split", "index", "fire_day", "center_lat", "center_lon", "fire_t_px", "label_px", "new_fire_px", "region"]
    stats = pd.read_csv(tmp_path / "channel_stats.csv", index_col="feature")
    assert list(stats.index) == list(CHANNELS)
    assert stats.loc["elevation", "mean"] == pytest.approx(500.0, abs=1.0)
    assert (stats["low"] < stats["high"]).all()
    # Resuming does nothing new and keeps the files
    builder.build(tmp_path, {"train": 6, "val": 3, "test": 3})
    assert np.load(tmp_path / "train_done.npy").all()


class FakeBulkAnyDay(FakeBulk):
    def year_file(self, var, year):
        class Plane:  # everything inside gridMET
            lats = 49.4 - np.arange(585) / 24
            lons = -124.7667 + np.arange(1386) / 24
            fills = (32767.0,)

            def raw(self, *args):
                return np.zeros((585, 1386))

        return Plane()

    def weather_layer(self, grid, var, days):
        ramp = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
        return (280.0 if var.startswith("tm") else 1.0) + ramp

    def wind_layers(self, grid, days):
        ramp = np.linspace(-1, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
        return ramp, -ramp

    def pdsi_layer(self, grid, day):
        return np.linspace(-2, 2, 64 * 64, dtype=np.float32).reshape(64, 64), day


def test_centres_outside_gridmet_are_dropped():
    fires = {"2020-07-01": points([(38.0, -120.0, 10, 0.5, 0.5, 0), (28.0, -105.0, 10, 0.5, 0.5, 0)])}
    plan, counts = builder.plan_tiles(with_neighbours(fires), {"train": 10, "val": 1, "test": 1}, np.random.default_rng(0), covered=lambda lat, lon: lat > 30)
    assert counts["train"] == {"growing": 0, "small": 1} and plan["fire_lat"].tolist() == [38.0]


def test_tiles_touching_a_data_gap_are_skipped():
    day = lambda d: points([(38.0 + d * 0.01, -120.0, 5, 0.5, 0.5, 0)])
    fires = {f"2020-07-{d:02d}": day(d) for d in range(1, 15) if d != 8}  # 2020-07-08 missing
    plan, counts = builder.plan_tiles(fires, {"train": 100, "val": 1, "test": 1}, np.random.default_rng(0))
    used = set(plan["fire_day"])
    # t = 07-07 (label day missing), 07-09 (t-1 missing) and 07-10 (t-2 missing) are dropped
    assert not used & {"2020-07-07", "2020-07-09", "2020-07-10"}
    # So are the first two days (no t-1/t-2) and the last (no t+1)
    assert not used & {"2020-07-01", "2020-07-02", "2020-07-14"}
    assert "2020-07-05" in used


def test_channel_stats_keep_sparse_channels():
    rng = np.random.default_rng(0)
    x = np.zeros((50, len(CHANNELS), 64, 64), dtype=np.float32)
    for channel, name in enumerate(CHANNELS):
        if name not in builder.MASK_CHANNELS + builder.SPARSE_CHANNELS:
            x[:, channel] = rng.normal(10 * (channel + 1), 3, size=(50, 64, 64))
    x[:, CHANNELS.index("fire_t"), 30, 30] = 1  # 1 pixel in 4096: far below 0.1%
    x[:3, CHANNELS.index("frp_t"), 30, 30] = [1.0, 2.0, 3.0]
    stats = builder.channel_stats(x, rng)
    assert (stats["low"] < stats["high"]).all()
    for name in builder.MASK_CHANNELS:
        assert (stats.loc[name, "low"], stats.loc[name, "high"]) == (0.0, 1.0)
    assert stats.loc["frp_t", "low"] == 0 and 2.9 < stats.loc["frp_t", "high"] <= 3.0
    # A channel that is constant can't be scaled: the builder must refuse it
    x[:, CHANNELS.index("tmmx")] = 280.0
    with pytest.raises(ValueError):
        builder.channel_stats(x, rng)


def test_stratum_rule():
    assert builder.stratum_of(10, 0) == "growing" and builder.stratum_of(1, 3) == "growing"
    assert builder.stratum_of(9, 2) == "small"


def test_cluster_features_measure_size_and_persistence():
    grid_step = PIXEL_DEG
    # 12 day-t cells in a row (12 km), and day t-1 fire on 4 cells next to them plus 1 far away
    today = points([(38.0, -120.0 + i * grid_step * 1.01, 5, 0.5, 0.5, 0) for i in range(12)])
    yesterday = points([(38.0 + 2 * grid_step, -120.0 + i * grid_step * 1.01, 5, 0.5, 0.5, 0) for i in range(4)]
                       + [(38.3, -120.0, 5, 0.5, 0.5, 0)])
    px_t, persist = builder.cluster_features(builder.DayIndex(today), builder.DayIndex(yesterday), 38.0, -119.95)
    assert px_t == 12 and persist == 4


def test_plan_is_half_growing_and_records_weights():
    fires = {}
    for d in range(1, 21):
        big = [(30 + d * 0.5, -110.0 + i * PIXEL_DEG * 1.01, 9, 0.5, 0.5, 0) for i in range(12)]  # growing
        small = [(30 + d * 0.5 + 0.0, -100.0 - k, 1, 0.5, 0.5, 0) for k in range(3)]  # 3 small fires
        fires[f"2020-07-{d:02d}"] = points(big + small)
    plan, counts = builder.plan_tiles(with_neighbours(fires), {"train": 10, "val": 1, "test": 1}, np.random.default_rng(0))
    train = plan[plan.split == "train"]
    assert counts["train"]["growing"] == 20 and counts["train"]["small"] == 60  # with_neighbours makes all 20 days valid
    assert (train.stratum == "growing").sum() == 5 and (train.stratum == "small").sum() == 5
    g = train[train.stratum == "growing"].iloc[0]
    assert g.stratum_candidates == 20 and g.sample_prob == pytest.approx(5 / 20)


def test_plan_tops_up_when_a_stratum_runs_short():
    fires = {f"2020-07-{d:02d}": points([(30 + d * 0.5, -100.0, 1, 0.5, 0.5, 0)]) for d in range(1, 21)}  # all small
    plan, counts = builder.plan_tiles(with_neighbours(fires), {"train": 6, "val": 1, "test": 1}, np.random.default_rng(0))
    assert (plan.split == "train").sum() == 6 and set(plan.stratum) == {"small"}
