import numpy as np
import pandas as pd
import pytest
import torch

from backend import spread_model_v2 as v2

C = len(v2.CHANNELS)
U, V = v2.CHANNELS.index("wind_u"), v2.CHANNELS.index("wind_v")
FIRE = v2.CHANNELS.index("fire_t")


def test_channel_contract_is_unchanged():
    assert v2.CHANNELS == (
        "fire_t", "fire_t1", "fire_t2", "frp_t", "elevation", "wind_u", "wind_v",
        "tmmn", "tmmx", "sph", "pr", "erc", "pdsi",
    )
    assert v2.WEATHER_OFFSETS == (-2, -1) and v2.VIIRS_SATELLITES == ("N20",)


def write_stats(path, channels=v2.CHANNELS, low=-10.0, high=10.0, mean=1.0, std=2.0):
    rows = [{"channel": name, "low": low, "high": high, "mean": mean, "std": std} for name in channels]
    pd.DataFrame(rows).set_index("channel").to_csv(path)
    return path


def test_normalize_fire_passthrough_clip_standardize_and_nan(tmp_path):
    stats = v2.load_stats(write_stats(tmp_path / "s.csv"))
    x = np.zeros((1, C, 64, 64), dtype=np.float32)
    x[0, FIRE, 0, 0] = 1
    x[0, FIRE, 0, 1] = np.nan
    x[0, U, 0, 0] = 5.0  # (5 - 1) / 2 = 2
    x[0, U, 0, 1] = 100.0  # clipped to 10 -> 4.5
    x[0, U, 0, 2] = np.nan  # -> 0 (train mean)
    out = v2.normalize(x, stats)
    assert out[0, FIRE, 0, 0] == 1 and out[0, FIRE, 0, 1] == 0 and out[0, FIRE, 1, 1] == 0
    assert out[0, U, 0, 0] == pytest.approx(2.0) and out[0, U, 0, 1] == pytest.approx(4.5) and out[0, U, 0, 2] == 0
    assert out[0, U, 5, 5] == pytest.approx(-0.5)
    assert np.isnan(x[0, U, 0, 2]), "normalize must not change its input"
    assert not np.isnan(out).any()


def test_fire_channels_ignore_stats_file(tmp_path):
    stats = v2.load_stats(write_stats(tmp_path / "s.csv", mean=5.0, std=3.0))
    assert (stats.loc["fire_t"] == [0.0, 1.0, 0.0, 1.0]).all()


def spreading_tile(east, north, shift_east, shift_north):
    """A fire blob and tomorrow's fire moved (shift_north rows, shift_east cols) downwind."""
    x = np.zeros((C, 64, 64), dtype=np.float32)
    x[FIRE, 28:34, 26:32] = 1
    x[U], x[V] = east, north
    y = np.zeros((64, 64), dtype=np.float32)
    y[28 + shift_north : 34 + shift_north, 26 + shift_east : 32 + shift_east] = 1
    return x, y


def centroid(mask):
    rows, cols = np.nonzero(mask > 0.5)
    return rows.mean(), cols.mean()


@pytest.mark.parametrize("wind", [(3.0, 0.0, 3, 0), (0.0, -2.0, 0, -2), (2.0, 1.0, 4, 2), (-1.0, 3.0, -1, 3)])
@pytest.mark.parametrize("k", range(8))
def test_transformed_wind_still_points_along_the_spread(wind, k):
    east, north, shift_east, shift_north = wind
    x, y = spreading_tile(east, north, shift_east, shift_north)
    tx, ty = v2.transform(x, y, k)
    r0, c0 = centroid(tx[FIRE])
    r1, c1 = centroid(ty)
    spread = np.array([c1 - c0, r1 - r0])  # (east, north): rows increase northward
    w = np.array([tx[U, 0, 0], tx[V, 0, 0]])
    assert np.linalg.norm(w) == pytest.approx(np.hypot(east, north))
    assert np.allclose(w / np.linalg.norm(w), spread / np.linalg.norm(spread), atol=1e-6)


def test_eight_distinct_transforms_and_inverse():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(C, 64, 64)).astype(np.float32)
    seen = set()
    for k in range(8):
        tx, _ = v2.transform(x, None, k)
        seen.add(tx[FIRE].tobytes())
        assert np.array_equal(v2.inverse_transform_map(tx[FIRE], k), x[FIRE])
        assert np.array_equal(v2.inverse_transform_map(tx[FIRE][None], k)[0], x[FIRE])
    assert len(seen) == 8
    with pytest.raises(ValueError):
        v2.transform(x, None, 8)


def test_transform_without_wind_channels_works_on_subsets():
    x = np.arange(2 * 64 * 64, dtype=np.float32).reshape(2, 64, 64)
    tx, _ = v2.transform(x, None, 5, channels=("fire_t", "elevation"))
    assert tx.shape == x.shape


def test_save_and_load_run_round_trip(tmp_path):
    stats_path = write_stats(tmp_path / "channel_stats.csv")
    channels = CHANNELS_NO_HISTORY = tuple(n for n in v2.CHANNELS if n not in ("fire_t1", "fire_t2"))
    torch.manual_seed(0)
    model = v2.build_model(channels).eval()
    stats = v2.load_stats(stats_path, channels)
    x = np.random.default_rng(1).normal(size=(3, len(channels), 64, 64)).astype(np.float32)
    before = v2.predict(x, model, stats, channels)
    folder = v2.save_run(tmp_path / "run", model, stats_path, {"channels": list(channels), "threshold": 0.4})
    loaded, loaded_stats, manifest = v2.load_run(folder)
    after = v2.predict(x, loaded, loaded_stats, tuple(manifest["channels"]))
    assert manifest["threshold"] == 0.4 and tuple(manifest["channels"]) == CHANNELS_NO_HISTORY
    assert np.allclose(before, after, atol=1e-6)
    assert before.shape == (3, 64, 64) and (before >= 0).all() and (before <= 1).all()
    with pytest.raises(ValueError):
        v2.save_run(tmp_path / "bad", model, stats_path, {"channels": list(channels)})


def test_select_channels():
    x = np.arange(C, dtype=np.float32)[None, :, None, None] * np.ones((1, C, 64, 64), np.float32)
    out = v2.select_channels(x, ("pdsi", "fire_t"))
    assert out[0, 0, 0, 0] == v2.CHANNELS.index("pdsi") and out[0, 1, 0, 0] == 0


def make_dataset(folder, n_train=24, n_val=12, seed=0):
    """A tiny synthetic dataset in the real file format: fire spreads one pixel downwind."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", n_train), ("val", n_val), ("test", 4)):
        x = np.zeros((n, C, 64, 64), dtype=np.float16)
        y = np.zeros((n, 64, 64), dtype=np.uint8)
        for i in range(n):
            r, c = rng.integers(10, 50, size=2)
            x[i, FIRE, r : r + 4, c : c + 4] = 1
            x[i, U], x[i, V] = 2.0, 0.0
            x[i, v2.CHANNELS.index("tmmx")] = 300
            y[i, r : r + 4, c : c + 5] = 1
        np.save(folder / f"{split}_inputs.npy", x)
        np.save(folder / f"{split}_labels.npy", y)
    write_stats(folder / "channel_stats.csv")
    return folder


def test_training_and_evaluation_end_to_end(tmp_path):
    from training import evaluate_v2, train_spread_v2

    data = make_dataset(tmp_path / "data")
    args = train_spread_v2.parse_args([
        "--name", "smoke", "--data-dir", str(data), "--runs-dir", str(tmp_path / "runs"),
        "--max-epochs", "2", "--batch-size", "8", "--cpu", "--max-minutes", "5",
    ])
    summary = train_spread_v2.train(args)
    assert summary["epochs_run"] >= 1 and summary["peak_memory_gb"] < train_spread_v2.MAX_MEMORY_GB
    rows = evaluate_v2.evaluate_split(tmp_path / "runs" / "smoke", "val", data_dir=data, reps=50)
    names = [row["forecast"] for row in rows]
    assert names[0].startswith("model") and "persistence" in names and "downwind 2" in names
    with pytest.raises(SystemExit):
        evaluate_v2.evaluate_split(tmp_path / "runs" / "smoke", "test", data_dir=data)


def test_training_refuses_unsafe_settings(tmp_path):
    from training import train_spread_v2

    data = make_dataset(tmp_path / "data")
    for flag, value in (("--batch-size", "128"), ("--max-minutes", "45")):
        args = train_spread_v2.parse_args(["--name", "x", "--data-dir", str(data), "--runs-dir", str(tmp_path), "--cpu", flag, value])
        with pytest.raises(SystemExit):
            train_spread_v2.train(args)


def test_evaluation_reports_each_stratum(tmp_path):
    from training import evaluate_v2, train_spread_v2

    data = make_dataset(tmp_path / "data", n_val=12)
    rows = [{"split": split, "index": i, "stratum": "growing" if i % 3 == 0 else "small", "region": "West-N"}
            for split, n in (("train", 24), ("val", 12), ("test", 4)) for i in range(n)]
    pd.DataFrame(rows).to_csv(data / "tiles.csv", index=False)
    args = train_spread_v2.parse_args(["--name", "s", "--data-dir", str(data), "--runs-dir", str(tmp_path / "runs"),
                                       "--max-epochs", "1", "--batch-size", "8", "--cpu"])
    train_spread_v2.train(args)
    out = evaluate_v2.evaluate_split(tmp_path / "runs" / "s", "val", data_dir=data, reps=20, by=("region",))
    groups = {(r["group"], r["tiles"]) for r in out}
    assert ("all", 12) in groups and ("stratum=growing", 4) in groups and ("stratum=small", 8) in groups
    assert ("region=West-N", 12) in groups
    assert not any(g == evaluate_v2.LIVE_MIX for g, _ in groups)  # no sample_prob column here
    pd.DataFrame([{"split": "val", "index": 0}]).to_csv(data / "tiles.csv", index=False)
    with pytest.raises(ValueError):
        evaluate_v2.evaluate_split(tmp_path / "runs" / "s", "val", data_dir=data, reps=20)


def test_collapsed_clip_range_fails_loudly(tmp_path):
    path = write_stats(tmp_path / "s.csv")
    table = pd.read_csv(path, index_col=0)
    table.loc["frp_t", ["low", "high"]] = 0.0
    table.to_csv(path)
    with pytest.raises(ValueError, match="frp_t"):
        v2.load_stats(path)
