import numpy as np
import pytest
import torch

from backend import spread_model_v2 as v2
from training import baselines, evaluate_v2

C = len(v2.CHANNELS)


def one_pixel(r=30, c=30, n=1):
    mask = np.zeros((n, 64, 64), dtype=bool)
    mask[:, r, c] = True
    return mask


def test_shift_moves_without_wrapping():
    m = one_pixel()[0]
    assert baselines.shift(m, 2, -3)[32, 27] and baselines.shift(m, 2, -3).sum() == 1
    assert baselines.shift(m, 40, 0).sum() == 0
    edge = np.zeros((64, 64), bool)
    edge[63, 0] = True
    assert baselines.shift(edge, 1, 0).sum() == 0


def test_dilate_is_a_square():
    assert baselines.dilate(one_pixel(), 1).sum() == 9
    assert baselines.dilate(one_pixel(), 2).sum() == 25
    assert baselines.dilate(one_pixel(0, 0), 1).sum() == 4  # clipped at the corner


@pytest.mark.parametrize(
    "u,v,expected",
    [
        (3.0, 0.0, [(30, 30), (30, 31), (30, 32)]),  # wind toward east -> columns increase
        (0.0, 3.0, [(30, 30), (31, 30), (32, 30)]),  # toward north -> rows increase (south-up)
        (0.0, -3.0, [(30, 30), (29, 30), (28, 30)]),
        (2.0, 2.0, [(30, 30), (31, 31)]),  # diagonal: steps 1 and 2 both round to (1, 1): rint(0.707) = rint(1.414) = 1
    ],
)
def test_downwind_growth(u, v, expected):
    fire = one_pixel().astype(np.float32)
    out = baselines.downwind(fire, np.full(fire.shape, u, np.float32), np.full(fire.shape, v, np.float32), 2)
    marked = set(zip(*np.nonzero(out[0])))
    assert marked == set(expected)


def test_calm_or_missing_wind_is_persistence():
    fire = one_pixel().astype(np.float32)
    calm = baselines.downwind(fire, np.full(fire.shape, 0.2, np.float32), np.zeros(fire.shape, np.float32), 2)
    nan = baselines.downwind(fire, np.full(fire.shape, np.nan, np.float32), np.full(fire.shape, np.nan, np.float32), 2)
    assert calm.sum() == 1 and nan.sum() == 1
    empty = baselines.downwind(np.zeros((1, 64, 64), np.float32), np.ones((1, 64, 64), np.float32), np.ones((1, 64, 64), np.float32), 2)
    assert empty.sum() == 0


def test_old_model_wind_direction_is_from_direction():
    x = np.zeros((2, C, 64, 64), dtype=np.float32)
    x[0, v2.CHANNELS.index("wind_v")] = -5.0  # blowing south = wind FROM the north
    x[1, v2.CHANNELS.index("wind_u")] = -5.0  # blowing west = wind FROM the east
    from backend.spread_model import INPUT_FEATURES

    mapped = baselines.old_model_inputs(x)
    th, vs = mapped[:, INPUT_FEATURES.index("th")], mapped[:, INPUT_FEATURES.index("vs")]
    assert np.allclose(th[0], 0.0) and np.allclose(th[1], 90.0) and np.allclose(vs, 5.0)
    assert np.isnan(mapped[:, INPUT_FEATURES.index("NDVI")]).all()


def test_all_baselines_shapes():
    x = np.zeros((3, C, 64, 64), dtype=np.float16)
    x[:, v2.CHANNELS.index("fire_t"), 30, 30] = 1
    x[:, v2.CHANNELS.index("wind_u")] = 3
    out = baselines.all_baselines(x, include_old_model=False)
    assert set(out) == {"persistence", "dilated 1", "dilated 2", "downwind 1", "downwind 2"}
    assert out["persistence"].sum() == 3 and out["dilated 2"].sum() == 75 and out["downwind 2"].sum() == 9


# ----------------------------------------------------------------------------- metrics


def test_pooled_scores_by_hand():
    fire_t = np.zeros((1, 2, 3))
    fire_t[0, 0, 0] = 1
    label = np.array([[[1, 1, 0], [1, 0, 0]]])  # (0,0) still burning; (0,1), (1,0) new
    predicted = np.array([[[1, 1, 1], [0, 0, 0]]])
    s = evaluate_v2.pooled_scores(predicted, label, fire_t)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 3)
    assert s["iou"] == pytest.approx(2 / 4) and s["f1"] == pytest.approx(2 / 3)
    assert s["new_fire_recall"] == pytest.approx(1 / 2)  # caught (0,1), missed (1,0)
    assert s["new_fire_precision"] == pytest.approx(1 / 2)  # forecast outside today: (0,1), (0,2)
    assert s["forecast_px"] == 3 and s["burned_px"] == 3


def test_persistence_has_zero_new_fire_recall():
    rng = np.random.default_rng(0)
    fire_t = rng.random((5, 64, 64)) > 0.9
    label = rng.random((5, 64, 64)) > 0.9
    assert evaluate_v2.pooled_scores(fire_t, label, fire_t)["new_fire_recall"] == 0


def test_average_precision_matches_the_training_notebook():
    rng = np.random.default_rng(1)
    p = rng.random(5000).astype(np.float32)
    y = (rng.random(5000) < p * 0.3).astype(np.float32)
    # The notebook's own definition (train_spread.ipynb, average_precision)
    pt, yt = torch.from_numpy(p), torch.from_numpy(y)
    hits = yt[pt.argsort(descending=True)]
    precision = hits.cumsum(0) / torch.arange(1, len(hits) + 1)
    notebook = ((precision * hits).sum() / hits.sum()).item()
    assert evaluate_v2.average_precision(p, y) == pytest.approx(notebook, rel=1e-5)


def test_choose_threshold_matches_brute_force():
    rng = np.random.default_rng(2)
    p = rng.random((4, 64, 64)).astype(np.float32)
    y = rng.random((4, 64, 64)) < p**3
    threshold, best = evaluate_v2.choose_threshold(p, y, bins=100)
    brute = max(
        ((t / 100, evaluate_v2.pooled_scores(p >= t / 100, y, np.zeros_like(p))["iou"]) for t in range(100)),
        key=lambda pair: pair[1],
    )
    assert best == pytest.approx(brute[1]) and threshold == pytest.approx(brute[0])
    # Perfectly separable: IoU 1
    separable = np.where(y, 0.9, 0.1).astype(np.float32)
    assert evaluate_v2.choose_threshold(separable, y)[1] == pytest.approx(1.0)


def test_bootstrap_difference():
    rng = np.random.default_rng(3)
    y = rng.random((50, 64, 64)) > 0.95
    same = evaluate_v2.bootstrap_iou_difference(y, y, y, reps=200)
    assert same["difference"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    worse = np.zeros_like(y)
    diff = evaluate_v2.bootstrap_iou_difference(y, worse, y, reps=200)
    assert diff["difference"] == pytest.approx(1.0) and diff["ci_low"] > 0.9


def test_per_tile_iou_nan_when_nothing_anywhere():
    out = evaluate_v2.per_tile_iou(np.zeros((2, 4, 4)), np.zeros((2, 4, 4)))
    assert np.isnan(out).all()



def test_weighted_scores_equal_duplicated_tiles():
    # Weight 2 on a tile must give the same pooled scores as that tile appearing twice
    rng = np.random.default_rng(4)
    pred = rng.random((3, 16, 16)) > 0.7
    label = rng.random((3, 16, 16)) > 0.7
    today = rng.random((3, 16, 16)) > 0.8
    weighted = evaluate_v2.weighted_scores(evaluate_v2.per_tile_counts(pred, label, today), [2.0, 1.0, 1.0])
    dup = evaluate_v2.pooled_scores(np.concatenate([pred[:1], pred]), np.concatenate([label[:1], label]), np.concatenate([today[:1], today]))
    for key in ("precision", "recall", "iou", "f1", "new_fire_recall", "new_fire_precision"):
        assert weighted[key] == pytest.approx(dup[key])
    unweighted = evaluate_v2.weighted_scores(evaluate_v2.per_tile_counts(pred, label, today))
    assert unweighted["iou"] == pytest.approx(evaluate_v2.pooled_scores(pred, label, today)["iou"])


def test_weighted_average_precision_equals_duplication():
    rng = np.random.default_rng(5)
    p = rng.random((40, 32, 32)).astype(np.float32)
    y = rng.random((40, 32, 32)) < p * 0.5
    weights = rng.choice([1.0, 2.0, 3.0], 40)
    w = evaluate_v2.weighted_average_precision(p, y, weights)
    repeat = np.repeat(np.arange(40), weights.astype(int))
    dup = evaluate_v2.average_precision(p[repeat], y[repeat])
    # Duplicated pixels tie; the unweighted sum scores the two copies at successive ranks, the
    # weighted one scores the pair once at the end, so they agree to about 1e-4, not exactly
    assert w == pytest.approx(dup, rel=1e-3)
    assert evaluate_v2.weighted_average_precision(p, y, None) == pytest.approx(evaluate_v2.average_precision(p, y))
