"""Boundary checks for the flood pipeline on tiny synthetic rasters (no dataset download needed).

    python -m pytest flood/tests -q
"""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from flood import contract
from flood.bundle import load as load_bundle
from flood.data import ContractError, read_manifest, read_radar, read_sample
from flood.geoid import assign_splits, split_groups
from flood.metrics import confusion, evaluate, flood_scores
from flood.model import UNet, predict_classes
from flood.predict import run

H = W = 40
UTM = "EPSG:32633"
T = from_origin(500000, 4600000, 10, 10)


def write(path, arr, crs=UTM, transform=T, descriptions=None):
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", count=arr.shape[0], height=arr.shape[1], width=arr.shape[2],
                       dtype=arr.dtype, crs=crs, transform=transform) as f:
        f.write(arr)
        if descriptions:
            f.descriptions = tuple(descriptions)
    return path


@pytest.fixture
def sample(tmp_path):
    rng = np.random.default_rng(0)
    radar = rng.uniform(0.001, 0.5, (2, H, W)).astype("float32")
    radar[:, :5, :5] = 0  # no data corner
    label = np.zeros((H, W), "uint8"); label[10:20, 10:20] = 2; label[30:, :] = 1; label[0, :] = 255
    perm = (label == 1).astype("uint8")
    valid = np.ones((H, W), "uint8")
    paths = {
        "radar_path": write(tmp_path / "t_s1grd_post_20240101T120000.tif", radar, descriptions=["VV", "VH"]).name,
        "label_path": write(tmp_path / "t_label.tif", label).name,
        "validity_path": write(tmp_path / "t_validity.tif", valid).name,
        "permwater_path": write(tmp_path / "t_permwater.tif", perm).name,
    }
    return tmp_path, {"sample_id": "EV1-1-1", "event_id": "EV1", **paths}


def test_read_sample_marks_invalid_and_keeps_grid(sample):
    root, row = sample
    s = read_sample(row, root)
    assert s["x"].shape == (2, H, W)
    assert (s["y"][:5, :5] == contract.IGNORE).all()  # radar no-data becomes ignored, not dropped
    assert (s["y"][0] == contract.IGNORE).all()
    assert s["valid"].sum() == H * W - W - 4 * 5  # row 0 plus the 4 remaining corner rows
    assert s["profile"]["crs"].to_string() == UTM


def test_misaligned_label_is_rejected(sample):
    root, row = sample
    write(root / "t_label.tif", np.zeros((H, W), "uint8"), transform=from_origin(500010, 4600000, 10, 10))
    with pytest.raises(ContractError, match="transform"):
        read_sample(row, root)


def test_other_crs_is_rejected(sample):
    root, row = sample
    write(root / "t_permwater.tif", np.zeros((H, W), "uint8"), crs="EPSG:32634")
    with pytest.raises(ContractError, match="CRS"):
        read_sample(row, root)


def test_unknown_label_value_is_rejected(sample):
    root, row = sample
    write(root / "t_label.tif", np.full((H, W), 7, "uint8"))
    with pytest.raises(ContractError, match="label values"):
        read_sample(row, root)


def test_radar_already_in_db_is_rejected(tmp_path):
    p = write(tmp_path / "db.tif", np.full((2, H, W), -15.0, "float32"), descriptions=["VV", "VH"])
    with pytest.raises(ContractError, match="linear sigma0"):
        read_radar(p)


def test_wrong_band_order_is_rejected(tmp_path):
    p = write(tmp_path / "swap.tif", np.full((2, H, W), 0.1, "float32"), descriptions=["VH", "VV"])
    with pytest.raises(ContractError, match="band order"):
        read_radar(p)


def test_geographic_grid_is_rejected(tmp_path):
    p = write(tmp_path / "geo.tif", np.full((2, H, W), 0.1, "float32"), crs="EPSG:4326",
              transform=from_origin(10, 45, 1e-4, 1e-4), descriptions=["VV", "VH"])
    with pytest.raises(ContractError, match="projected"):
        read_radar(p)


def test_flood_scores_known_values():
    y = np.array([[2, 2, 2, 2, 0, 0, 1, 255]], "uint8")
    p = np.array([[2, 2, 0, 1, 2, 0, 2, 2]], "uint8")
    s = flood_scores(confusion(p, y))
    # flood: tp 2, fp 2 (one background, one permanent water), fn 2; the ignored pixel is not scored
    assert s["flood_iou"] == pytest.approx(2 / 6)
    assert s["flood_dice"] == pytest.approx(4 / 8)
    assert s["flood_precision"] == pytest.approx(0.5) and s["flood_recall"] == pytest.approx(0.5)
    assert s["permwater_called_flood"] == pytest.approx(1.0)
    assert s["flood_called_permwater"] == pytest.approx(0.25)
    assert s["pixels_scored"] == 7


def test_dry_scene_alarm_and_area():
    prof = {"crs": rasterio.crs.CRS.from_string(UTM), "transform": T}
    dry_y = np.zeros((100, 100), "uint8")
    wet_y = dry_y.copy(); wet_y[:10, :10] = 2
    alarm = np.zeros_like(dry_y); alarm[:30, :30] = 2  # 900 px * 100 m2 = 0.09 km2 >= 0.05
    speck = np.zeros_like(dry_y); speck[:5, :5] = 2  # 0.0025 km2, below the alarm size
    mk = lambda y, i: {"y": y, "valid": y != 255, "profile": prof, "sample_id": f"s{i}", "event_id": f"e{i}"}
    rep = evaluate([alarm, speck, wet_y], [mk(dry_y, 0), mk(dry_y, 1), mk(wet_y, 2)])
    assert rep["dry_scenes"]["count"] == 2 and rep["dry_scenes"]["with_false_alarm"] == 1
    assert rep["area_km2"]["true"] == pytest.approx(0.01)
    assert rep["events_with_flood"] == 1 and rep["mean_event_flood_iou"] == 1.0


def test_manifest_rejects_event_in_two_splits(tmp_path):
    cols = {"radar_path": "r", "label_path": "l", "validity_path": "v", "permwater_path": "p"}
    df = pd.DataFrame([{"sample_id": "a", "event_id": "E1", "split": "train", "split_group": "g1", **cols},
                       {"sample_id": "b", "event_id": "E1", "split": "test", "split_group": "g2", **cols}])
    df.to_csv(tmp_path / "m.csv", index=False)
    with pytest.raises(ContractError, match="events in more than one split"):
        read_manifest(tmp_path / "m.csv")


def test_split_groups_keep_events_and_clusters_together():
    rows = []
    # E1 and E2 share cluster 5 (same ground, different floods); E1 also has a tile in cluster 6.
    for i, (ev, cl) in enumerate([("E1", 5), ("E1", 6), ("E2", 5), ("E3", 7), ("E4", 8)] * 3):
        rows.append({"event_id": ev, "cluster_id": cl, "tree": "geoid-flood", "sample_id": f"{ev}-{i}"})
    rows.append({"event_id": "E9", "cluster_id": -1, "tree": "geoid-flood-heldout", "sample_id": "E9-0"})
    c = pd.DataFrame(rows)
    c["split_group"] = split_groups(c)
    c["my_split"] = assign_splits(c)
    assert c[c.event_id.isin(["E1", "E2"])].split_group.nunique() == 1
    assert (c.groupby("event_id").my_split.nunique() == 1).all()
    assert c[c.event_id == "E9"].my_split.iloc[0] == "heldout"


def test_predict_classes_any_size():
    model = UNet(base=4)
    out = predict_classes(model, np.zeros((2, 37, 53), "float16"), "cpu", window=32)
    assert out.shape == (37, 53) and out.dtype == np.uint8


def make_bundle(path, **override):
    path.mkdir()
    model = UNet(base=4)
    torch.save(model.state_dict(), path / "weights.pt")
    meta = {"hazard": "flood", "task": "detection", "model_version": "test-v0", "architecture": model.config,
            "input_schema": contract.INPUT_SCHEMA, "preprocessing_version": contract.PREPROCESSING_VERSION,
            "normalization": {"mean": [-10.0, -17.0], "std": [5.0, 6.0]}, "weights_file": "weights.pt",
            "weights_sha256": hashlib.sha256((path / "weights.pt").read_bytes()).hexdigest(), "chosen_variant": "model"}
    meta.update(override)
    (path / "bundle.json").write_text(json.dumps(meta))
    return path


def test_bundle_with_other_preprocessing_is_rejected(tmp_path):
    b = make_bundle(tmp_path / "b", preprocessing_version="something-else")
    with pytest.raises(ContractError, match="preprocessing"):
        load_bundle(b)


def test_bundle_with_tampered_weights_is_rejected(tmp_path):
    b = make_bundle(tmp_path / "b", weights_sha256="0" * 64)
    with pytest.raises(ContractError, match="checksum"):
        load_bundle(b)


def test_predict_writes_georeferenced_result(sample, monkeypatch):
    root, row = sample
    monkeypatch.setattr("flood.bundle.device", lambda: "cpu")
    b = make_bundle(root / "bundle")
    rec = run(str(root / row["radar_path"]), root / "results", b, validity=str(root / row["validity_path"]))
    out = root / "results" / rec["prediction_id"]
    required = {"prediction_id", "hazard", "task", "model_version", "preprocessing_version", "input_ids",
                "acquisition_time_utc", "generated_at_utc", "geometry", "valid_coverage", "output_units", "availability_status"}
    assert required <= set(rec)
    assert rec["hazard"] == "flood" and rec["task"] == "detection" and rec["availability_status"] == "available"
    assert rec["acquisition_time_utc"] == "2024-01-01T12:00:00+00:00"
    assert rec["scores_are_probabilities"] is False
    with rasterio.open(out / "classes.tif") as f:
        assert f.crs.to_string() == UTM and f.transform == T
        cls = f.read(1)
    assert (cls[:5, :5] == contract.IGNORE).all()
    assert json.loads((out / "flood.geojson").read_text())["type"] == "FeatureCollection"
    # same inputs and model give the same id, so a re-run overwrites rather than duplicates
    assert run(str(root / row["radar_path"]), root / "results", b, validity=str(root / row["validity_path"]))["prediction_id"] == rec["prediction_id"]


def test_predict_reports_unavailable_when_no_valid_pixels(tmp_path, monkeypatch):
    monkeypatch.setattr("flood.bundle.device", lambda: "cpu")
    p = write(tmp_path / "x_s1grd_post_20240101T000000.tif", np.zeros((2, H, W), "float32"), descriptions=["VV", "VH"])
    rec = run(str(p), tmp_path / "results", make_bundle(tmp_path / "b"))
    assert rec["availability_status"] == "unavailable" and rec["geometry"] is None
