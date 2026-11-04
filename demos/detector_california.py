"""Run the trained fire detector on California patches it never trained on.

Reads the model and index downloaded from Kaggle into data/training, fetches
only the needed patches through the Kaggle API, writes figures to docs/figures,
and writes the map and tables to data/demos/california.
"""

import contextlib
import io
import sys
import time
import zipfile
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import tifffile
import torch

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "docs" / "figures"
OUTPUT = REPO / "data" / "demos" / "california"
sys.path.insert(0, str(REPO / "notebooks"))
from inspect_landsat import _geo_tags, _ground_xy, _utm_to_latlon, _zone, false_color  # noqa: E402

LOCATIONS = {
    "045032": "Northern California (Trinity Alps, Six Rivers NF)",
    "042035": "Southern Sierra Nevada (Sequoia NF)",
}
SPLIT_NOTE = {"test": "test split, never seen in training", "val": "val split, never trained on"}
FILE_BANDS = (1, 2, 3, 4, 5, 6, 7, 9, 10, 11)
INPUT_BANDS = (7, 6, 2)
THRESHOLD = 0.25
PIXEL_HECTARES = 0.09  # 30 m by 30 m
DATA_ROOT = REPO / "data" / "kaggle"


def load_model():
    """Build U-Net Light from the training notebook's own model cell."""
    notebook = nbformat.read(REPO / "notebooks" / "train_unet.ipynb", as_version=4)
    source = next(cell.source for cell in notebook.cells if cell.get("id") == "a1f0c019")
    namespace = {"torch": torch}
    exec(source.split("parameters =")[0], namespace)
    model = namespace["UNet"](len(INPUT_BANDS))
    weights = REPO / "data" / "training" / "detection_unet.pt"
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.eval()


def fetch(api, ref, name):
    root = DATA_ROOT / ref.split("/")[1]
    target = root / name
    if target.exists():
        return target
    zipped = target.with_name(target.name + ".zip")
    for attempt in range(6):
        try:
            if not zipped.exists():
                with contextlib.redirect_stdout(io.StringIO()):
                    api.dataset_download_file(ref, name, path=root, quiet=True)
            break
        except Exception as error:
            if "429" not in str(error) or attempt == 5:
                raise
            time.sleep(10 * 2**attempt)
    if zipped.exists():
        with zipfile.ZipFile(zipped) as archive:
            target.write_bytes(archive.read(archive.namelist()[0]))
        zipped.unlink()
    return target


def to_latlon(geo, row, col):
    zone, northern = _zone(geo["crs"])
    return _utm_to_latlon(*_ground_xy(geo, row, col), zone, northern)


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    index = pd.read_csv(REPO / "data" / "training" / "index.csv", dtype={"path_row": str, "acquired": str})
    rows = index[index.path_row.isin(LOCATIONS)].reset_index(drop=True)
    print(f"{len(rows)} patches from {sorted(LOCATIONS)}")

    model = load_model()
    channels = [FILE_BANDS.index(band) for band in INPUT_BANDS]
    records, views = [], []
    for row in rows.itertuples():
        image_path = fetch(api, row.dataset, row.image_file)
        image = tifffile.imread(image_path)
        truth = np.zeros(image.shape[:2], dtype=bool)
        if isinstance(row.voting_file, str) and row.voting_file:
            truth = tifffile.imread(fetch(api, row.dataset, row.voting_file)) == 1
        inputs = image[:, :, channels].astype(np.float32).transpose(2, 0, 1) / 65535
        with torch.no_grad():
            probability = torch.sigmoid(model(torch.from_numpy(inputs)[None]))[0, 0].numpy()
        predicted = probability > THRESHOLD

        geo = _geo_tags(image_path)
        center_lat, center_lon = to_latlon(geo, 128, 128)
        fire_rows, fire_cols = np.nonzero(predicted)
        fire_points = [to_latlon(geo, r, c) for r, c in zip(fire_rows, fire_cols)]
        hits = int((predicted & truth).sum())
        records.append(
            {
                "location": LOCATIONS[row.path_row],
                "path_row": row.path_row,
                "date": f"{row.acquired[:4]}-{row.acquired[4:6]}-{row.acquired[6:]}",
                "split": row.split,
                "patch_id": row.patch_id,
                "center_lat": round(center_lat, 4),
                "center_lon": round(center_lon, 4),
                "predicted_fire_pixels": int(predicted.sum()),
                "true_fire_pixels": int(truth.sum()),
                "correct_fire_pixels": hits,
                "false_alarm_pixels": int((predicted & ~truth).sum()),
                "missed_pixels": int((~predicted & truth).sum()),
                "predicted_hectares": round(predicted.sum() * PIXEL_HECTARES, 1),
                "true_hectares": round(truth.sum() * PIXEL_HECTARES, 1),
            }
        )
        views.append((row, image, predicted, truth, fire_points))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(records)
    report.to_csv(OUTPUT / "patch_report.csv", index=False)
    write_summary(report)
    for path_row in LOCATIONS:
        draw_location(path_row, [view for view in views if view[0].path_row == path_row], report)
    draw_map(views, report)
    print(f"Wrote figures to {FIGURES} and the map and tables to {OUTPUT}")


def scores(frame):
    hits = frame.correct_fire_pixels.sum()
    predicted = frame.predicted_fire_pixels.sum()
    truth = frame.true_fire_pixels.sum()
    union = predicted + truth - hits
    return {
        "patches": len(frame),
        "patches_with_fire": int((frame.true_fire_pixels > 0).sum()),
        "patches_flagged": int((frame.predicted_fire_pixels > 0).sum()),
        "true_fire_hectares": round(truth * PIXEL_HECTARES, 1),
        "predicted_fire_hectares": round(predicted * PIXEL_HECTARES, 1),
        "recall": round(hits / truth, 3) if truth else float("nan"),
        "precision": round(hits / predicted, 3) if predicted else float("nan"),
        "iou": round(hits / union, 3) if union else float("nan"),
    }


def write_summary(report):
    table = pd.DataFrame(
        {path_row: scores(report[report.path_row == path_row]) for path_row in LOCATIONS} | {"all": scores(report)}
    ).T
    table.insert(0, "date", [report[report.path_row == pr].date.iloc[0] for pr in LOCATIONS] + [""])
    table.insert(0, "location", [LOCATIONS[pr] for pr in LOCATIONS] + ["Both locations"])
    table.to_csv(OUTPUT / "summary.csv")
    print(table.to_string())


def overlay(image, mask, color):
    picture = false_color(image).copy()
    picture[mask] = color
    return picture


def draw_location(path_row, views, report):
    """The patches with the most fire: image, model output, and the label."""
    views = sorted(views, key=lambda view: view[3].sum() + view[2].sum(), reverse=True)[:4]
    fig, axes = plt.subplots(len(views), 3, figsize=(10.5, 3.6 * len(views)), squeeze=False)
    for line, (row, image, predicted, truth, _) in enumerate(views):
        found = int((predicted & truth).sum())
        axes[line, 0].imshow(false_color(image))
        axes[line, 0].set_title(f"Landsat-8, bands 7-6-2\n{row.patch_id}", fontsize=10)
        axes[line, 1].imshow(overlay(image, predicted, (1.0, 0.1, 0.1)))
        axes[line, 1].set_title(f"Model: {int(predicted.sum())} fire px", fontsize=10)
        axes[line, 2].imshow(overlay(image, truth, (1.0, 0.85, 0.0)))
        axes[line, 2].set_title(f"Label: {int(truth.sum())} fire px, {found} found", fontsize=10)
        for axis in axes[line]:
            axis.axis("off")
    location = report[report.path_row == path_row].iloc[0]
    fig.suptitle(f"{LOCATIONS[path_row]}, {location.date}", fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES / f"detector_{path_row}.png", dpi=150)
    plt.close(fig)


def draw_map(views, report):
    center = [report.center_lat.mean(), report.center_lon.mean()]
    fire_map = folium.Map(location=center, zoom_start=6, tiles="OpenStreetMap")
    for row, _, predicted, truth, fire_points in views:
        if not fire_points:
            continue
        latitudes, longitudes = zip(*fire_points)
        hits = int((predicted & truth).sum())
        folium.CircleMarker(
            location=[float(np.mean(latitudes)), float(np.mean(longitudes))],
            radius=4 + min(int(predicted.sum()) ** 0.5, 20),
            color="red",
            fill=True,
            fill_opacity=0.6,
            popup=(
                f"{LOCATIONS[row.path_row]}<br>{row.acquired}<br>{row.patch_id}<br>"
                f"Model: {int(predicted.sum())} fire px ({predicted.sum() * PIXEL_HECTARES:.1f} ha)<br>"
                f"Label: {int(truth.sum())} fire px, {hits} found"
            ),
        ).add_to(fire_map)
    fire_map.save(str(OUTPUT / "detector_map.html"))


if __name__ == "__main__":
    main()
