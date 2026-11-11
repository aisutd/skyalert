"""Show the trained spread model on test fires: today, the forecast, and tomorrow.

Uses the test file in data/ndws, the weights and input stats downloaded from
Kaggle into data/training, the TFRecord reader from train_spread.ipynb, and the
model and input scaling from backend/spread_model.py. Writes the figure to
docs/figures and the per-tile table to data/demos/spread.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from backend.spread_model import INPUT_FEATURES, THRESHOLD, predict  # noqa: E402

FIGURES = REPO / "docs" / "figures"
OUTPUT = REPO / "data" / "demos" / "spread"
MIN_TOMORROW_KM2 = 30


def notebook_reader():
    """The TFRecord reader from train_spread.ipynb, without running the rest of the notebook."""
    notebook = nbformat.read(REPO / "notebooks" / "train_spread.ipynb", as_version=4)
    cells = {cell.get("id"): cell.source for cell in notebook.cells}
    namespace = {}
    exec(cells["b2e0d003"].split("ON_KAGGLE = ")[0], namespace)
    namespace["DATA_ROOT"] = REPO / "data" / "ndws"
    exec(cells["b2e0d007"].split("data, rows = {}, []")[0], namespace)
    return namespace


def main():
    inputs, labels, _ = notebook_reader()["load_split"]("test")
    probability = predict(inputs)

    today = inputs[:, INPUT_FEATURES.index("PrevFireMask")]
    valid = labels != -1
    actual = (labels == 1) & valid
    predicted = (probability > THRESHOLD) & valid
    copied = (today == 1) & valid

    def iou(guess):
        hits = (guess & actual).sum((1, 2))
        union = (guess | actual).sum((1, 2))
        return np.where(union > 0, hits / np.maximum(union, 1), np.nan)

    rows = pd.DataFrame(
        {
            "tile": np.arange(len(labels)),
            "today_km2": (today == 1).sum((1, 2)),
            "tomorrow_km2": actual.sum((1, 2)),
            "predicted_km2": predicted.sum((1, 2)),
            "caught_km2": (predicted & actual).sum((1, 2)),
            "model_iou": iou(predicted).round(3),
            "copy_today_iou": iou(copied).round(3),
        }
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUTPUT / "spread_tiles.csv", index=False)

    sizable = rows[rows.tomorrow_km2 >= MIN_TOMORROW_KM2].sort_values("model_iou", ascending=False)
    good = sizable.head(3)
    typical = sizable.iloc[[len(sizable) // 2]]
    chosen = pd.concat([good.assign(kind="Good forecast"), typical.assign(kind="Typical forecast")])
    draw(chosen, today, probability, labels)
    print(chosen.to_string(index=False))
    print(f"{len(sizable)} test fires with at least {MIN_TOMORROW_KM2} km2 burning tomorrow")


def draw(chosen, today, probability, labels):
    fire_colors = ListedColormap(["#9e9e9e", "black", "red"])  # uncertain, no fire, fire
    fig, axes = plt.subplots(len(chosen), 3, figsize=(10.5, 3.7 * len(chosen)), squeeze=False)
    for line, row in enumerate(chosen.itertuples()):
        tile = row.tile
        axes[line, 0].imshow(today[tile], cmap=fire_colors, vmin=-1, vmax=1)
        axes[line, 0].set_title(f"{row.kind}\nDay 1: fire today, {row.today_km2} km²", fontsize=10)
        axes[line, 1].imshow(probability[tile], cmap="magma", vmin=0, vmax=1)
        axes[line, 1].contour(today[tile] == 1, levels=[0.5], colors="cyan", linewidths=0.8)
        axes[line, 1].set_title(f"Model forecast for day 2\n{row.predicted_km2} km² above {THRESHOLD:.2f}", fontsize=10)
        axes[line, 2].imshow(labels[tile], cmap=fire_colors, vmin=-1, vmax=1)
        axes[line, 2].set_title(
            f"Day 2: what actually burned, {row.tomorrow_km2} km²\nmodel caught {row.caught_km2} km²", fontsize=10
        )
        for axis in axes[line]:
            axis.axis("off")
    fig.suptitle("Next-day fire spread, 64 × 64 km areas, 1 km per pixel", fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES / "spread_examples.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
