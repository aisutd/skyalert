"""Train the flood U-Net on GEOID-Flood train-split tiles and score it on EMSR712-3 (test split).

Same model and inputs as Flood.ipynb (after-flood Sentinel-1 VV + VH, U-Net with a ResNet-34 encoder,
3 classes), but trained on random crops from many tiles instead of 5 fixed chips: the 18 sample tiles
of EMSR712-10, plus whatever fetch_more.py downloaded into data/full/.

Run from the flood-model folder, after Flood.ipynb has downloaded the sample into data/:
    python fetch_more.py               # optional: about 300 more training tiles from many floods
    python train_flood.py              # train from scratch, then test
    python train_flood.py --sample-only  # train on the 18 sample tiles only
    python train_flood.py --test-only --heldout  # score on the 2025 floods from `fetch_more.py --heldout`

The test uses the map of normal rivers and lakes (permwater) to call water outside it flood. Download it once with:
    python -c "from huggingface_hub import snapshot_download as s; s('links-ads/geoid-flood', repo_type='dataset', allow_patterns=['sample/geoid-flood/*/permwater/*'], local_dir='data')"
    python train_flood.py --resume     # load data/flood_unet.pt and train more, then test
    python train_flood.py --test-only  # just score the saved model
"""
import argparse
import glob
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap

ROOT, FULL, HELDOUT = "data/sample/geoid-flood", "data/full", "data/heldout"  # FULL / HELDOUT come from fetch_more.py
TRAIN_EVENT, TEST_EVENT = "EMSR712-10", "EMSR712-3"  # the dataset's own train / test split
CKPT, RESULTS, FIGURE = "data/flood_unet.pt", "data/flood_results.json", "data/flood_test_examples.png"
IGNORE, CROP = 255, 256
NAMES = ["background", "permanent water", "flooded water"]


def tile_files(dirs):
    """(tile id, radar, label, validity) paths for every tile in these area folders."""
    out = []
    for d in dirs:
        for lf in sorted(glob.glob(f"{d}/label/*.tif")):
            tid = os.path.basename(lf).split("_label")[0]
            post = glob.glob(f"{d}/s1grd/{tid}_s1grd_post_*.tif")
            valid = glob.glob(f"{d}/validity/{tid}_validity*.tif")
            if post and valid:  # skip a tile whose radar didn't download
                out.append((tid, post[0], lf, valid[0]))
    return out


def read_tile(post, lf, valid):
    """After-flood radar in dB [bands, H, W] (NaN where missing) and the label with bad pixels set to 255."""
    with rasterio.open(post) as f:
        x = f.read().astype("float32")  # [2, H, W]: band 1 VV, band 2 VH
    with rasterio.open(lf) as f:
        y = f.read(1).astype("uint8")
    with rasterio.open(valid) as f:
        ok = f.read(1) == 1
    bad = (x <= 0).any(0) | ~np.isfinite(x).all(0) | ~ok  # radar has no data here, or the dataset marks it invalid
    x[(x <= 0) | ~np.isfinite(x)] = np.nan
    x = 10 * np.log10(x)  # radar values to decibels
    y[bad] = IGNORE  # no usable input means no answer to learn from
    return x, y


def band_stats(files):
    """Per-band mean and std over all training tiles, one tile at a time so it fits in memory."""
    n = s = ss = 0
    for _, post, lf, valid in files:
        x, _ = read_tile(post, lf, valid)
        v = np.isfinite(x)
        n += v.sum((1, 2)); s += np.where(v, x, 0).sum((1, 2), dtype="float64"); ss += np.where(v, x * x, 0).sum((1, 2), dtype="float64")
    mean = s / n
    return mean.astype("float32"), np.sqrt(ss / n - mean ** 2).astype("float32")


def load_tiles(files, mean, std):
    """Scaled radar stored as float16 (half the memory) and labels, for every tile."""
    tiles = []
    for tid, post, lf, valid in files:
        x, y = read_tile(post, lf, valid)
        x = np.nan_to_num((x - mean[:, None, None]) / std[:, None, None], nan=0.0).astype("float16")
        tiles.append((tid, x, y))
    return tiles


def sample_batch(tiles, flood_px, rng, n):
    """n random 256 x 256 crops, half of them centred near a flood pixel so the rare class shows up often."""
    xs, ys = [], []
    for _ in range(n):
        k = rng.integers(len(tiles))
        _, x, y = tiles[k]
        H, W = y.shape
        if rng.random() < 0.5 and len(flood_px[k]):
            r, c = flood_px[k][rng.integers(len(flood_px[k]))]
            r = int(np.clip(r - CROP // 2 + rng.integers(-64, 65), 0, H - CROP))
            c = int(np.clip(c - CROP // 2 + rng.integers(-64, 65), 0, W - CROP))
        else:
            r, c = rng.integers(0, H - CROP + 1), rng.integers(0, W - CROP + 1)
        cx, cy = x[:, r:r + CROP, c:c + CROP], y[r:r + CROP, c:c + CROP]
        t = rng.integers(4)  # random rotation and flip: the same flood seen from another angle
        cx, cy = np.rot90(cx, t, axes=(1, 2)), np.rot90(cy, t)
        if rng.random() < 0.5:
            cx, cy = cx[:, :, ::-1], cy[:, ::-1]
        xs.append(cx.copy()); ys.append(cy.copy())
    return torch.from_numpy(np.stack(xs).astype("float32")), torch.from_numpy(np.stack(ys).astype("int64"))


def new_model(device):
    return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=2, classes=3).to(device)


@torch.no_grad()
def predict(model, tiles, device):
    model.eval()
    return [model(torch.from_numpy(x.astype("float32"))[None].to(device)).argmax(1)[0].cpu().numpy() for _, x, _ in tiles]


def confusion(preds, tiles):
    cm = np.zeros((3, 3), dtype=np.int64)  # rows = truth, columns = prediction
    for p, (_, _, y) in zip(preds, tiles):
        v = y != IGNORE
        cm += np.bincount(y[v].astype(np.int64) * 3 + p[v], minlength=9).reshape(3, 3)
    return cm


def scores(cm):
    tp = cm[2, 2]; fp = cm[:, 2].sum() - tp; fn = cm[2, :].sum() - tp
    total = cm.sum()
    flood_vs_rest_correct = tp + (total - tp - fp - fn)
    water_t = cm[1:, :].sum(1).sum(); water_p = cm[:, 1:].sum(0).sum(); water_tp = cm[1:, 1:].sum()
    return {
        "flood_precision": tp / max(tp + fp, 1),  # of the pixels called flooded, how many really were
        "flood_recall": tp / max(tp + fn, 1),  # of the really flooded pixels, how many were found
        "flood_iou": tp / max(tp + fp + fn, 1),
        "flood_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "accuracy_3_class": np.trace(cm) / total,  # background / permanent water / flood all right
        "accuracy_flood_vs_not": flood_vs_rest_correct / total,
        "error_rate_flood_vs_not": 1 - flood_vs_rest_correct / total,
        "any_water_iou": water_tp / max(water_t + water_p - water_tp, 1),  # flood + permanent water together
        "always_no_flood_accuracy": 1 - cm[2, :].sum() / total,  # what you'd get by never predicting flood
        "flood_share_of_pixels": cm[2, :].sum() / total,
        "pixels_scored": int(total),
    }


def threshold_baseline(train, test, mean, std):
    """Classic radar trick: water is dark in VH. Pick the dB cut-off on train, score 'any water' on test."""
    def water_iou(tiles, t_db):
        t = (t_db - mean[1]) / std[1]  # the same cut-off on the scaled values
        i = u = 0
        for _, x, y in tiles:
            v = (y != IGNORE) & (x[1] != 0)  # x == 0 exactly is where radar was missing
            p, w = (x[1] < t)[v], (y[v] == 1) | (y[v] == 2)
            i += (p & w).sum(); u += (p | w).sum()
        return i / max(u, 1)
    best = max(np.arange(-30, -14, 0.5), key=lambda t: water_iou(train, t))
    return float(best), float(water_iou(test, best))


def permwater_maps(tiles, folders):
    """Map of normal rivers and lakes for each tile (1 = always water), or None if the files aren't downloaded."""
    maps = []
    for tid, _, _ in tiles:
        f = [p for d in folders for p in glob.glob(f"{d}/permwater/{tid}_permwater*.tif")]
        if not f:
            return None
        with rasterio.open(f[0]) as src:
            maps.append(src.read(1) == 1)
    return maps


def split_water(preds, maps):
    """Keep where the model sees water, but let the map decide which: inside the map = permanent, outside = flood."""
    return [np.where(p > 0, np.where(m, 1, 2), 0) for p, m in zip(preds, maps)]


def flood_baseline(test, mean, std, cut, maps):
    """Simple method with the same map: dark in VH and not a normal river or lake = flood."""
    t = (cut - mean[1]) / std[1]
    preds = [np.where((x[1] < t) & (x[1] != 0), np.where(m, 1, 2), 0) for (_, x, _), m in zip(test, maps)]
    return scores(confusion(preds, test))


def save_figure(preds, tiles, path, n=4):
    order = sorted(range(len(tiles)), key=lambda i: -(tiles[i][2] == 2).sum())[:n]  # the most-flooded test tiles
    cmap = ListedColormap(["#dddddd", "#1f4fd8", "#ff7f0e", "#000000"])
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for row, i in enumerate(order):
        tid, x, y = tiles[i]
        axes[row, 0].imshow(x[1], cmap="gray", vmin=-3, vmax=3)
        axes[row, 1].imshow(np.where(y == IGNORE, 3, y), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        axes[row, 2].imshow(np.where(y == IGNORE, 3, preds[i]), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        axes[row, 0].set_ylabel(tid)
        for c, t in enumerate(["after radar (VH)", "truth", "prediction"]):
            axes[row, c].set_title(t, fontsize=9); axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=50, help="batches per epoch")
    ap.add_argument("--heldout", action="store_true", help="test on data/heldout (2025 floods) instead of EMSR712-3")
    ap.add_argument("--sample-only", action="store_true", help="train on the 18 sample tiles only, ignoring data/full")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--resume", action="store_true", help="continue training the saved model")
    ap.add_argument("--test-only", action="store_true", help="score the saved model without training")
    a = ap.parse_args()

    torch.manual_seed(0); rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device, flush=True)

    train_dirs = [f"{ROOT}/{TRAIN_EVENT}"] + ([] if a.sample_only else sorted(glob.glob(f"{FULL}/*")))  # + tiles from fetch_more.py
    test_dirs = sorted(glob.glob(f"{HELDOUT}/*")) if a.heldout else [f"{ROOT}/{TEST_EVENT}"]
    test_name = f"held-out 2025 floods ({HELDOUT})" if a.heldout else TEST_EVENT
    results, figure = (RESULTS.replace(".json", "_heldout.json"), FIGURE.replace(".png", "_heldout.png")) if a.heldout else (RESULTS, FIGURE)
    train_files, test_files = tile_files(train_dirs), tile_files(test_dirs)
    n_floods = len({tid.split("-")[0] for tid, *_ in train_files})
    test_floods = len({tid.split("-")[0] for tid, *_ in test_files})
    print(f"train: {len(train_files)} tiles from {n_floods} floods   test {test_name}: {len(test_files)} tiles from {test_floods} floods", flush=True)

    ckpt = torch.load(CKPT, map_location=device, weights_only=False) if (a.resume or a.test_only) else None
    if ckpt:
        mean, std = np.array(ckpt["mean"], "float32"), np.array(ckpt["std"], "float32")  # keep the scaling the model learned with
    else:
        mean, std = band_stats(train_files)  # training tiles only
    print("scaling mean", mean, "std", std, flush=True)
    train, test = load_tiles(train_files, mean, std), load_tiles(test_files, mean, std)

    model = new_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    history, done = [], 0
    if ckpt:
        model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["opt"])
        history, done = ckpt["history"], ckpt["epochs_done"]
        print(f"loaded {CKPT}: {done} epochs already trained", flush=True)

    if not a.test_only:
        ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
        dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE)  # rewards overlap, so the small flood class counts
        flood_px = [np.argwhere(y == 2) for _, _, y in train]
        t0 = time.time()
        for ep in range(done, done + a.epochs):
            model.train(); total = 0.0
            for _ in range(a.steps):
                x, y = sample_batch(train, flood_px, rng, a.batch)
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = ce(out, y) + dice(out, y)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            history.append(total / a.steps)
            print(f"epoch {ep + 1:3d}  loss {history[-1]:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "mean": mean.tolist(), "std": std.tolist(),
                        "history": history, "epochs_done": ep + 1}, CKPT)  # saved every epoch, so stopping early loses nothing
        done += a.epochs

    raw = predict(model, test, device)
    cut, base_iou = threshold_baseline(train, test, mean, std)
    maps = permwater_maps(test, test_dirs)
    preds = split_water(raw, maps) if maps else raw  # the final answer uses the map when it's there
    if not maps:
        print("no permwater maps found: scoring the model alone (see the top of this file to download them)")
    rnd = lambda d: {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}
    s = rnd(scores(confusion(preds, test)))
    out = {"final": s, "model_alone": rnd(scores(confusion(raw, test))), "uses_permwater_map": bool(maps),
           "baseline_vh_threshold_db": cut, "baseline_any_water_iou": round(base_iou, 4),
           "epochs_trained": done, "train_tiles": len(train), "train_floods": n_floods}
    if maps:
        out["baseline_with_map"] = rnd(flood_baseline(test, mean, std, cut, maps))
    out["test_set"], out["test_tiles"], out["test_floods"] = test_name, len(test), test_floods
    json.dump(out, open(results, "w"), indent=2)
    save_figure(preds, test, figure)

    m = out["model_alone"]
    print(f"\nTest on {test_name}: {len(test)} tiles from {test_floods} floods the model never trained on, after {done} epochs")
    print(f"  model alone:          flood precision {m['flood_precision']:.1%}  recall {m['flood_recall']:.1%}  IoU {m['flood_iou']:.3f}")
    if maps:
        b = out["baseline_with_map"]
        print(f"  model + water map:    flood precision {s['flood_precision']:.1%}  recall {s['flood_recall']:.1%}  IoU {s['flood_iou']:.3f}")
        print(f"  dark-pixel + map:     flood precision {b['flood_precision']:.1%}  recall {b['flood_recall']:.1%}  IoU {b['flood_iou']:.3f}")
    print(f"  accuracy (flood vs not)  {s['accuracy_flood_vs_not']:.1%}   error rate {s['error_rate_flood_vs_not']:.1%}")
    print(f"  accuracy (all 3 classes) {s['accuracy_3_class']:.1%}")
    print(f"  'never say flood' accuracy {s['always_no_flood_accuracy']:.1%}   (flood is {s['flood_share_of_pixels']:.1%} of pixels)")
    print(f"  any-water IoU: model {s['any_water_iou']:.3f}   VH < {cut} dB threshold {base_iou:.3f}")
    print(f"saved {results}, {figure}" + ("" if a.test_only else f", {CKPT}"))


if __name__ == "__main__":
    main()
