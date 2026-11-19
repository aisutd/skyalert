"""Train the flood U-Net on GEOID-Flood train-split tiles and score it on tiles it never trained on.

Same model as Flood.ipynb (U-Net with a ResNet-34 encoder, 3 classes: background, permanent water, flood),
trained on random crops from the 18 sample tiles of EMSR712-10 plus whatever fetch_more.py put in data/full/.

Inputs: after-flood Sentinel-1 VV + VH, and with --pre also the before-flood VV + VH (4 bands), so the
model can see which water is new. After every 2 epochs it is scored on the validation tiles (data/val) and
only the best round is kept. The final answer uses the map of normal rivers and lakes (permwater): water
inside it is permanent, water outside it is flood.

Run from the flood-model folder:
    python fetch_more.py                         # about 300 training tiles      -> data/full
    python fetch_more.py --split val --tiles 80  # validation tiles              -> data/val
    python fetch_more.py --heldout --tiles 50    # 2025 floods, final test only  -> data/heldout
    python fetch_more.py --pre --split train     # (and --split val / heldout) before-flood radar, for --pre

    python train_flood.py --pre                  # train from scratch, keep the best round on val, score val
    python train_flood.py --pre --resume         # train the saved model more
    python train_flood.py --pre --test-only --eval heldout --tta   # final score on the 2025 floods
    python train_flood.py --test-only            # score the after-only model (data/flood_unet.pt)

--eval picks the tiles to score: val (default), heldout, or emsr712 (EMSR712-3, the sample's test area).
--tta averages the model over 8 flips and rotations of each tile (slower, usually a little better).
Only use --eval heldout for the final number; compare changes on val, so the held-out score stays honest.
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

ROOT = "data/sample/geoid-flood"
FULL, VAL, HELDOUT = "data/full", "data/val", "data/heldout"  # filled by fetch_more.py
TRAIN_EVENT, TEST_EVENT = "EMSR712-10", "EMSR712-3"  # the sample's own train / test split
IGNORE, CROP, Q = 255, 256, 20.0  # Q: scaled radar is stored as int8 in steps of 1/20 of a std, to fit in 8 GB RAM
EVAL_SETS = {"val": lambda: sorted(glob.glob(f"{VAL}/*")), "heldout": lambda: sorted(glob.glob(f"{HELDOUT}/*")),
             "emsr712": lambda: [f"{ROOT}/{TEST_EVENT}"]}


def tile_files(dirs, pre, dem=False):
    """(tile id, [radar files (+ ground-height file)], label, validity, permwater or None) for every tile in these area folders."""
    out = []
    for d in dirs:
        for lf in sorted(glob.glob(f"{d}/label/*.tif")):
            tid = os.path.basename(lf).split("_label")[0]
            post = glob.glob(f"{d}/s1grd/{tid}_s1grd_post_*.tif")
            before = glob.glob(f"{d}/s1grd/{tid}_s1grd_pre_*.tif")
            valid = glob.glob(f"{d}/validity/{tid}_validity*.tif")
            pw = glob.glob(f"{d}/permwater/{tid}_permwater*.tif")
            height = glob.glob(f"{d}/dem/{tid}_dem*.tif")
            if post and valid and (before or not pre) and (height or not dem):  # skip a tile with a missing download
                out.append((tid, post[:1] + (before[:1] if pre else []) + (height[:1] if dem else []), lf, valid[0], pw[0] if pw else None))
    return out


def read_tile(radar, lf, valid):
    """Radar in dB [bands, H, W] (after VV, VH, then before VV, VH; NaN where missing) and the label, bad pixels = 255."""
    xs, height = [], None
    for p in radar:
        with rasterio.open(p) as f:
            if "/dem/" in p:
                height = f.read(1).astype("float32")
            else:
                xs.append(f.read().astype("float32"))
    x = np.concatenate(xs)
    with rasterio.open(lf) as f:
        y = f.read(1).astype("uint8")
    with rasterio.open(valid) as f:
        ok = f.read(1) == 1
    bad = (x <= 0).any(0) | ~np.isfinite(x).all(0) | ~ok  # radar has no data here, or the dataset marks it invalid
    x[(x <= 0) | ~np.isfinite(x)] = np.nan
    x = 10 * np.log10(x)  # radar values to decibels
    y[bad] = IGNORE  # no usable input means no answer to learn from
    if height is not None:
        height[~np.isfinite(height) | (height < -500)] = np.nan
        h = torch.from_numpy(np.nan_to_num(height, nan=1e4))[None, None]
        low = nn.functional.max_pool2d(-h, (151, 1), stride=1, padding=(75, 0))  # lowest ground within about 750 m: rows, then columns (fast)
        low = -nn.functional.max_pool2d(low, (1, 151), stride=1, padding=(0, 75))[0, 0].numpy()
        rel = np.clip(height - low, 0, 30)  # metres above that low point: floods sit near 0, dark fields on higher ground do not
        x = np.concatenate([x, rel[None]])
    return x, y


def band_stats(files):
    """Per-band mean and std over all training tiles, one tile at a time so it fits in memory."""
    n = s = ss = 0
    for _, radar, lf, valid, _ in files:
        x, _ = read_tile(radar, lf, valid)
        v = np.isfinite(x)
        n += v.sum((1, 2)); s += np.where(v, x, 0).sum((1, 2), dtype="float64"); ss += np.where(v, x * x, 0).sum((1, 2), dtype="float64")
    mean = s / n
    return mean.astype("float32"), np.sqrt(ss / n - mean ** 2).astype("float32")


def scaled(x, mean, std):
    return np.nan_to_num((x - mean[:, None, None]) / std[:, None, None], nan=0.0)  # missing radar -> 0, the average


def load_train(files, mean, std):
    """Training tiles kept in memory as int8 (a quarter of float32), plus where their flood pixels are."""
    tiles = []
    for tid, radar, lf, valid, _ in files:
        x, y = read_tile(radar, lf, valid)
        q = np.clip(np.round(scaled(x, mean, std) * Q), -127, 127).astype("int8")
        tiles.append((tid, q, y, np.argwhere(y == 2).astype("int16")))
    return tiles


def sample_batch(tiles, rng, n):
    """n random 256 x 256 crops, half of them centred near a flood pixel so the rare class shows up often."""
    xs, ys = [], []
    for _ in range(n):
        _, x, y, fp = tiles[rng.integers(len(tiles))]
        H, W = y.shape
        if rng.random() < 0.5 and len(fp):
            r, c = fp[rng.integers(len(fp))]
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
    return torch.from_numpy(np.stack(xs).astype("float32") / Q), torch.from_numpy(np.stack(ys).astype("int64"))


def new_model(device, bands):
    return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=bands, classes=3).to(device)


@torch.no_grad()
def predict_one(model, x, device, tta, conf=0.0):
    """Class per pixel for one scaled tile [bands, H, W]; with tta, the average over 8 flips and rotations.
    With conf, a pixel is only called water when the model is at least that sure (0-1): fewer false alarms, more misses."""
    t = torch.from_numpy(x.astype("float32"))[None].to(device)
    if not tta:
        p = model(t).softmax(1)
    else:
        p = 0
        for k in range(4):
            for flip in (False, True):
                v = torch.rot90(t, k, (2, 3))
                v = torch.flip(v, (3,)) if flip else v
                o = model(v).softmax(1)
                o = torch.flip(o, (3,)) if flip else o
                p = p + torch.rot90(o, -k, (2, 3)) / 8  # turn the answer back the same way
    p = p[0].cpu().numpy()
    if not conf:
        return p.argmax(0)
    return np.where(1 - p[0] > conf, 1 + (p[2] >= p[1]), 0)  # sure it is water -> whichever water class is likelier


def evaluate(model, files, mean, std, device, tta=False, keep=0, conf=0.0):
    """Score tiles one at a time (read from disk, so memory stays small). Returns the confusion matrices and examples."""
    model.eval()
    cm_raw, cm_map, cm_base, examples = np.zeros((3, 3), np.int64), np.zeros((3, 3), np.int64), np.zeros((3, 3), np.int64), []
    for tid, radar, lf, valid, pw in files:
        x, y = read_tile(radar, lf, valid)
        xs = scaled(x, mean, std)
        raw = predict_one(model, xs, device, tta, conf)
        v = y != IGNORE
        cm_raw += confusion(raw, y, v)
        if pw:
            with rasterio.open(pw) as f:
                m = f.read(1) == 1
            final = np.where(raw > 0, np.where(m, 1, 2), 0)  # the map decides which water: inside = permanent, outside = flood
            dark = np.isfinite(x[1]) & (x[1] < BASE_DB)  # simple method: dark in after-flood VH
            cm_map += confusion(final, y, v); cm_base += confusion(np.where(dark, np.where(m, 1, 2), 0), y, v)
        else:
            final = raw
        if keep:
            examples.append(((y == 2).sum(), tid, xs[1], y, final))
            examples = sorted(examples, key=lambda e: -e[0])[:keep]  # the most-flooded tiles, for the figure
    return cm_raw, cm_map if cm_map.sum() else None, cm_base if cm_base.sum() else None, examples


BASE_DB = -22.5  # the dB cut-off the dark-pixel method picked on the training tiles in earlier runs


def confusion(pred, y, v):
    return np.bincount(y[v].astype(np.int64) * 3 + pred[v], minlength=9).reshape(3, 3)  # rows = truth, columns = prediction


def scores(cm):
    tp = cm[2, 2]; fp = cm[:, 2].sum() - tp; fn = cm[2, :].sum() - tp
    total = cm.sum()
    right = total - fp - fn  # flood vs not-flood
    water_tp = cm[1:, 1:].sum()
    r = {
        "flood_precision": tp / max(tp + fp, 1),  # of the pixels called flooded, how many really were
        "flood_recall": tp / max(tp + fn, 1),  # of the really flooded pixels, how many were found
        "flood_iou": tp / max(tp + fp + fn, 1),
        "flood_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "accuracy_3_class": np.trace(cm) / total,  # background / permanent water / flood all right
        "accuracy_flood_vs_not": right / total,
        "error_rate_flood_vs_not": 1 - right / total,
        "any_water_iou": water_tp / max(cm[1:, :].sum() + cm[:, 1:].sum() - water_tp, 1),  # flood + permanent water together
        "always_no_flood_accuracy": 1 - cm[2, :].sum() / total,  # what you'd get by never predicting flood
        "flood_share_of_pixels": cm[2, :].sum() / total,
        "pixels_scored": int(total),
    }
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}


def save_figure(examples, path):
    cmap = ListedColormap(["#dddddd", "#1f4fd8", "#ff7f0e", "#000000"])
    fig, axes = plt.subplots(len(examples), 3, figsize=(9, 3 * len(examples)), squeeze=False)
    for row, (_, tid, vh, y, pred) in enumerate(examples):
        axes[row, 0].imshow(vh, cmap="gray", vmin=-3, vmax=3)
        axes[row, 1].imshow(np.where(y == IGNORE, 3, y), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        axes[row, 2].imshow(np.where(y == IGNORE, 3, pred), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        axes[row, 0].set_ylabel(tid)
        for c, t in enumerate(["after radar (VH)", "truth", "prediction"]):
            axes[row, c].set_title(t, fontsize=9); axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", action="store_true", help="also give the model the before-flood radar (4 bands)")
    ap.add_argument("--dem", action="store_true", help="also give the model the ground height above the lowest point nearby (1 more band)")
    ap.add_argument("--conf", type=float, default=0.0, help="when scoring, only call water if the model is at least this sure (e.g. 0.7): precision up, recall down")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=50, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay: pulls weights toward zero, against memorising the training tiles")
    ap.add_argument("--cosine", action="store_true", help="lower the learning rate smoothly to zero over this run")
    ap.add_argument("--val-every", type=int, default=2, help="score the validation tiles every N epochs")
    ap.add_argument("--eval", choices=list(EVAL_SETS), default="val", help="tiles to score at the end")
    ap.add_argument("--heldout", action="store_true", help="same as --eval heldout")
    ap.add_argument("--tta", action="store_true", help="average over 8 flips and rotations when scoring")
    ap.add_argument("--sample-only", action="store_true", help="train on the 18 sample tiles only, ignoring data/full")
    ap.add_argument("--resume", action="store_true", help="continue training the saved model")
    ap.add_argument("--test-only", action="store_true", help="score the saved model without training")
    ap.add_argument("--ckpt", help="model file (default data/flood_unet.pt, plus _pre and _dem for those inputs)")
    a = ap.parse_args()
    eval_name = "heldout" if a.heldout else a.eval
    ckpt_path = a.ckpt or f"data/flood_unet{'_pre' if a.pre else ''}{'_dem' if a.dem else ''}.pt"
    tag = (os.path.splitext(os.path.basename(ckpt_path))[0].replace("flood_unet", "") + f"_{eval_name}" + ("_tta" if a.tta else "")
           + (f"_conf{round(a.conf * 100)}" if a.conf else ""))
    results, figure = f"data/flood_results{tag}.json", f"data/flood_examples{tag}.png"

    torch.manual_seed(0); rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    bands = (4 if a.pre else 2) + a.dem
    inputs = ("after + before" if a.pre else "after only") + (" + ground height" if a.dem else "")
    print(f"device: {device}   inputs: {inputs} ({bands} bands)", flush=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False) if (a.resume or a.test_only) else None
    if ckpt and len(ckpt["mean"]) != bands:
        raise SystemExit(f"{ckpt_path} was trained with {len(ckpt['mean'])} bands, not {bands}; check --pre / --dem")
    model = new_model(device, bands)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)  # the same as Adam when --wd is 0
    history, done, best = [], 0, {"flood_iou": -1}
    if ckpt:
        mean, std = np.array(ckpt["mean"], "float32"), np.array(ckpt["std"], "float32")  # keep the scaling the model learned with
        model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["opt"])
        for g in opt.param_groups: g["lr"], g["weight_decay"] = a.lr, a.wd  # the saved optimizer carries its old settings; --lr and --wd win on resume
        history, done = ckpt["history"], ckpt["epochs_done"]
        best = ckpt.get("best_val", best)
        print(f"loaded {ckpt_path}: {done} epochs trained" + (f", val flood IoU {best['flood_iou']:.3f}" if best["flood_iou"] >= 0 else ""), flush=True)

    if not a.test_only:
        train_dirs = [f"{ROOT}/{TRAIN_EVENT}"] + ([] if a.sample_only else sorted(glob.glob(f"{FULL}/*")))
        train_files, val_files = tile_files(train_dirs, a.pre, a.dem), tile_files(EVAL_SETS["val"](), a.pre, a.dem)
        n_floods = len({f[0].split("-")[0] for f in train_files})
        print(f"train: {len(train_files)} tiles from {n_floods} floods   val: {len(val_files)} tiles", flush=True)
        if not ckpt:
            mean, std = band_stats(train_files)  # training tiles only
        print("scaling mean", mean.round(2), "std", std.round(2), flush=True)
        train = load_train(train_files, mean, std)

        ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
        dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE)  # rewards overlap, so the small flood class counts
        t0 = time.time()
        for ep in range(done, done + a.epochs):
            model.train(); total = 0.0
            if a.cosine:
                for g in opt.param_groups: g["lr"] = a.lr * 0.5 * (1 + np.cos(np.pi * (ep - done) / a.epochs))  # big steps first, small steps at the end
            for _ in range(a.steps):
                x, y = sample_batch(train, rng, a.batch)
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = ce(out, y) + dice(out, y)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            history.append(total / a.steps)
            line = f"epoch {ep + 1:3d}  loss {history[-1]:.4f}"
            if val_files and ((ep + 1) % a.val_every == 0 or ep + 1 == done + a.epochs):
                _, cm_map, _, _ = evaluate(model, val_files, mean, std, device)
                s = scores(cm_map) if cm_map is not None else {"flood_iou": 0}
                line += f"  val flood IoU {s['flood_iou']:.3f}"
                if s["flood_iou"] > best["flood_iou"]:
                    best = dict(s, epoch=ep + 1)
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "mean": mean.tolist(), "std": std.tolist(),
                                "history": history, "epochs_done": ep + 1, "best_val": best}, ckpt_path)  # only the best round is kept
                    line += "  <- best, saved"
            print(f"{line}  ({time.time() - t0:.0f}s)", flush=True)
        if not val_files:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "mean": mean.tolist(), "std": std.tolist(),
                        "history": history, "epochs_done": done + a.epochs, "best_val": best}, ckpt_path)
        del train
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)  # score the best round, not the last one
        model.load_state_dict(ck["model"]); done = ck["epochs_done"]
        print(f"best round: epoch {done}", flush=True)

    files = tile_files(EVAL_SETS[eval_name](), a.pre, a.dem)
    floods = len({f[0].split("-")[0] for f in files})
    cm_raw, cm_map, cm_base, examples = evaluate(model, files, mean, std, device, tta=a.tta, keep=4, conf=a.conf)
    out = {"model": ckpt_path, "inputs": inputs, "tta": a.tta, "min_confidence": a.conf, "eval_set": eval_name,
           "eval_tiles": len(files), "eval_floods": floods, "epochs_trained": done, "best_val": best,
           "model_alone": scores(cm_raw)}
    if cm_map is not None:
        out["final"] = scores(cm_map)  # model + permanent-water map: the number to report
        out["dark_pixel_baseline_with_map"] = scores(cm_base)
    json.dump(out, open(results, "w"), indent=2)
    save_figure(examples, figure)

    s, m = out.get("final", out["model_alone"]), out["model_alone"]
    print(f"\nScore on {eval_name}: {len(files)} tiles from {floods} floods the model never trained on"
          f"{'  (8-way flip/rotate average)' if a.tta else ''}")
    print(f"  model alone:         flood precision {m['flood_precision']:.1%}  recall {m['flood_recall']:.1%}  IoU {m['flood_iou']:.3f}")
    if cm_map is not None:
        b = out["dark_pixel_baseline_with_map"]
        print(f"  model + water map:   flood precision {s['flood_precision']:.1%}  recall {s['flood_recall']:.1%}  IoU {s['flood_iou']:.3f}")
        print(f"  dark-pixel + map:    flood precision {b['flood_precision']:.1%}  recall {b['flood_recall']:.1%}  IoU {b['flood_iou']:.3f}")
    print(f"  accuracy (flood vs not) {s['accuracy_flood_vs_not']:.1%}   error rate {s['error_rate_flood_vs_not']:.1%}"
          f"   ('never say flood' would score {s['always_no_flood_accuracy']:.1%})")
    print(f"saved {results}, {figure}")


if __name__ == "__main__":
    main()
