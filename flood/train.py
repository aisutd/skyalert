"""Train the flood U-Net on the manifest's train split, pick the checkpoint on val, save a model bundle.

    python -m flood.train                                  # data/flood/geoid/manifest.csv -> data/flood/bundles/<version>/
    python -m flood.train --epochs 5 --steps 20            # quick smoke run

The bundle holds weights.pt and bundle.json (architecture, ordered input schema, normalization,
preprocessing version, label config, dataset/split versions, code revision, training history).
Test and held-out tiles are never read here; run `python -m flood.evaluate` afterwards.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import contract
from .data import fit_band_stats, iter_split, load_scaled, read_manifest
from .metrics import confusion, flood_scores
from .model import UNet, predict_classes

CROP = 256


def sample_batch(samples, flood_px, rng, n):
    """n random crops, half centred near a flood pixel; image and label rotated/flipped together."""
    xs, ys = [], []
    for _ in range(n):
        k = int(rng.integers(len(samples)))
        z, y = samples[k]["z"], samples[k]["y"]
        H, W = y.shape
        if rng.random() < 0.5 and len(flood_px[k]):
            r, c = flood_px[k][rng.integers(len(flood_px[k]))]
            r = int(np.clip(r - CROP // 2 + rng.integers(-64, 65), 0, H - CROP))
            c = int(np.clip(c - CROP // 2 + rng.integers(-64, 65), 0, W - CROP))
        else:
            r, c = int(rng.integers(0, H - CROP + 1)), int(rng.integers(0, W - CROP + 1))
        cz, cy = z[:, r:r + CROP, c:c + CROP], y[r:r + CROP, c:c + CROP]
        t = int(rng.integers(4))
        cz, cy = np.rot90(cz, t, axes=(1, 2)), np.rot90(cy, t)
        if rng.random() < 0.5:
            cz, cy = cz[:, :, ::-1], cy[:, ::-1]
        xs.append(cz.astype("float32")); ys.append(cy.astype("int64"))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def flood_dice_loss(logits, y):
    """Soft Dice on the flood class over labelled pixels, so the rare class carries weight."""
    valid = (y != contract.IGNORE).float()
    p = torch.softmax(logits, 1)[:, contract.FLOOD] * valid
    t = (y == contract.FLOOD).float() * valid
    return 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)


def val_scores(model, samples, device):
    cm = sum(confusion(predict_classes(model, s["z"], device), s["y"]) for s in samples)
    return flood_scores(cm)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/flood/geoid/manifest.csv")
    ap.add_argument("--out", default="data/flood/bundles")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=60, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16, help="U-Net width")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    manifest_path = Path(a.manifest)
    root = manifest_path.parent
    manifest = read_manifest(manifest_path)
    meta_path = root / "manifest_meta.json"
    mmeta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    mean, std = fit_band_stats(iter_split(manifest, "train", root))  # training tiles only
    print(f"device {device}; normalization from train: mean {mean.round(3)} std {std.round(3)}", flush=True)
    train = load_scaled(manifest, "train", root, mean, std)
    val = load_scaled(manifest, "val", root, mean, std)
    print(f"train {len(train)} tiles / {len({s['event_id'] for s in train})} events; "
          f"val {len(val)} tiles / {len({s['event_id'] for s in val})} events", flush=True)

    model = UNet(in_channels=len(contract.INPUT_SCHEMA["bands"]), classes=len(contract.CLASSES), base=a.base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ce = nn.CrossEntropyLoss(ignore_index=contract.IGNORE)
    flood_px = [np.argwhere(s["y"] == contract.FLOOD) for s in train]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(a.out) / f"flood-unet-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    history, best, since, t0 = [], -1.0, 0, time.time()
    for epoch in range(1, a.epochs + 1):
        model.train(); losses = []
        for _ in range(a.steps):
            x, y = sample_batch(train, flood_px, rng, a.batch)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y) + flood_dice_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        v = val_scores(model, val, device)
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5),
                        **{f"val_{k}": (round(v[k], 4) if v[k] is not None else None) for k in ("flood_iou", "flood_precision", "flood_recall", "any_water_iou")}})
        improved = v["flood_iou"] > best
        print(f"epoch {epoch:3d}  loss {np.mean(losses):.4f}  val flood IoU {v['flood_iou']:.3f}  "
              f"P {v['flood_precision'] or 0:.3f}  R {v['flood_recall'] or 0:.3f}  ({time.time() - t0:.0f}s){'  saved' if improved else ''}", flush=True)
        if improved:
            best, since = v["flood_iou"], 0
            torch.save(model.state_dict(), out / "weights.pt")
        else:
            since += 1
            if since >= a.patience:
                print(f"no val improvement for {a.patience} epochs, stopping", flush=True)
                break

    weights_sha = hashlib.sha256((out / "weights.pt").read_bytes()).hexdigest()
    best_epoch = max(history, key=lambda h: h["val_flood_iou"])
    bundle = {
        "model_version": f"flood-unet-{stamp}-{weights_sha[:8]}",
        "hazard": contract.HAZARD, "task": contract.TASK,
        "architecture": model.config,
        "input_schema": contract.INPUT_SCHEMA,
        "normalization": {"mean": mean.tolist(), "std": std.tolist(), "fitted_on": "train split valid pixels, dB"},
        "preprocessing_version": contract.PREPROCESSING_VERSION,
        "labels": {"classes": contract.CLASSES, "ignore_index": contract.IGNORE, "semantics": contract.LABEL_SEMANTICS},
        "decision": "argmax over the 3 classes; no probability calibration, so scores are not probabilities",
        "dataset": mmeta.get("dataset", "unknown"),
        "split_version": mmeta.get("split_version", "unknown"),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "train_tiles": len(train), "val_tiles": len(val),
        "train_events": sorted({s["event_id"] for s in train}), "val_events": sorted({s["event_id"] for s in val}),
        "code_revision": contract.code_revision(),
        "training": {**vars(a), "device": device, "crop": CROP, "loss": "cross-entropy (ignore 255) + soft Dice on flood",
                     "selected_epoch": best_epoch["epoch"], "selected_on": "val flood IoU", "seconds": round(time.time() - t0)},
        "weights_file": "weights.pt", "weights_sha256": weights_sha,
        "history": history,
        "evaluation_report": None,
        "created_utc": stamp,
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=2))
    print(f"saved {out}  model_version {bundle['model_version']}  best val flood IoU {best:.3f} (epoch {best_epoch['epoch']})")


if __name__ == "__main__":
    main()
