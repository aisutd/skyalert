"""Train the v2 next-day spread model on data/retrain/ (docs/spread_retrain_plan.md, R3/R4).

Mac safety (the plan's hard rules) is enforced here, not left to the caller:
- the process lowers its own priority to nice 10;
- at most 4 CPU threads (torch, OpenMP, Accelerate);
- MPS may use at most half the GPU memory (PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5);
- batches of at most 64 tiles; the dataset is memory-mapped from disk, never loaded whole;
- the run stops after --max-minutes (30 by default), or if the process passes MAX_MEMORY_GB.

Model selection: after every epoch the val split is predicted, the threshold that maximizes
pooled val IoU is chosen, and that best val IoU is the early-stopping score (IoU is what the
model is judged on). The test split is never read here.

    nice -n 10 python -m training.train_spread_v2 --name bce_pw10 --pos-weight 10
"""

import os

# Thread and GPU-memory caps must be set before numpy / torch start
for _var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "4"
_ratio = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
if _ratio is None or float(_ratio) > 0.5 or float(_ratio) == 0.0:  # 0.0 would mean "no limit"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.5"
# PyTorch requires the low watermark (default 1.4) to be at or below the high one
if float(os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO", "1.4")) > float(os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]):
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.4"

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from backend import spread_model_v2 as v2  # noqa: E402
from training.evaluate_v2 import average_precision, choose_threshold, pooled_scores  # noqa: E402

MAX_BATCH = 64
MAX_THREADS = 4
MAX_MEMORY_GB = 6.0
CHANNEL_SETS = {
    "all": v2.CHANNELS,
    "no-history": tuple(name for name in v2.CHANNELS if name not in ("fire_t1", "fire_t2")),
}


def apply_safety():
    """Lower priority to nice 10 (if not already lower) and cap torch threads."""
    current = os.nice(0)
    if current < 10:
        os.nice(10 - current)
    torch.set_num_threads(MAX_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # can only be set once per process, before any parallel work
        pass


def peak_memory_gb():
    """Peak resident memory of this process (macOS reports bytes, Linux kilobytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


class TileDataset(Dataset):
    """Memory-mapped tiles; float16 on disk, float32 per item, optional random symmetry."""

    def __init__(self, data_dir, split, channels, stats, augment=False, limit=None, seed=0):
        self.inputs = np.load(Path(data_dir) / f"{split}_inputs.npy", mmap_mode="r")
        self.labels = np.load(Path(data_dir) / f"{split}_labels.npy", mmap_mode="r")
        if self.inputs.shape[1] != len(v2.CHANNELS):
            raise ValueError(f"{split}_inputs.npy has {self.inputs.shape[1]} channels, expected {len(v2.CHANNELS)}")
        self.n = len(self.labels) if limit is None else min(limit, len(self.labels))
        self.channels, self.stats, self.augment = tuple(channels), stats, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = np.asarray(self.inputs[i], dtype=np.float32)  # all channels, so wind is found for symmetry
        y = np.asarray(self.labels[i], dtype=np.float32)
        if self.augment:
            x, y = v2.transform(x, y, int(self.rng.integers(v2.N_TRANSFORMS)))
        x = v2.normalize(v2.select_channels(x, self.channels), self.stats, self.channels)
        return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(y)[None])


def dice_loss(logits, target):
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum()
    return 1 - (2 * intersection + 1) / (probabilities.sum() + target.sum() + 1)


def loss_fn(logits, target, pos_weight, dice_weight):
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(pos_weight, device=logits.device))
    if dice_weight:
        loss = loss + dice_weight * dice_loss(logits, target)
    return loss


@torch.no_grad()
def predict_split(model, dataset, device, batch_size):
    model.eval()
    out = np.empty((len(dataset), v2.TILE_SIZE, v2.TILE_SIZE), dtype=np.float32)
    position = 0
    for x, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        p = torch.sigmoid(model(x.to(device)))[:, 0].float().cpu().numpy()
        out[position : position + len(p)] = p
        position += len(p)
    return out


def init_from_ndws(model, channels):
    """Copy the old NDWS weights into a v2 model; first-layer weights for v2 channels with an NDWS
    counterpart are copied, the rest start at zero (so new channels begin with no effect)."""
    from backend import spread_model

    old = torch.load(spread_model.WEIGHTS_PATH, map_location="cpu", weights_only=True)
    counterpart = {
        "elevation": "elevation", "tmmn": "tmmn", "tmmx": "tmmx", "sph": "sph", "pr": "pr",
        "erc": "erc", "pdsi": "pdsi", "fire_t": "PrevFireMask",
    }
    state = model.state_dict()
    for key, value in old.items():
        if key == "down.0.0.weight":
            first = torch.zeros_like(state[key])
            for i, name in enumerate(channels):
                if name in counterpart:
                    first[:, i] = value[:, spread_model.INPUT_FEATURES.index(counterpart[name])]
            state[key] = first
        else:
            state[key] = value
    model.load_state_dict(state, strict=True)
    return model


def train(args):
    apply_safety()
    if args.batch_size > MAX_BATCH:
        raise SystemExit(f"--batch-size is capped at {MAX_BATCH} (Mac safety rule)")
    if args.max_minutes > 30:
        raise SystemExit("--max-minutes is capped at 30 (Mac safety rule)")
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    stats_path = data_dir / "channel_stats.csv"
    channels = CHANNEL_SETS.get(args.channels) or tuple(args.channels.split(","))
    stats = v2.load_stats(stats_path, channels)
    device = torch.device("mps" if torch.backends.mps.is_available() and not args.cpu else "cpu")

    train_set = TileDataset(data_dir, "train", channels, stats, augment=not args.no_augment, limit=args.train_limit, seed=args.seed)
    val_set = TileDataset(data_dir, "val", channels, stats, augment=False, limit=args.val_limit)
    val_labels = np.asarray(val_set.labels[: len(val_set)]) > 0
    val_fire_t = np.asarray(val_set.inputs[: len(val_set), v2.CHANNELS.index("fire_t")], dtype=np.float32)

    model = v2.build_model(channels)
    if args.init_ndws:
        model = init_from_ndws(model, channels)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed))

    out = Path(args.runs_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.csv"
    started = time.time()
    best, best_epoch, since_best, history = -1.0, 0, 0, []
    stop_reason = "max epochs"
    print(f"device {device}, {len(train_set)} train / {len(val_set)} val tiles, channels {channels}", flush=True)
    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.time()
        model.train()
        losses = []
        for x, y in loader:
            loss = loss_fn(model(x.to(device)), y.to(device), args.pos_weight, args.dice_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        probabilities = predict_split(model, val_set, device, args.batch_size)
        threshold, val_iou = choose_threshold(probabilities, val_labels)
        scores = pooled_scores(probabilities >= threshold, val_labels, val_fire_t)
        row = {
            "epoch": epoch, "train_loss": float(np.mean(losses)), "val_iou": val_iou, "val_threshold": threshold,
            "val_pr_auc": average_precision(probabilities, val_labels),
            "val_precision": scores["precision"], "val_recall": scores["recall"],
            "val_new_fire_recall": scores["new_fire_recall"], "val_forecast_px": scores["forecast_px"],
            "val_burned_px": scores["burned_px"], "seconds": time.time() - epoch_start,
            "peak_memory_gb": peak_memory_gb(),
        }
        history.append(row)
        with log_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        improved = val_iou > best
        print(
            f"epoch {epoch:2d} loss {row['train_loss']:.4f} val IoU {val_iou:.4f} @ {threshold:.3f} "
            f"PR-AUC {row['val_pr_auc']:.4f} new-fire recall {row['val_new_fire_recall']:.3f} "
            f"{row['seconds']:.0f}s mem {row['peak_memory_gb']:.2f} GB{'  saved' if improved else ''}",
            flush=True,
        )
        if improved:
            best, best_epoch, since_best = val_iou, epoch, 0
            v2.save_run(out, model, stats_path, {
                "channels": list(channels), "threshold": threshold, "best_epoch": epoch,
                "val": {k: v for k, v in row.items() if k.startswith("val_")},
                "settings": vars(args), "created": dt.datetime.now().isoformat(timespec="seconds"),
                "n_train": len(train_set), "n_val": len(val_set), "device": str(device),
            })
        else:
            since_best += 1
        minutes = (time.time() - started) / 60
        if since_best >= args.patience:
            stop_reason = f"no val IoU gain for {args.patience} epochs"
            break
        if minutes + row["seconds"] / 60 > args.max_minutes:
            stop_reason = f"time budget ({args.max_minutes} min)"
            break
        if row["peak_memory_gb"] > MAX_MEMORY_GB:
            stop_reason = f"memory above {MAX_MEMORY_GB} GB"
            break
    summary = {
        "stop_reason": stop_reason, "best_epoch": best_epoch, "best_val_iou": best,
        "minutes": (time.time() - started) / 60, "peak_memory_gb": peak_memory_gb(), "epochs_run": len(history),
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text())
        manifest["training_summary"] = summary
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"done: {summary}", flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="run folder name under --runs-dir")
    parser.add_argument("--data-dir", default=str(v2.RETRAIN_DIR))
    parser.add_argument("--runs-dir", default=str(v2.RUNS_DIR))
    parser.add_argument("--channels", default="all", help="all, no-history, or a comma list of CHANNELS")
    parser.add_argument("--pos-weight", type=float, default=10.0)
    parser.add_argument("--dice-weight", type=float, default=0.0, help="0 = BCE only; e.g. 1.0 = BCE + Dice")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-minutes", type=float, default=30)
    parser.add_argument("--train-limit", type=int, default=None, help="use only the first N train tiles (smoke tests)")
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--init-ndws", action="store_true", help="start from the old NDWS weights where channels match")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
