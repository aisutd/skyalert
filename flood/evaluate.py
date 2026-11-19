"""Score a flood bundle on held-out events and write the evaluation report into the bundle.

    python -m flood.evaluate                     # latest bundle, test + heldout splits
    python -m flood.evaluate --bundle data/flood/bundles/flood-unet-...

Anything tuned (the permanent-water post-processing choice, the radar threshold baseline) is
tuned on val only. Test and heldout are scored once with those choices fixed.
Writes <bundle>/evaluation/{report.json, report.md, examples_<split>.png}.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from . import bundle as bundles
from . import contract
from .data import load_scaled, read_manifest
from .metrics import confusion, evaluate, flood_scores
from .model import predict_classes

THRESHOLDS_DB = np.arange(-30.0, -11.5, 0.5)


def with_permwater(pred, permwater):
    """Water the model found, reassigned by the static reference: inside it permanent, outside it flood."""
    return np.where(pred > 0, np.where(permwater, contract.PERMANENT_WATER, contract.FLOOD), contract.BACKGROUND).astype("uint8")


def vh_db(s, mean, std):
    return s["z"][1].astype("float32") * std[1] + mean[1]


def threshold_pred(s, t_db, mean, std):
    """Classic SAR baseline: dark VH is water; water outside the permanent-water reference is flood."""
    water = (vh_db(s, mean, std) < t_db) & s["valid"]
    return with_permwater(water.astype("uint8"), s["permwater"])


def tune_threshold(val, mean, std):
    def iou(t):
        return flood_scores(sum(confusion(threshold_pred(s, t, mean, std), s["y"]) for s in val))["flood_iou"]
    scores = {float(t): iou(t) for t in THRESHOLDS_DB}
    best = max(scores, key=scores.get)
    return best, scores[best]


def figure(samples, preds, path, mean, std, title):
    """Two best and the worst flooded tiles, plus the worst dry-scene false alarm if there is one."""
    wet = [(i, flood_scores(confusion(p, s["y"]))) for i, (p, s) in enumerate(zip(preds, samples)) if (s["y"] == contract.FLOOD).any()]
    wet.sort(key=lambda t: -t[1]["flood_iou"])
    picks = [(i, "success") for i, _ in wet[:2]] + [(i, "failure") for i, _ in wet[-1:]]
    dry = [(i, int((p == contract.FLOOD).sum())) for i, (p, s) in enumerate(zip(preds, samples)) if not (s["y"] == contract.FLOOD).any()]
    if dry:
        i, n = max(dry, key=lambda t: t[1])
        picks.append((i, "dry scene" + (" false alarm" if n else ", no alarm")))
    cmap = ListedColormap(["#d9d9d9", "#1f4fd8", "#ff7f0e", "#000000"])
    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)), squeeze=False)
    for row, (i, kind) in enumerate(picks):
        s, p = samples[i], preds[i]
        axes[row, 0].imshow(np.where(s["valid"], vh_db(s, mean, std), np.nan), cmap="gray", vmin=-28, vmax=-8)
        axes[row, 1].imshow(np.where(s["y"] == contract.IGNORE, 3, s["y"]), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        axes[row, 2].imshow(np.where(s["y"] == contract.IGNORE, 3, p), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        iou = flood_scores(confusion(p, s["y"]))["flood_iou"]
        axes[row, 0].set_ylabel(f"{s['sample_id']}\n{kind}", fontsize=8)
        for c, t in enumerate(["post-event VH (dB)", "label", f"prediction (flood IoU {iou:.2f})"]):
            axes[row, c].set_title(t, fontsize=8); axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
    fig.suptitle(f"{title}   grey background, blue permanent water, orange flood, black ignored", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close(fig)


def fmt(v, pct=False):
    if v is None:
        return "n/a"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def markdown(rep):
    b = rep["bundle"]
    lines = [
        f"# Flood detection evaluation: `{b['model_version']}`", "",
        f"Generated {rep['generated_utc']} from code `{rep['code_revision']}`. Dataset {b['dataset']}, split version `{b['split_version']}`.",
        "Input: one post-event Sentinel-1 GRD image (VV, VH). Output: per-pixel background / permanent water / flooded water at the acquisition time. "
        "This is observed flood extent, not a forecast, and a historical evaluation does not establish live coverage.", "",
        f"Selected on val: **{rep['chosen_variant']}** (val flood IoU: model alone {fmt(rep['val']['model_flood_iou'])}, "
        f"with permanent-water reference {fmt(rep['val']['model_permwater_flood_iou'])}). "
        f"Radar threshold baseline tuned on val: VH < {rep['val']['threshold_db']} dB (val flood IoU {fmt(rep['val']['threshold_flood_iou'])}).", "",
    ]
    for split, r in rep["splits"].items():
        m = r["methods"]
        lines += [f"## {split}: {r['description']}", "",
                  f"{m['model']['tiles']} tiles, {m['model']['events']} events ({m['model']['events_with_flood']} with flood). "
                  f"Valid coverage {fmt(m['model']['valid_coverage'], True)}. "
                  "Tiles were selected with flooded tiles over-represented (see manifest_meta.json), so pooled pixel metrics describe this sample, not a natural flood prevalence.", "",
                  "| method | flood IoU | Dice | precision | recall | mean event IoU | dry-scene alarms | permanent water called flood | flood called permanent water | flood km² true / predicted |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, label in [("chosen", f"**U-Net ({rep['chosen_variant']})**"), ("model", "U-Net alone"), ("model_permwater", "U-Net + permanent-water reference"),
                            ("threshold", f"VH < {rep['val']['threshold_db']} dB + reference"), ("never_flood", "never flood")]:
            e = m[name]; p = e["pooled"]; d = e["dry_scenes"]
            area = e["area_km2"]
            area_s = f"{area['true']} / {area['predicted']}" if isinstance(area, dict) else "n/a"
            lines.append(f"| {label} | {fmt(p['flood_iou'])} | {fmt(p['flood_dice'])} | {fmt(p['flood_precision'], True)} | {fmt(p['flood_recall'], True)} | "
                         f"{fmt(e['mean_event_flood_iou'])} | {d['with_false_alarm']}/{d['count']} | {fmt(p['permwater_called_flood'], True)} | "
                         f"{fmt(p['flood_called_permwater'], True)} | {area_s} |")
        lines += ["", f"Dry-scene alarm: at least {m['model']['dry_scenes']['alarm_threshold_km2']} km² mapped as flood on a tile with no labelled flood.", "",
                  f"![{split} examples](examples_{split}.png)", "", "Per event (chosen variant):", "",
                  "| event | tiles | flood IoU | Dice | flood px true | flood px predicted |", "|---|---|---|---|---|---|"]
        for ev in m["chosen"]["per_event"]:
            lines.append(f"| {ev['event_id']} | {ev['tiles']} | {fmt(ev['flood_iou']) if ev['flood_px_true'] else 'dry'} | "
                         f"{fmt(ev['flood_dice']) if ev['flood_px_true'] else '-'} | {ev['flood_px_true']} | {ev['flood_px_pred']} |")
        lines.append("")
    lines += ["## Limitations", "",
              "- Single post-event image: flood vs permanent water relies on what the model learned about river and lake context, or on the static permanent-water reference. No pre-event change detection.",
              "- Labels are CEMS Rapid Mapping delineations; label is not masked under cloud, but radar sees through cloud.",
              "- The source dataset is provisional pending the T2 dataset decision; a different source needs its own adapter and re-evaluation.",
              "- Scores are argmax classes, not calibrated probabilities.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle")
    ap.add_argument("--manifest", default="data/flood/geoid/manifest.csv")
    ap.add_argument("--splits", nargs="+", default=["test", "heldout"])
    a = ap.parse_args()

    dev = bundles.device()
    path = Path(a.bundle) if a.bundle else bundles.latest()
    model, meta, mean, std = bundles.load(path, dev)
    manifest_path = Path(a.manifest)
    root = manifest_path.parent
    manifest = read_manifest(manifest_path)
    out = path / "evaluation"; out.mkdir(exist_ok=True)
    print(f"bundle {meta['model_version']} on {dev}", flush=True)

    val = load_scaled(manifest, "val", root, mean, std)
    val_raw = [predict_classes(model, s["z"], dev) for s in val]
    iou = lambda preds, ss: flood_scores(sum(confusion(p, s["y"]) for p, s in zip(preds, ss)))["flood_iou"]
    v_model = iou(val_raw, val)
    v_pw = iou([with_permwater(p, s["permwater"]) for p, s in zip(val_raw, val)], val)
    chosen = "model_permwater" if v_pw > v_model else "model"
    t_db, t_iou = tune_threshold(val, mean, std)
    print(f"val flood IoU: model {v_model:.3f}, +permwater {v_pw:.3f} -> {chosen}; threshold {t_db} dB IoU {t_iou:.3f}", flush=True)
    del val, val_raw

    descriptions = {"test": "held-out events from the main tree (event + touching-area groups never seen in training)",
                    "heldout": "GEOID-Flood held-out activations EMSR857-871 (Jan-Mar 2026), later than every training event; tiles overlapping main-tree ground removed"}
    report = {"bundle": {k: meta[k] for k in ("model_version", "dataset", "split_version", "preprocessing_version", "code_revision")},
              "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code_revision": contract.code_revision(),
              "chosen_variant": "U-Net + permanent-water reference" if chosen == "model_permwater" else "U-Net alone",
              "val": {"model_flood_iou": round(v_model, 4), "model_permwater_flood_iou": round(v_pw, 4), "threshold_db": t_db, "threshold_flood_iou": round(t_iou, 4)},
              "splits": {}}
    for split in a.splits:
        samples = load_scaled(manifest, split, root, mean, std)
        if not samples:
            print(f"{split}: no samples in manifest, skipped")
            continue
        raw = [predict_classes(model, s["z"], dev) for s in samples]
        methods = {
            "model": raw,
            "model_permwater": [with_permwater(p, s["permwater"]) for p, s in zip(raw, samples)],
            "threshold": [threshold_pred(s, t_db, mean, std) for s in samples],
            "never_flood": [np.where(s["y"] == contract.FLOOD, contract.BACKGROUND, s["y"]).astype("uint8") for s in samples],
        }
        methods["chosen"] = methods[chosen]
        res = {name: evaluate(preds, samples) for name, preds in methods.items()}
        report["splits"][split] = {"description": descriptions.get(split, split), "methods": res}
        figure(samples, methods["chosen"], out / f"examples_{split}.png", mean, std, f"{split}: {report['chosen_variant']}")
        p = res["chosen"]["pooled"]; d = res["chosen"]["dry_scenes"]
        print(f"{split}: {len(samples)} tiles / {res['chosen']['events']} events  flood IoU {p['flood_iou']:.3f}  Dice {p['flood_dice']:.3f}  "
              f"P {fmt(p['flood_precision'], True)}  R {fmt(p['flood_recall'], True)}  dry alarms {d['with_false_alarm']}/{d['count']}  "
              f"(threshold baseline IoU {res['threshold']['pooled']['flood_iou']:.3f})", flush=True)
        del samples, raw, methods

    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "report.md").write_text(markdown(report))
    meta["evaluation_report"] = "evaluation/report.json"
    meta["chosen_variant"] = chosen
    (path / "bundle.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
