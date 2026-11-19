"""Pick the best flood model on val, then score it once on held-out events.

    python -m flood.evaluate --bundles data/flood/bundles/A data/flood/bundles/B --manifest data/flood/geoid/manifest_v2.csv

For every candidate bundle and both decision variants (U-Net alone; U-Net water + permanent-water
reference) the flood threshold is tuned on val to maximize F2 (recall counts twice as much as
precision). The best (bundle, variant, threshold) by val F2 is fixed, then test and heldout are
scored once. The radar-threshold baseline is tuned on val the same way.
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
from .model import decide, predict_probs

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
        return flood_scores(sum(confusion(threshold_pred(s, t, mean, std), s["y"]) for s in val))["flood_f2"]
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
    b = rep["bundle"]; d = rep["decision"]
    lines = [
        f"# Flood detection evaluation: `{b['model_version']}`", "",
        f"Generated {rep['generated_utc']} from code `{rep['code_revision']}`. Dataset {b['dataset']}, split version `{b['split_version']}`.",
        "Input: one post-event Sentinel-1 GRD image (VV, VH" + (", plus a static permanent-water reference" if b.get("permwater_input") else "") +
        "). Output: per-pixel background / permanent water / flooded water at the acquisition time. "
        "This is observed flood extent, not a forecast, and a historical evaluation does not establish live coverage.", "",
        "## Selection on val (test not used)", "",
        f"Rule: maximize flood F2 (recall weighted 2x) on {rep['val_tiles']} val tiles from {rep['val_events']} events.", "",
        "| candidate | decision | threshold | val F2 | val IoU | val precision | val recall |", "|---|---|---|---|---|---|---|"]
    for c in rep["candidates"]:
        star = "**" if (c["model_version"], c["variant"]) == (d["model_version"], d["variant"]) else ""
        lines.append(f"| {star}{c['model_version']}{star} | {c['variant']} | {c['threshold']}/255 | {fmt(c['f2'])} | {fmt(c['iou'])} | {fmt(c['precision'], True)} | {fmt(c['recall'], True)} |")
    lines += ["", f"Radar baseline tuned on val the same way: VH < {rep['baseline_threshold_db']} dB, outside the permanent-water reference.", ""]
    for split, r in rep["splits"].items():
        m = r["methods"]
        lines += [f"## {split}: {r['description']}", "",
                  f"{m['chosen']['tiles']} tiles, {m['chosen']['events']} events ({m['chosen']['events_with_flood']} with flood). "
                  f"Valid coverage {fmt(m['chosen']['valid_coverage'], True)}. Flooded tiles are over-represented in the selection, "
                  "so pooled pixel metrics describe this sample, not natural flood prevalence. Accuracy is per labelled pixel and is dominated by background.", "",
                  "| method | recall | precision | flood IoU | Dice | F2 | accuracy (flood vs not) | accuracy (3 class) | dry-scene alarms | permanent water called flood | flood km² true / predicted |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, label in [("chosen", "**selected U-Net**"), ("baseline", f"VH < {rep['baseline_threshold_db']} dB + reference"), ("never_flood", "never flood")]:
            e = m[name]; p = e["pooled"]; dr = e["dry_scenes"]; area = e["area_km2"]
            area_s = f"{area['true']} / {area['predicted']}" if isinstance(area, dict) else "n/a"
            lines.append(f"| {label} | {fmt(p['flood_recall'], True)} | {fmt(p['flood_precision'], True)} | {fmt(p['flood_iou'])} | {fmt(p['flood_dice'])} | {fmt(p['flood_f2'])} | "
                         f"{fmt(p['accuracy_flood_vs_not'], True)} | {fmt(p['accuracy_3_class'], True)} | {dr['with_false_alarm']}/{dr['count']} | "
                         f"{fmt(p['permwater_called_flood'], True)} | {area_s} |")
        lines += ["", f"Dry-scene alarm: at least {m['chosen']['dry_scenes']['alarm_threshold_km2']} km² mapped as flood on a tile with no labelled flood.", "",
                  f"![{split} examples](examples_{split}.png)", "", "Per event (selected U-Net):", "",
                  "| event | tiles | flood IoU | flood px true | flood px predicted |", "|---|---|---|---|---|"]
        for ev in m["chosen"]["per_event"]:
            lines.append(f"| {ev['event_id']} | {ev['tiles']} | {fmt(ev['flood_iou']) if ev['flood_px_true'] else 'dry'} | {ev['flood_px_true']} | {ev['flood_px_pred']} |")
        lines.append("")
    lines += ["## Limitations", "",
              "- Single post-event image; no pre-event change detection.",
              "- The decision threshold favours recall, so precision and dry-scene false alarms are the price; read them together.",
              "- Labels are CEMS Rapid Mapping delineations. The source dataset is provisional pending the T2 decision.",
              "- Scores are uncalibrated softmax outputs, not probabilities.", ""]
    return "\n".join(lines)


def val_histograms(model, meta, mean, std, manifest, root, dev):
    """Per variant: 256-bin histograms of the decision score over flood-truth and other labelled pixels."""
    hist = {v: np.zeros((2, 256), np.int64) for v in ("model", "model_permwater")}
    n_tiles, events = 0, set()
    for s in load_iter(manifest, "val", root, mean, std, meta["permwater_input"]):
        probs = predict_probs(model, s["z"], dev).astype(np.int16)
        lab = s["y"] != contract.IGNORE
        fl = (s["y"] == contract.FLOOD) & lab
        other = lab & ~fl
        scores = {"model": (probs[1], lab), "model_permwater": (np.clip(probs[0] + probs[1], 0, 255), lab & ~s["permwater"])}
        for v, (sc, elig) in scores.items():
            hist[v][0] += np.bincount(sc[fl & elig], minlength=256)
            hist[v][1] += np.bincount(sc[other & elig], minlength=256)
            hist[v][0][0] += int((fl & ~elig).sum())  # never eligible: counted at score 0, below any threshold >= 1
        n_tiles += 1; events.add(s["event_id"])
    return hist, n_tiles, len(events)


def best_threshold(h):
    flood_total = h[0].sum()
    tp = np.cumsum(h[0][::-1])[::-1]; fp = np.cumsum(h[1][::-1])[::-1]  # pixels with score >= t
    best = None
    for t in range(1, 256):
        fn = flood_total - tp[t]
        f2 = 5 * tp[t] / max(5 * tp[t] + 4 * fn + fp[t], 1)
        if best is None or f2 > best["f2"]:
            best = {"threshold": t, "f2": f2, "iou": tp[t] / max(tp[t] + fp[t] + fn, 1),
                    "precision": tp[t] / max(tp[t] + fp[t], 1), "recall": tp[t] / max(flood_total, 1)}
    return {k: (round(float(v), 4) if k != "threshold" else v) for k, v in best.items()}


def load_iter(manifest, split, root, mean, std, permwater_input):
    from .data import iter_split, model_input
    for s in iter_split(manifest, split, root):
        s["z"] = model_input(s.pop("x"), mean, std, s["permwater"] if permwater_input else None)
        yield s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundles", nargs="+")
    ap.add_argument("--manifest", default="data/flood/geoid/manifest_v2.csv")
    ap.add_argument("--splits", nargs="+", default=["test", "heldout"])
    a = ap.parse_args()

    dev = bundles.device()
    paths = [Path(p) for p in a.bundles] if a.bundles else [bundles.latest()]
    manifest_path = Path(a.manifest)
    root = manifest_path.parent
    manifest = read_manifest(manifest_path)

    candidates = []
    for path in paths:
        model, meta, mean, std = bundles.load(path, dev)
        hist, n_val, n_ev = val_histograms(model, meta, mean, std, manifest, root, dev)
        for v, h in hist.items():
            c = {"bundle": str(path), "model_version": meta["model_version"], "variant": v, **best_threshold(h)}
            candidates.append(c)
            print(f"val {meta['model_version']} {v:16s} t={c['threshold']:3d}  F2 {c['f2']:.3f}  IoU {c['iou']:.3f}  P {c['precision']:.1%}  R {c['recall']:.1%}", flush=True)
        del model
    d = max(candidates, key=lambda c: c["f2"])
    path = Path(d["bundle"])
    model, meta, mean, std = bundles.load(path, dev)
    print(f"selected on val: {d['model_version']} {d['variant']} threshold {d['threshold']}/255", flush=True)

    val = list(load_iter(manifest, "val", root, mean, std, meta["permwater_input"]))
    for s in val:
        s.pop("profile", None)
    t_db, _ = tune_threshold(val, mean, std)
    del val

    out = path / "evaluation"; out.mkdir(exist_ok=True)
    descriptions = {"test": "held-out events from the main tree (event + touching-area groups never seen in training or selection)",
                    "heldout": "GEOID-Flood held-out activations EMSR857-871 (Jan-Mar 2026), later than every training event; tiles overlapping main-tree ground removed"}
    report = {"bundle": {**{k: meta[k] for k in ("model_version", "dataset", "split_version", "preprocessing_version", "code_revision")},
                         "permwater_input": meta["permwater_input"]},
              "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code_revision": contract.code_revision(),
              "selection_rule": "max val flood F2 over candidate bundles x decision variants x thresholds",
              "val_tiles": n_val, "val_events": n_ev, "candidates": candidates, "decision": d, "baseline_threshold_db": t_db, "splits": {}}
    for split in a.splits:
        samples, preds, base = [], [], []
        for s in load_iter(manifest, split, root, mean, std, meta["permwater_input"]):
            preds.append(decide(predict_probs(model, s["z"], dev), d["variant"], d["threshold"], s["permwater"]))
            base.append(threshold_pred(s, t_db, mean, std))
            s["z"] = s["z"][:2]  # keep radar only, for the figure
            samples.append(s)
        if not samples:
            continue
        never = [np.where(s["y"] == contract.FLOOD, contract.BACKGROUND, s["y"]).astype("uint8") for s in samples]
        res = {"chosen": evaluate(preds, samples), "baseline": evaluate(base, samples), "never_flood": evaluate(never, samples)}
        report["splits"][split] = {"description": descriptions.get(split, split), "methods": res}
        figure(samples, preds, out / f"examples_{split}.png", mean, std, f"{split}: {d['model_version']} {d['variant']} t={d['threshold']}")
        p = res["chosen"]["pooled"]
        print(f"{split}: {len(samples)} tiles  recall {fmt(p['flood_recall'], True)}  precision {fmt(p['flood_precision'], True)}  IoU {p['flood_iou']:.3f}  "
              f"accuracy {p['accuracy_flood_vs_not']:.1%}  (baseline recall {fmt(res['baseline']['pooled']['flood_recall'], True)}, IoU {res['baseline']['pooled']['flood_iou']:.3f})", flush=True)
        del samples, preds, base, never

    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    (out / "report.md").write_text(markdown(report))
    meta.pop("permwater_input", None)
    meta["evaluation_report"] = "evaluation/report.json"
    meta["chosen_variant"] = d["variant"]
    meta["decision_threshold_255"] = d["threshold"]
    meta["decision"] = f"{d['variant']}; flood score >= {d['threshold']}/255, chosen on val by F2 (recall-weighted)"
    (path / "bundle.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
