"""Build an event-split manifest for GEOID-Flood v1.1 and download only the tiles it selects.

PROVISIONAL SOURCE ADAPTER. Dataset selection is T2 (Lokesh, with Ariana). GEOID-Flood is the source
Ariana's flood notebook already uses, so this adapter lets the T5 baseline run now. If T2 picks a
different source, write a new adapter that emits the same manifest columns; nothing else changes.

Why not GEOID-Flood's own train/val/test: they are assigned per area of interest, so 50 of 210
flood events (e.g. EMSR712, whose AoI 10 is train and AoIs 1-9 are test) appear in more than one
split. Here every tile of an event, and of every event that shares a touching-AoI cluster with it,
lands in one split. The held-out tree (activations EMSR857-871, Jan-Mar 2026) is kept as a later-in-
time test; held-out tiles whose footprint overlaps any main-tree tile are dropped.

Run from the repo root:
    python -m flood.geoid                       # default budget, about 1.6 GB of radar
    python -m flood.geoid --train 120 --val 30 --test 60 --heldout 30   # smaller
Writes data/flood/geoid/{manifest.csv, manifest_meta.json, <event-aoi>/<layer>/*.tif}.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
import shapely
from huggingface_hub import hf_hub_download, hf_hub_url

from .contract import FLOOD, IGNORE, PERMANENT_WATER

REPO, REVISION = "links-ads/geoid-flood", "v1.1"
DATASET = f"GEOID-Flood {REVISION}"
LICENSE = "CC BY 4.0 (labels, splits); contains modified Copernicus Sentinel-1 data and CEMS Rapid Mapping products, (c) European Union"
OUT = Path("data/flood/geoid")
MASK_LAYERS = ("label", "validity", "permwater")
SPLIT_FRACTIONS = {"train": 0.7, "val": 0.1, "test": 0.2}
SEED = "skyalert-flood-v1"
MIN_VALID = 0.2  # tiles with less than this share of labelled pixels are skipped
FLOODED = 0.01  # a tile counts as flooded when flood pixels exceed this share of its valid pixels


def meta(name):
    return Path(hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=OUT / "_meta"))


def tile_key(event, aoi, bbox):
    return f"{event}-{aoi}-{bbox}"


# ---------------------------------------------------------------- catalog and per-tile statistics

def load_catalog():
    frames = []
    for tree in ("geoid-flood", "geoid-flood-heldout"):
        c = pd.read_parquet(meta(f"{tree}/tile_catalog.parquet"))
        c["tree"] = tree
        if "cluster_id" not in c:
            c["cluster_id"] = -1
        frames.append(c)
    c = pd.concat(frames, ignore_index=True)
    c["sample_id"] = [tile_key(e, a, b) for e, a, b in zip(c.event_id, c.tile_id, c.bbox_id)]
    c["aoi"] = [f"{e}-{a}" for e, a in zip(c.event_id, c.tile_id)]
    c["footprint"] = shapely.from_wkb(c.geometry.tolist())
    return c


def mask_stats(shards):
    """Valid, flood and permanent-water pixel counts for every tile, streamed from the label shards."""
    rows = []
    for s in shards:
        if "/label/" not in s["path"]:
            continue
        with tarfile.open(meta(s["path"])) as tar:
            for m in tar:
                if not m.isfile():
                    continue
                with rasterio.open(io.BytesIO(tar.extractfile(m).read())) as f:
                    y = f.read(1)
                rows.append({
                    "sample_id": os.path.basename(m.name).split("_label")[0],
                    "valid_frac": float((y != IGNORE).mean()),
                    "flood_px": int((y == FLOOD).sum()),
                    "permwater_px": int((y == PERMANENT_WATER).sum()),
                    "valid_px": int((y != IGNORE).sum()),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- event + geography split

def split_groups(c):
    """Connected components of (event, touching-AoI cluster): one group never straddles splits."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e, k, tree in zip(c.event_id, c.cluster_id, c.tree):
        if tree == "geoid-flood":
            parent[find("e" + e)] = find(f"c{k}")
        else:
            find("e" + e)
    return [find("e" + e) for e in c.event_id]


def assign_splits(c):
    """Groups in a seeded hash order, each going to the split furthest below its target tile share."""
    main = c[c.tree == "geoid-flood"]
    sizes = main.groupby("split_group").size()
    order = sorted(sizes.index, key=lambda g: hashlib.sha1(f"{SEED}:{g}".encode()).hexdigest())
    have = dict.fromkeys(SPLIT_FRACTIONS, 0)
    out = {}
    for g in order:
        s = min(SPLIT_FRACTIONS, key=lambda k: have[k] / SPLIT_FRACTIONS[k])
        out[g] = s
        have[s] += sizes[g]
    split = c.split_group.map(out)
    split[c.tree == "geoid-flood-heldout"] = "heldout"
    return split


def footprint_leaks(c, a, b):
    """Tiles of split b whose footprint intersects any tile of split a (same ground in both)."""
    A, B = c[c.my_split == a], c[c.my_split == b]
    tree = shapely.STRtree(A.footprint.values)
    hits = tree.query(B.footprint.values, predicate="intersects")
    return set(B.sample_id.values[np.unique(hits[0])])


# ---------------------------------------------------------------- tile selection

def round_robin(df, n, rng):
    """Up to n tiles, spread evenly across events."""
    by_event = defaultdict(list)
    for sid, ev in zip(df.sample_id, df.event_id):
        by_event[ev].append(sid)
    for v in by_event.values():
        rng.shuffle(v)
    out, k = [], 0
    while len(out) < n and any(len(v) > k for v in by_event.values()):
        out += [v[k] for v in sorted(by_event.values(), key=lambda v: v[0]) if len(v) > k]
        k += 1
    return out[:n]


def select(c, budgets, flooded_share):
    rng = np.random.default_rng(0)
    picked = []
    for split, n in budgets.items():
        pool = c[(c.my_split == split) & (c.valid_frac >= MIN_VALID) & ~c.excluded]
        wet = pool[pool.flood_px > FLOODED * pool.valid_px]
        dry = pool[pool.flood_px == 0]
        n_wet = int(round(n * flooded_share[split]))
        picked += round_robin(wet, n_wet, rng)
        picked += round_robin(dry, n - min(n_wet, len(wet)), rng)
    return set(picked)


# ---------------------------------------------------------------- download

def extract_masks(shards, wanted):
    for s in shards:
        if not any(f"/{layer}/" in s["path"] for layer in MASK_LAYERS):
            continue
        with tarfile.open(meta(s["path"])) as tar:
            for m in tar:
                if m.isfile() and os.path.basename(m.name).split("_")[0] in wanted:
                    dest = OUT / m.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(m) as src, open(dest, "wb") as f:
                        shutil.copyfileobj(src, f)


def _header(sess, url, off):
    r = sess.get(url, headers={"Range": f"bytes={off}-{off + 511}"}, timeout=60)
    r.raise_for_status()
    return None if r.content == b"\0" * 512 else tarfile.TarInfo.frombuf(r.content, "utf-8", "surrogateescape")


def fetch_members(path, wanted):
    """Walk an uncompressed tar shard's headers with range requests; download only `wanted` members."""
    sess = requests.Session()
    url = sess.head(hf_hub_url(REPO, path, repo_type="dataset", revision=REVISION), allow_redirects=True, timeout=60).url
    off, long_name, left = 0, None, set(wanted)
    while left:
        h = _header(sess, url, off)
        if h is None:
            break
        data, step = off + 512, 512 * ((h.size + 511) // 512)
        if h.type in (tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME):
            body = sess.get(url, headers={"Range": f"bytes={data}-{data + h.size - 1}"}, timeout=60).content
            long_name = body.split(b"path=")[1].split(b"\n")[0].decode() if h.type == tarfile.XHDTYPE else body.rstrip(b"\0").decode()
            off = data + step
            continue
        name, long_name = long_name or h.name, None
        if name in left:
            dest = OUT / name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                r = sess.get(url, headers={"Range": f"bytes={data}-{data + h.size - 1}"}, timeout=300)
                r.raise_for_status()
                if len(r.content) != h.size:
                    raise IOError(f"short read for {name}")
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.replace(dest)
            left.discard(name)
        off = data + step
    return path, len(wanted) - len(left)


# ---------------------------------------------------------------- manifest

def manifest_rows(c, picked, radar_files):
    rows = []
    for r in c[c.sample_id.isin(picked)].itertuples():
        radar = radar_files.get(r.sample_id)
        if radar is None:
            continue
        acquired = datetime.strptime(radar.split("_s1grd_post_")[1][:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        with rasterio.open(OUT / radar) as f:
            crs, transform, (h, w) = f.crs.to_string(), tuple(f.transform)[:6], f.shape
        folder = f"{r.aoi}"
        rows.append({
            "sample_id": r.sample_id, "hazard": "flood", "task": "detection", "dataset": DATASET,
            "event_id": r.event_id, "aoi_id": r.aoi, "scene_id": "|".join(r.s1_post_products),
            "split": r.my_split, "split_group": r.split_group, "source_split": r.split,
            "event_time_utc": r.event_time, "acquisition_time_utc": acquired.isoformat(),
            "availability_time_utc": "",  # not recorded by GEOID-Flood; needed only for forecasting
            "radar_path": radar,
            "label_path": f"{folder}/label/{r.sample_id}_label.tif",
            "validity_path": f"{folder}/validity/{r.sample_id}_validity.tif",
            "permwater_path": f"{folder}/permwater/{r.sample_id}_permwater.tif",
            "bands": "VV|VH", "units": "linear sigma0", "label_semantics": "0 bg|1 permanent water|2 flooded water|255 ignore",
            "crs": crs, "transform": json.dumps(transform), "height": h, "width": w,
            "valid_frac": round(r.valid_frac, 4), "flood_frac": round(r.flood_px / max(r.valid_px, 1), 4),
            "permwater_frac": round(r.permwater_px / max(r.valid_px, 1), 4), "countries": r.countries,
        })
    return pd.DataFrame(rows).sort_values(["split", "sample_id"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=240)
    ap.add_argument("--val", type=int, default=50)
    ap.add_argument("--test", type=int, default=110)
    ap.add_argument("--heldout", type=int, default=50)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    budgets = {"train": a.train, "val": a.val, "test": a.test, "heldout": a.heldout}
    # Training leans on flooded tiles so the rare class shows up; evaluation keeps many dry tiles
    # so dry-scene false alarms can be measured. Pixel metrics are therefore on an enriched sample.
    flooded_share = {"train": 0.75, "val": 0.6, "test": 0.6, "heldout": 0.6}

    OUT.mkdir(parents=True, exist_ok=True)
    shards = json.load(gzip.open(meta("shard_index.json.gz")))["shards"]
    c = load_catalog()
    stats_path = OUT / "_meta" / "tile_stats.csv"
    if stats_path.exists():
        stats = pd.read_csv(stats_path)
    else:
        print("counting label pixels in every tile ...", flush=True)
        stats = mask_stats(shards)
        stats.to_csv(stats_path, index=False)
    c = c.merge(stats, on="sample_id", how="inner")
    c["split_group"] = split_groups(c)
    c["my_split"] = assign_splits(c)
    c["excluded"] = False
    overlap = footprint_leaks(c, "train", "heldout") | footprint_leaks(c, "val", "heldout") | footprint_leaks(c, "test", "heldout")
    c.loc[c.sample_id.isin(overlap), "excluded"] = True
    leaks = {f"{a_}->{b}": len(footprint_leaks(c, a_, b)) for a_, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    print("tiles per split:", c.my_split.value_counts().to_dict(), " footprint overlaps across splits:", leaks,
          f" held-out tiles dropped for overlapping main-tree ground: {len(overlap)}", flush=True)
    if any(leaks.values()):
        raise SystemExit(f"split leaks ground between splits: {leaks}")

    picked = select(c, budgets, flooded_share)
    print(f"selected {len(picked)} tiles", flush=True)
    extract_masks(shards, picked)

    want = defaultdict(set)
    for s in shards:
        if "/s1grd/" in s["path"]:
            for m in s["members"]:
                base = os.path.basename(m)
                if "_s1grd_post_" in base and base.split("_s1grd_")[0] in picked:
                    want[s["path"]].add(m)
    print(f"downloading {sum(map(len, want.values()))} post-event radar files from {len(want)} shards ...", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for path, got in ex.map(lambda kv: fetch_members(*kv), want.items()):
            print(f"  {os.path.basename(path)}: {got}", flush=True)
    radar_files = {os.path.basename(m).split("_s1grd_")[0]: m for ms in want.values() for m in ms if (OUT / m).exists()}

    manifest = manifest_rows(c, picked, radar_files)
    manifest.to_csv(OUT / "manifest.csv", index=False)
    group_map = c[["split_group", "my_split"]].drop_duplicates().sort_values("split_group")
    split_version = hashlib.sha1(group_map.to_csv(index=False).encode()).hexdigest()[:12]
    info = {
        "dataset": DATASET, "repo": REPO, "revision": REVISION, "license": LICENSE,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_method": "connected components of event_id and GEOID touching-AoI cluster_id, seeded hash order, "
                        "70/10/20 by tile count; GEOID-Flood held-out tree (EMSR857-871, 2026) as 'heldout'",
        "split_seed": SEED, "split_version": split_version,
        "split_tiles_available": c.my_split.value_counts().to_dict(),
        "split_events_available": c.groupby("my_split").event_id.nunique().to_dict(),
        "footprint_overlaps_across_splits": leaks, "heldout_tiles_dropped_for_overlap": len(overlap),
        "selection": {"budgets": budgets, "flooded_share": flooded_share, "min_valid_frac": MIN_VALID, "flooded_if_flood_frac_over": FLOODED},
        "selected_tiles": manifest.split.value_counts().to_dict(),
        "selected_events": manifest.groupby("split").event_id.nunique().to_dict(),
    }
    (OUT / "manifest_meta.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
