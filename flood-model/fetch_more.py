"""Download a few hundred GEOID-Flood training tiles without downloading the 2 GB shards they live in.

The full dataset is stored as .tar shards on Hugging Face. A tar file is a header + file, header + file, ...
so we read the headers with small HTTP range requests and download only the after-flood radar of the
tiles we pick. Labels and validity masks come from their own small shards (about 120 MB together).

Run from the flood-model folder:
    python fetch_more.py              # about 300 tiles, about 1.1 GB
    python fetch_more.py --tiles 150  # fewer, if disk is tight
    python fetch_more.py --heldout --tiles 50  # random tiles from the held-out 2025 floods, for testing only
    python fetch_more.py --split val --tiles 80  # random validation tiles, for picking the best training round
    python fetch_more.py --pre [--split ...]   # add the before-flood radar for tiles already downloaded
    python fetch_more.py --dem [--split ...]   # add the ground-height map for tiles already downloaded
Files land in data/full/<area>/{s1grd,label,validity}/, the same layout as the sample.
"""
import argparse
import gzip
import io
import json
import os
import random
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
import requests
from huggingface_hub import hf_hub_download, hf_hub_url

REPO, REV = "links-ads/geoid-flood", "v1.1"
META, OUT = "data/full_meta", "data/full"
TILE_PX = 1024 * 1024


def get_meta(name):
    return hf_hub_download(REPO, name, repo_type="dataset", revision=REV, local_dir=META)


def pick_tiles(label_tar, n, seed=0):
    """80% tiles that are over 2% flooded, spread across as many floods as possible; 20% any other tile."""
    counts = []
    with tarfile.open(label_tar) as tar:
        for m in tar:
            if m.isfile():
                with rasterio.open(io.BytesIO(tar.extractfile(m).read())) as f:
                    flood = int((f.read(1) == 2).sum())
                counts.append((os.path.basename(m.name).split("_label")[0], flood))
    rng = random.Random(seed)
    by_event = defaultdict(list)
    for tid, flood in counts:
        if flood > 0.02 * TILE_PX:
            by_event[tid.split("-")[0]].append(tid)  # EMSR151-1-10 -> EMSR151, one flood activation
    for v in by_event.values():
        rng.shuffle(v)
    flooded, k = [], 0
    while len(flooded) < int(n * 0.8) and any(len(v) > k for v in by_event.values()):
        flooded += [v[k] for v in by_event.values() if len(v) > k]  # round robin: one tile per flood, then the next
        k += 1
    flooded = flooded[:int(n * 0.8)]
    rest = [t for t, _ in counts if t not in set(flooded)]
    return sorted(flooded + rng.sample(rest, n - len(flooded))), len(by_event)


def pick_random(label_tar, n, seed=0):
    """Random tiles that have some labelled pixels, spread evenly across floods. No preference for flooded ones."""
    by_event = defaultdict(list)
    with tarfile.open(label_tar) as tar:
        for m in tar:
            if m.isfile():
                with rasterio.open(io.BytesIO(tar.extractfile(m).read())) as f:
                    if (f.read(1) != 255).mean() > 0.1:  # skip tiles that are almost all 'ignore'
                        tid = os.path.basename(m.name).split("_label")[0]
                        by_event[tid.split("-")[0]].append(tid)
    rng = random.Random(seed)
    for v in by_event.values():
        rng.shuffle(v)
    out, k = [], 0
    while len(out) < n and any(len(v) > k for v in by_event.values()):
        out += [v[k] for v in by_event.values() if len(v) > k]
        k += 1
    return sorted(out[:n]), len(by_event)


def read_header(sess, url, off):
    r = sess.get(url, headers={"Range": f"bytes={off}-{off + 511}"}, timeout=60)
    r.raise_for_status()
    if r.content == b"\0" * 512:
        return None  # end of archive
    return tarfile.TarInfo.frombuf(r.content, "utf-8", "surrogateescape")


def fetch_from_shard(path, wanted):
    """Walk one shard's headers with range requests, downloading only the members in `wanted`."""
    sess = requests.Session()
    url = sess.head(hf_hub_url(REPO, path, repo_type="dataset", revision=REV), allow_redirects=True, timeout=60).url
    off, long_name, got = 0, None, 0
    while got < len(wanted):
        h = read_header(sess, url, off)
        if h is None:
            break
        data_off, step = off + 512, 512 * ((h.size + 511) // 512)
        if h.type in (tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME):  # the real name is in this block's data
            body = sess.get(url, headers={"Range": f"bytes={data_off}-{data_off + h.size - 1}"}, timeout=60).content
            long_name = body.split(b"path=")[1].split(b"\n")[0].decode() if h.type == tarfile.XHDTYPE else body.rstrip(b"\0").decode()
            off = data_off + step
            continue
        name, long_name = long_name or h.name, None
        if name in wanted:
            dest = os.path.join(OUT, name)
            if not os.path.exists(dest):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                r = sess.get(url, headers={"Range": f"bytes={data_off}-{data_off + h.size - 1}"}, timeout=300)
                r.raise_for_status()
                assert len(r.content) == h.size, f"short read for {name}"
                with open(dest + ".part", "wb") as f:
                    f.write(r.content)
                os.replace(dest + ".part", dest)  # no half-written files if stopped
            got += 1
        off = data_off + step
    return path, got


def extract(tar_path, members):
    with tarfile.open(tar_path) as tar:
        for m in tar:
            if m.name in members:
                dest = os.path.join(OUT, m.name)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with tar.extractfile(m) as src, open(dest, "wb") as f:
                    f.write(src.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=300)
    ap.add_argument("--split", choices=["train", "val", "heldout"], default="train",
                    help="train -> data/full, val (for picking the best training round) -> data/val, heldout (2025 floods, final test) -> data/heldout")
    ap.add_argument("--heldout", action="store_true", help="same as --split heldout")
    ap.add_argument("--pre", action="store_true", help="add the before-flood radar for the tiles already in that split's folder")
    ap.add_argument("--dem", action="store_true", help="add the ground-height map (DEM) for the tiles already in that split's folder")
    a = ap.parse_args()
    split = "heldout" if a.heldout else a.split
    global OUT
    OUT = {"train": "data/full", "val": "data/val", "heldout": "data/heldout"}[split]
    prefix = "geoid-flood-heldout/shards/test/" if split == "heldout" else f"geoid-flood/shards/{split}/"

    index = json.load(gzip.open(get_meta("shard_index.json.gz")))["shards"]
    shards = [s for s in index if s["path"].startswith(prefix)]
    label_tar = get_meta(next(s["path"] for s in shards if "/label/" in s["path"]))
    valid_tar = get_meta(next(s["path"] for s in shards if "/validity/" in s["path"]))
    extra = [] if split == "train" else [get_meta(next(s["path"] for s in shards if "/permwater/" in s["path"]))]  # map of normal water, for scoring

    layer = "dem" if a.dem else "s1grd"
    if a.pre or a.dem:
        tiles = sorted({os.path.basename(p).split("_s1grd_")[0] for p in _walk("s1grd")})  # no new picks: the same tiles, plus one more layer
        kind = "_dem" if a.dem else "_s1grd_pre_"
        print(f"{'ground height' if a.dem else 'before-flood radar'} for the {len(tiles)} tiles in {OUT}/", flush=True)
    else:
        tiles, n_events = pick_tiles(label_tar, a.tiles) if split == "train" else pick_random(label_tar, a.tiles)
        kind = "_s1grd_post_"
        print(f"picked {len(tiles)} {split} tiles ({n_events} floods to pick from)", flush=True)
    tile_set = set(tiles)

    want = defaultdict(set)  # shard path -> files to pull from it
    for s in shards:
        if f"/{layer}/" in s["path"]:
            for m in s["members"]:
                base = os.path.basename(m)
                if kind in base and base.split(f"_{layer}")[0] in tile_set:
                    want[s["path"]].add(m)
    print(f"{sum(map(len, want.values()))} {layer} files in {len(want)} shards", flush=True)

    with ThreadPoolExecutor(14) as ex:
        for path, got in ex.map(lambda kv: fetch_from_shard(*kv), want.items()):
            print(f"  {os.path.basename(path)}: {got} files", flush=True)

    have = {os.path.basename(p).split("_s1grd_")[0] for p in _walk("s1grd")}
    for tar_path in [label_tar, valid_tar] + extra:
        with tarfile.open(tar_path) as tar:
            names = {m.name for m in tar if os.path.basename(m.name).split("_")[0] in have}
        extract(tar_path, names)
    print(f"done: {len(have)} tiles in {OUT}/")


def _walk(layer):
    for root, _, files in os.walk(OUT):
        if root.endswith(layer):
            yield from (os.path.join(root, f) for f in files if f.endswith(".tif"))


if __name__ == "__main__":
    main()
