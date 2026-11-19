"""Run a flood bundle on one post-event Sentinel-1 GRD GeoTIFF and store a georeferenced result.

    python -m flood.predict --radar path/to/..._s1grd_post_YYYYMMDDTHHMMSS.tif [--validity v.tif] [--permwater p.tif]
    python -m flood.predict --sample-id EMSR123-4-5          # a tile from the manifest

Writes to data/flood/results/<prediction_id>/:
  classes.tif   uint8 on the input grid: 0 background, 1 permanent water, 2 flooded water, 255 no data
  flood.geojson flooded-water polygons in EPSG:4326
  result.json   the shared result record (docs/architecture: prediction_id, hazard, task, model_version, ...)
Inputs that break the contract (band order, units, unprojected grid, misaligned masks) are rejected.
This is a batch job; the API should read result.json, never run inference inside a request.
"""
import argparse
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import shapely.geometry
from rasterio.features import shapes
from rasterio.warp import transform_bounds, transform_geom

from . import bundle as bundles
from . import contract
from .data import ContractError, _open, check_grid, model_input, read_manifest, read_radar
from .metrics import pixel_km2
from .model import decide, predict_classes, predict_probs

ACQUIRED = re.compile(r"_post_(\d{8}T\d{6})")
MIN_POLYGON_PX = 25  # drop specks under 0.25 ha at 10 m from the vector layer (the raster keeps them)


def acquisition_time(path, tags):
    m = ACQUIRED.search(os.path.basename(path))
    if m:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).isoformat()
    for k in ("ACQUISITION_TIME", "TIFFTAG_DATETIME"):
        if k in tags:
            return tags[k]
    return None


def run(radar, out_root, bundle_dir=None, validity=None, permwater=None, source_ids=None):
    dev = bundles.device()
    model, meta, mean, std = bundles.load(bundle_dir or bundles.latest(), dev)
    use_permwater = meta.get("chosen_variant") == "model_permwater"
    if (use_permwater or meta["permwater_input"]) and permwater is None:
        raise ContractError("this bundle needs the permanent-water reference on the image grid; pass --permwater")

    valid_mask = pw = None
    with _open(radar) as grid:
        if validity:
            with _open(validity) as f:
                check_grid(grid, f, "validity"); valid_mask = f.read(1) == 1
        if permwater:
            with _open(permwater) as f:
                check_grid(grid, f, "permwater"); pw = f.read(1) == 1
    x, profile, tags = read_radar(radar, valid_mask)
    valid = np.isfinite(x).all(0)
    acquired = acquisition_time(radar, tags)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    inputs = list(source_ids or []) + [os.path.basename(radar)]
    try:
        inputs += json.loads(tags.get("SOURCE_PRODUCTS", "[]"))
    except json.JSONDecodeError:
        pass
    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{meta['model_version']}|{meta['preprocessing_version']}|{'|'.join(inputs)}"))
    out = Path(out_root) / pid
    out.mkdir(parents=True, exist_ok=True)

    record = {
        "prediction_id": pid, "hazard": contract.HAZARD, "task": contract.TASK,
        "product": "observed flood extent at acquisition time (not a forecast)",
        "model_version": meta["model_version"], "preprocessing_version": meta["preprocessing_version"],
        "input_ids": inputs, "acquisition_time_utc": acquired, "generated_at_utc": generated,
        "valid_coverage": round(float(valid.mean()), 4),
        "output_units": {"raster": "class codes 0 background, 1 permanent water, 2 flooded water, 255 no data", "area": "km2"},
        "scores_are_probabilities": False,
        "decision": meta.get("decision", "argmax"),
        "label_semantics": contract.LABEL_SEMANTICS,
    }
    if not valid.any():
        record.update(availability_status="unavailable", reason="no valid radar pixels", geometry=None, flood_area_km2=None)
        (out / "result.json").write_text(json.dumps(record, indent=2))
        return record

    z = model_input(x, mean, std, pw if meta["permwater_input"] else None)
    if "decision_threshold_255" in meta:  # threshold and variant chosen on val by flood.evaluate
        pred = decide(predict_probs(model, z, dev), meta["chosen_variant"], meta["decision_threshold_255"], pw)
    else:
        pred = predict_classes(model, z, dev)
        if use_permwater:
            pred = np.where(pred > 0, np.where(pw, contract.PERMANENT_WATER, contract.FLOOD), contract.BACKGROUND).astype("uint8")
    pred[~valid] = contract.IGNORE

    raster_profile = {k: profile[k] for k in ("crs", "transform", "width", "height")}
    with rasterio.open(out / "classes.tif", "w", driver="GTiff", count=1, dtype="uint8", nodata=contract.IGNORE, compress="deflate", **raster_profile) as dst:
        dst.write(pred, 1)
        dst.update_tags(prediction_id=pid, model_version=meta["model_version"], classes="0 background|1 permanent water|2 flooded water")

    flood = pred == contract.FLOOD
    px_size = abs(profile["transform"].a)
    features = []
    for geom, _ in shapes(flood.astype("uint8"), mask=flood, transform=profile["transform"]):
        n_px = abs(_ring_area(geom["coordinates"][0])) / abs(profile["transform"].a * profile["transform"].e)
        if n_px >= MIN_POLYGON_PX:
            # one-pixel simplification in the native metre grid, so the vector layer stays small for the map
            simple = shapely.geometry.mapping(shapely.geometry.shape(geom).simplify(px_size, preserve_topology=True))
            features.append({"type": "Feature", "properties": {"class": "flooded water"},
                             "geometry": transform_geom(profile["crs"], "EPSG:4326", simple, precision=6)})
    (out / "flood.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    km2 = pixel_km2(profile)
    w, s, e, n = transform_bounds(profile["crs"], "EPSG:4326", *rasterio.transform.array_bounds(profile["height"], profile["width"], profile["transform"]))
    record.update(
        availability_status="available",
        geometry={"raster": "classes.tif", "raster_crs": profile["crs"].to_string(), "vector": "flood.geojson", "vector_crs": "EPSG:4326",
                  "bounds_wgs84": [round(v, 6) for v in (w, s, e, n)], "flood_polygons": len(features),
                  "min_polygon_px": MIN_POLYGON_PX, "vector_simplify_m": px_size},
        flood_area_km2=round(float(flood.sum() * km2), 3) if km2 else None,
        permanent_water_area_km2=round(float((pred == contract.PERMANENT_WATER).sum() * km2), 3) if km2 else None,
    )
    (out / "result.json").write_text(json.dumps(record, indent=2))
    return record


def _ring_area(ring):
    xs, ys = np.array([p[0] for p in ring]), np.array([p[1] for p in ring])
    return 0.5 * float(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radar")
    ap.add_argument("--validity")
    ap.add_argument("--permwater")
    ap.add_argument("--sample-id", help="take the radar and masks of this manifest row")
    ap.add_argument("--manifest", default="data/flood/geoid/manifest.csv")
    ap.add_argument("--bundle")
    ap.add_argument("--out", default="data/flood/results")
    a = ap.parse_args()
    source_ids = []
    if a.sample_id:
        root = Path(a.manifest).parent
        rows = read_manifest(a.manifest)
        row = rows[rows.sample_id == a.sample_id]
        if row.empty:
            raise SystemExit(f"{a.sample_id} not in {a.manifest}")
        row = row.iloc[0]
        a.radar, a.validity, a.permwater = (str(root / row[k]) for k in ("radar_path", "validity_path", "permwater_path"))
        source_ids = [f"{row['dataset']}:{row['sample_id']}"]
    if not a.radar:
        raise SystemExit("pass --radar or --sample-id")
    rec = run(a.radar, a.out, a.bundle, a.validity, a.permwater, source_ids)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
