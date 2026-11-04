"""Download the Landsat-8 Active Fire sample and print image-mask pairs.

The full archive is about 192 GB. The authors also publish samples.zip,
about 231 MB: 100 patches of 256 by 256 pixels, ten bands each, with
Schroeder, Murphy, and Kumar-Roy fire masks. This script fetches that
sample only.
"""

import io
import json
import math
import re
import subprocess
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

FILE_ID = "1gwQdhXrxCybcO16vem09DfW5fPadAA_p"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landsat8"
SAMPLE_COUNT = 5

# Channel order from Pereira et al. 2021. The TIFF files do not store names.
# Landsat-8 band 8, the panchromatic band, is left out. All ten stay in the array.
BANDS = (
    (1, "Coastal aerosol"),
    (2, "Blue"),
    (3, "Green"),
    (4, "Red"),
    (5, "Near-infrared"),
    (6, "Shortwave infrared 1"),
    (7, "Shortwave infrared 2"),
    (9, "Cirrus"),
    (10, "Thermal infrared 1"),
    (11, "Thermal infrared 2"),
)

# Model input, the same three bands as the paper's 3-channel U-Nets.
# File bands 7, 6, and 2 are Landsat bands 7, 6, and 2, because band 8 is
# absent and the first seven file positions match Landsat bands 1 through 7.
# The authors' U-Net (3c) divides those values by 65535.
DISPLAY_BANDS = (7, 6, 2)
MAX_PIXEL_VALUE = 65535
BATCH_SIZE = 4

MASK_SOURCES = {
    "schroeder": ("samples/masks/patches_schroeder.zip", "Schroeder"),
    "murphy": ("samples/masks/patches_murphy.zip", "Murphy"),
    "kumar-roy": ("samples/masks/patches_goli.zip", "Kumar-Roy"),
}


def download_samples():
    """Save samples.zip from the authors' Google Drive folder."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / "samples.zip"
    if destination.exists() and zipfile.is_zipfile(destination):
        print(f"Using existing {destination}")
        return destination

    cookie = DATA_DIR / "gdrive.cookies"
    page = subprocess.check_output(
        [
            "curl",
            "-fsSL",
            "-A",
            "Mozilla/5.0",
            "-c",
            str(cookie),
            f"https://drive.google.com/uc?export=download&id={FILE_ID}",
        ]
    )
    text = page.decode("utf-8", errors="replace")
    if "Too many users have viewed or downloaded this file recently" in text:
        raise RuntimeError(
            "Google Drive refused the sample download because too many people "
            "fetched it recently. Wait and run this again."
        )
    confirm = re.search(r'name="confirm" value="([^"]+)"', text)
    uuid = re.search(r'name="uuid" value="([^"]+)"', text)
    if confirm is None or uuid is None:
        raise RuntimeError("Google Drive did not return a download confirmation page")

    url = (
        "https://drive.usercontent.google.com/download"
        f"?id={FILE_ID}&export=download&confirm={confirm.group(1)}&uuid={uuid.group(1)}"
    )
    print("Downloading samples.zip (about 231 MB)")
    subprocess.check_call(
        ["curl", "-fL", "-A", "Mozilla/5.0", "-b", str(cookie), "-o", str(destination), url]
    )
    cookie.unlink(missing_ok=True)
    if not zipfile.is_zipfile(destination):
        destination.unlink(missing_ok=True)
        raise RuntimeError("Download was not a zip file")
    print(f"Saved {destination}")
    return destination


def _patch_key(filename):
    stem = Path(filename).stem
    for token in ("_Schroeder", "_Murphy", "_Kumar-Roy", "_GOLI_v2"):
        stem = stem.replace(token, "")
    return stem


def _unpack(outer, member, dest_dir, algorithm_token):
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = list(dest_dir.glob("*.tif"))
    if len(existing) >= 100:
        return
    payload = outer.read(member)
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        for info in inner.infolist():
            if not info.filename.endswith(".tif"):
                continue
            name = Path(info.filename).name.replace("GOLI_v2", algorithm_token)
            target = dest_dir / name
            if not target.exists():
                target.write_bytes(inner.read(info))


def extract_samples(zip_path):
    """Unpack the 100 image patches and the three algorithm masks."""
    images = DATA_DIR / "images"
    with zipfile.ZipFile(zip_path) as outer:
        _unpack(outer, "samples/images/patches.zip", images, "")
        for folder, (member, token) in MASK_SOURCES.items():
            _unpack(outer, member, DATA_DIR / "masks" / folder, token)
    return images


def _read_tif(path):
    return tifffile.imread(path)


def band_index(landsat_band):
    for index, (number, _name) in enumerate(BANDS):
        if number == landsat_band:
            return index
    raise KeyError(landsat_band)


def load_pairs(limit=SAMPLE_COUNT):
    """Return the patches with the most Schroeder fire pixels.

    Every patch in the sample has some fire, but most of it is a few pixels.
    The busiest masks are the ones worth looking at.
    """
    images = DATA_DIR / "images"
    mask_dirs = {name: DATA_DIR / "masks" / name for name in MASK_SOURCES}
    by_key = {}
    for folder, directory in mask_dirs.items():
        for path in directory.glob("*.tif"):
            key = _patch_key(path.name)
            by_key.setdefault(key, {})[folder] = path

    ranked = []
    for key, masks in by_key.items():
        schroeder = _read_tif(masks["schroeder"])
        ranked.append((int((schroeder == 1).sum()), key, masks, schroeder))
    ranked.sort(key=lambda item: item[0], reverse=True)

    pairs = []
    for fire_pixels, key, masks, schroeder in ranked[:limit]:
        image_path = images / f"{key}.tif"
        image = _read_tif(image_path)
        counts = {"schroeder": fire_pixels}
        for folder in ("murphy", "kumar-roy"):
            mask = _read_tif(masks[folder])
            counts[folder] = int((mask == 1).sum())
        pairs.append(
            {
                "patch": key,
                "image": image,
                "mask": schroeder,
                "fire_pixels": counts,
                "image_path": image_path,
                "mask_path": masks["schroeder"],
                "mask_paths": masks,
            }
        )
    return pairs


def _relative(path):
    return str(path.relative_to(DATA_DIR.parents[1]))


def iter_image_masks():
    """Yield each image and the Schroeder mask the pairing rule names."""
    images = sorted((DATA_DIR / "images").glob("*.tif"))
    mask_dir = DATA_DIR / "masks" / "schroeder"
    for image_path in images:
        mask_path = mask_dir / expected_mask_name(image_path.name, "Schroeder")
        yield image_path, mask_path


def classifier_label(mask):
    """Fire when any valid mask pixel is 1. No-fire when every valid pixel is 0.

    Valid pixels are 0 or 1. Any other value is ignored. A mask with no valid
    pixels has no classifier label.
    """
    valid = np.isin(mask, (0, 1))
    if not valid.any():
        return "invalid"
    if np.any(mask[valid] == 1):
        return "fire"
    return "no-fire"


def prepare_samples():
    """Pair every sample, check it, and derive the classifier label.

    The segmentation target is the mask itself. The classifier label is only
    a summary of that mask.
    """
    pair_rows = []
    check_rows = []
    for image_path, mask_path in iter_image_masks():
        parsed = parse_patch_name(image_path.name)
        mask_exists = mask_path.exists()
        pair_rows.append(
            {
                "image_path": _relative(image_path),
                "mask_path": _relative(mask_path) if mask_exists else "",
                "scene_id": parsed["scene"],
            }
        )
        image = _read_tif(image_path)
        if mask_exists:
            mask = _read_tif(mask_path)
        else:
            mask = None
        band_count = int(image.shape[2]) if image.ndim == 3 else 1
        missing = int(np.all(image == 0, axis=-1).sum()) if image.ndim == 3 else int((image == 0).sum())
        if mask is None:
            mask_values = ""
            invalid = None
            label = "missing-mask"
            dimensions_match = False
            mask_shape = ""
        else:
            values = np.unique(mask)
            mask_values = ",".join(str(int(value)) for value in values)
            invalid = int((~np.isin(mask, (0, 1))).sum())
            label = classifier_label(mask)
            dimensions_match = image.shape[:2] == mask.shape[:2]
            mask_shape = f"{mask.shape[0]}x{mask.shape[1]}"
        check_rows.append(
            {
                "scene_id": parsed["scene"],
                "patch_id": parsed["patch"],
                "image_shape": f"{image.shape[0]}x{image.shape[1]}x{band_count}",
                "mask_shape": mask_shape,
                "dimensions_match": dimensions_match,
                "band_count": band_count,
                "band_count_ok": band_count == len(BANDS),
                "image_dtype": str(image.dtype),
                "mask_values": mask_values,
                "invalid_mask_pixels": invalid,
                "missing_image_pixels": missing,
                "classifier_label": label,
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(check_rows)


def make_batch(size=BATCH_SIZE):
    """Scale bands 7, 6, and 2 for the first `size` patches in filename order.

    Image values are float32 divided by 65535. The mask stays 0 or 1.
    """
    indexes = [band_index(number) for number in DISPLAY_BANDS]
    images = []
    masks = []
    labels = []
    scene_ids = []
    patch_ids = []
    for image_path, mask_path in list(iter_image_masks())[:size]:
        parsed = parse_patch_name(image_path.name)
        image = _read_tif(image_path)
        mask = _read_tif(mask_path).astype(np.float32)
        selected = image[:, :, indexes].astype(np.float32) / MAX_PIXEL_VALUE
        if mask.ndim == 2:
            mask = mask[:, :, None]
        images.append(selected)
        masks.append(mask)
        labels.append(classifier_label(mask))
        scene_ids.append(parsed["scene"])
        patch_ids.append(parsed["patch"])
    return np.stack(images), np.stack(masks), labels, scene_ids, patch_ids


def parse_patch_name(filename):
    """Split a dataset filename into the Landsat scene id and the patch id.

    An image is ``{scene}_p#####.tif``. A mask is ``{scene}_{algorithm}_p#####.tif``.
    That is the pairing rule used by the files in the authors' sample.
    """
    stem = Path(filename).stem
    algorithm = ""
    for token in ("Schroeder", "Murphy", "Kumar-Roy", "GOLI_v2"):
        marker = f"_{token}_"
        if marker in stem:
            algorithm = token
            stem = stem.replace(marker, "_")
            break
    scene, patch = stem.rsplit("_", 1)
    parts = scene.split("_")
    path_row = parts[2]
    return {
        "scene": scene,
        "patch": patch,
        "algorithm": algorithm,
        "wrs_path": int(path_row[:3]),
        "wrs_row": int(path_row[3:]),
        "acquired": parts[3],
    }


def expected_mask_name(image_name, algorithm):
    parsed = parse_patch_name(image_name)
    return f"{parsed['scene']}_{algorithm}_{parsed['patch']}.tif"


def _geo_tags(path):
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        crs = str(page.tags["GeoAsciiParamsTag"].value).strip("|").replace("|", " | ")
        return {
            "width": int(page.imagewidth),
            "height": int(page.imagelength),
            "scale": tuple(float(value) for value in page.tags["ModelPixelScaleTag"].value),
            "tie": tuple(float(value) for value in page.tags["ModelTiepointTag"].value),
            "crs": crs,
        }


def _ground_xy(tags, row, col):
    """Map a pixel to the ground coordinate stored in the GeoTIFF.

    The tie point is the upper-left corner of pixel (0, 0). Row numbers
    increase south, so northing decreases as the row increases.
    """
    easting = tags["tie"][3] + (col + 0.5) * tags["scale"][0]
    northing = tags["tie"][4] - (row + 0.5) * tags["scale"][1]
    return easting, northing


def _utm_to_latlon(easting, northing, zone, northern):
    """Convert a WGS 84 UTM coordinate to latitude and longitude in degrees."""
    equatorial_radius = 6378137.0
    flattening = 1 / 298.257223563
    eccentricity2 = flattening * (2 - flattening)
    second_eccentricity2 = eccentricity2 / (1 - eccentricity2)
    scale = 0.9996
    x = easting - 500000.0
    y = northing if northern else northing - 10000000.0
    meridian = y / scale
    mu = meridian / (
        equatorial_radius
        * (1 - eccentricity2 / 4 - 3 * eccentricity2**2 / 64 - 5 * eccentricity2**3 / 256)
    )
    e1 = (1 - math.sqrt(1 - eccentricity2)) / (1 + math.sqrt(1 - eccentricity2))
    foot = (
        mu
        + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
        + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
        + (151 * e1**3 / 96) * math.sin(6 * mu)
        + (1097 * e1**4 / 512) * math.sin(8 * mu)
    )
    sine = math.sin(foot)
    cosine = math.cos(foot)
    n1 = equatorial_radius / math.sqrt(1 - eccentricity2 * sine**2)
    t1 = math.tan(foot) ** 2
    c1 = second_eccentricity2 * cosine**2
    r1 = equatorial_radius * (1 - eccentricity2) / (1 - eccentricity2 * sine**2) ** 1.5
    d = x / (n1 * scale)
    latitude = foot - (n1 * math.tan(foot) / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * second_eccentricity2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * second_eccentricity2 - 3 * c1**2) * d**6 / 720
    )
    longitude = (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * second_eccentricity2 + 24 * t1**2) * d**5 / 120
    ) / cosine
    return math.degrees(latitude), math.degrees(longitude) + (zone * 6 - 183)


def _zone(crs):
    match = re.search(r"UTM zone (\d+)([NS])", crs)
    if match is None:
        raise ValueError(f"No UTM zone in {crs}")
    return int(match.group(1)), match.group(2) == "N"


def _fire_pixel(mask):
    rows, cols = np.where(mask == 1)
    if len(rows) == 0:
        return None
    row = int(round(float(rows.mean())))
    col = int(round(float(cols.mean())))
    nearest = np.argmin((rows - row) ** 2 + (cols - col) ** 2)
    return int(rows[nearest]), int(cols[nearest])


def _usgs_scene(wrs_path, wrs_row, acquired):
    """Find the Landsat-8 overpass USGS now lists for this path, row, and day."""
    day = datetime.strptime(acquired, "%Y%m%d")
    end = day + timedelta(days=1)
    query = {
        "collections": ["landsat-c2l1"],
        "datetime": f"{day:%Y-%m-%d}T00:00:00Z/{end:%Y-%m-%d}T00:00:00Z",
        "limit": 5,
        "query": {
            "landsat:wrs_path": {"eq": f"{wrs_path:03d}"},
            "landsat:wrs_row": {"eq": f"{wrs_row:03d}"},
            "platform": {"eq": "LANDSAT_8"},
        },
    }
    completed = subprocess.run(
        [
            "curl",
            "-fsSL",
            "-A",
            "Mozilla/5.0",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(query),
            "https://landsatlook.usgs.gov/stac-server/search",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout:
        return None
    features = json.loads(completed.stdout).get("features", [])
    if not features:
        return None
    feature = features[0]
    return {
        "usgs_scene": feature["id"],
        "acquired_utc": feature["properties"].get("datetime", ""),
        "bbox": feature.get("bbox"),
    }


def pairing_table(pairs):
    """Show that each mask filename is the image filename plus the algorithm name."""
    rows = []
    for pair in pairs:
        image_name = pair["image_path"].name
        mask_name = pair["mask_path"].name
        image_id = parse_patch_name(image_name)
        mask_id = parse_patch_name(mask_name)
        rows.append(
            {
                "image_file": image_name,
                "mask_file": mask_name,
                "scene_id": image_id["scene"],
                "patch_id": image_id["patch"],
                "scene_ids_match": image_id["scene"] == mask_id["scene"],
                "patch_ids_match": image_id["patch"] == mask_id["patch"],
                "mask_name_follows_rule": mask_name == expected_mask_name(image_name, "Schroeder"),
                "murphy_name_follows_rule": pair["mask_paths"]["murphy"].name
                == expected_mask_name(image_name, "Murphy"),
                "kumar_roy_name_follows_rule": pair["mask_paths"]["kumar-roy"].name
                == expected_mask_name(image_name, "Kumar-Roy"),
            }
        )
    return pd.DataFrame(rows)


def alignment_table(pairs):
    """Show that one pixel index is the same place on the ground in both files."""
    rows = []
    for pair in pairs:
        image_geo = _geo_tags(pair["image_path"])
        mask_geo = _geo_tags(pair["mask_path"])
        fire = _fire_pixel(pair["mask"])
        image_xy = _ground_xy(image_geo, *fire)
        mask_xy = _ground_xy(mask_geo, *fire)
        rows.append(
            {
                "patch_id": parse_patch_name(pair["image_path"].name)["patch"],
                "same_width_height": (
                    image_geo["width"] == mask_geo["width"] == pair["mask"].shape[1]
                    and image_geo["height"] == mask_geo["height"] == pair["mask"].shape[0]
                ),
                "same_pixel_size_m": image_geo["scale"] == mask_geo["scale"],
                "same_upper_left": image_geo["tie"] == mask_geo["tie"],
                "same_crs": image_geo["crs"] == mask_geo["crs"],
                "crs": image_geo["crs"],
                "fire_row": fire[0],
                "fire_col": fire[1],
                "image_easting": round(image_xy[0], 2),
                "image_northing": round(image_xy[1], 2),
                "mask_easting": round(mask_xy[0], 2),
                "mask_northing": round(mask_xy[1], 2),
                "ground_coordinates_match": image_xy == mask_xy,
            }
        )
    return pd.DataFrame(rows)


def scene_table(pairs):
    """Check each source scene id against the Landsat-8 overpass USGS lists."""
    rows = []
    for pair in pairs:
        parsed = parse_patch_name(pair["image_path"].name)
        image_geo = _geo_tags(pair["image_path"])
        fire = _fire_pixel(pair["mask"])
        easting, northing = _ground_xy(image_geo, *fire)
        zone, northern = _zone(image_geo["crs"])
        latitude, longitude = _utm_to_latlon(easting, northing, zone, northern)
        usgs = _usgs_scene(parsed["wrs_path"], parsed["wrs_row"], parsed["acquired"])
        bbox = None if usgs is None else usgs["bbox"]
        inside = False
        if bbox is not None:
            west, south, east, north = bbox
            inside = west <= longitude <= east and south <= latitude <= north
        rows.append(
            {
                "file_scene_id": parsed["scene"],
                "wrs_path": parsed["wrs_path"],
                "wrs_row": parsed["wrs_row"],
                "acquired": parsed["acquired"],
                "usgs_scene_id": None if usgs is None else usgs["usgs_scene"],
                "usgs_acquired_utc": None if usgs is None else usgs["acquired_utc"],
                "fire_latitude": round(latitude, 5),
                "fire_longitude": round(longitude, 5),
                "fire_inside_usgs_scene": inside,
            }
        )
    return pd.DataFrame(rows)


def band_table(pairs):
    stacked = np.stack([pair["image"] for pair in pairs])
    rows = []
    for index, (number, name) in enumerate(BANDS):
        channel = stacked[:, :, :, index]
        rows.append(
            {
                "channel": index,
                "landsat_band": number,
                "name": name,
                "shape": f"{channel.shape[1]}x{channel.shape[2]}",
                "dtype": str(channel.dtype),
                "min": int(channel.min()),
                "max": int(channel.max()),
                "mean": round(float(channel.mean()), 1),
            }
        )
    return pd.DataFrame(rows)


def sample_table(pairs):
    rows = []
    for pair in pairs:
        image = pair["image"]
        mask = pair["mask"]
        rows.append(
            {
                "patch": pair["patch"],
                "image_shape": f"{image.shape[0]}x{image.shape[1]}x{image.shape[2]}",
                "image_dtype": str(image.dtype),
                "mask_shape": f"{mask.shape[0]}x{mask.shape[1]}",
                "mask_dtype": str(mask.dtype),
                "schroeder_fire_pixels": pair["fire_pixels"]["schroeder"],
                "murphy_fire_pixels": pair["fire_pixels"]["murphy"],
                "kumar_roy_fire_pixels": pair["fire_pixels"]["kumar-roy"],
            }
        )
    return pd.DataFrame(rows)


def band_name_text():
    """One line per band, in file order. Same list for every patch."""
    return "\n".join(f"Band {number}: {name}" for number, name in BANDS)


def false_color(image):
    """Stretch bands 7, 6, and 2 into an RGB picture. Fire is bright in band 7."""
    channels = []
    for landsat_band in DISPLAY_BANDS:
        plane = image[:, :, band_index(landsat_band)].astype(np.float32)
        low, high = np.percentile(plane, (2, 98))
        if high <= low:
            high = low + 1
        channels.append(np.clip((plane - low) / (high - low), 0, 1))
    return np.stack(channels, axis=-1)


def main():
    archive = download_samples()
    extract_samples(archive)
    pairs = load_pairs(SAMPLE_COUNT)
    image = pairs[0]["image"]
    mask = pairs[0]["mask"]
    print(f"\nRead {len(pairs)} image-and-mask pairs")
    print(f"Image shape {image.shape} {image.dtype}")
    print(f"Mask shape {mask.shape} {mask.dtype}")
    print("Band names are not stored in the TIFF. They follow the paper's channel order.")
    print("\nBands")
    print(band_table(pairs).to_string(index=False))
    print("\nPairs with the most Schroeder fire pixels")
    print(sample_table(pairs).to_string(index=False))
    print("\nFile ids and pairing rule")
    print(pairing_table(pairs).to_string(index=False))
    print("\nPixel alignment")
    print(alignment_table(pairs).to_string(index=False))
    print("\nSource scene ids")
    print(scene_table(pairs).to_string(index=False))


if __name__ == "__main__":
    main()
