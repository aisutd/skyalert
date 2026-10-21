"""Download the Landsat-8 Active Fire sample and print image-mask pairs.

The full archive is about 192 GB. The authors also publish samples.zip,
about 231 MB: 100 patches of 256 by 256 pixels, ten bands each, with
Schroeder, Murphy, and Kumar-Roy fire masks. This script fetches that
sample only.
"""

import io
import re
import subprocess
import zipfile
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

# Only for drawing a picture. A screen has three color channels, so the view
# uses bands 7, 6, and 2. The array still has all ten.
DISPLAY_BANDS = (7, 6, 2)

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
            }
        )
    return pairs


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


if __name__ == "__main__":
    main()
