"""The input contract, label semantics and versions shared by training, evaluation and inference.

Anything that changes how pixels become model inputs must bump PREPROCESSING_VERSION, so a bundle
trained under one version refuses inputs prepared under another.
"""
import subprocess
from pathlib import Path

HAZARD, TASK = "flood", "detection"
PREPROCESSING_VERSION = "s1grd-post-db-v1"

# Input: one post-event Sentinel-1 GRD acquisition, in the band order the GeoTIFF names them.
INPUT_SCHEMA = {
    "sensor": "Sentinel-1 GRD (IW, dual-pol VV+VH)",
    "pass": "post-event (single image, no pre-event pair)",
    "bands": ["VV", "VH"],
    "file_units": "linear sigma0 (backscatter coefficient), float32",
    "model_units": "dB = 10*log10(sigma0), then (dB - train mean) / train std per band",
    "invalid": "sigma0 <= 0, NaN/inf, or validity == 0; filled with 0 after scaling and ignored by the loss",
    "grid": "projected CRS in metres (GEOID-Flood ships each event on its own UTM grid at 10 m)",
}

# Labels: CEMS Rapid Mapping delineations as released in GEOID-Flood v1.1.
CLASSES = ["background", "permanent water", "flooded water"]
BACKGROUND, PERMANENT_WATER, FLOOD = 0, 1, 2
IGNORE = 255  # outside the mapped area or no usable radar
LABEL_SEMANTICS = (
    "0 background, 1 permanent water, 2 flooded water (water outside the permanent-water reference "
    "at the post-event acquisition), 255 ignored. Flooded water is the CEMS delineation, so the output "
    "is observed flood extent at acquisition time, not a forecast."
)

# Sanity range for linear sigma0. Values outside it mean the file is in dB or another unit.
SIGMA0_MAX = 100.0


def code_revision():
    """Git commit of this checkout, marked -dirty if flood/ has uncommitted edits."""
    root = Path(__file__).resolve().parent.parent
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "flood"], cwd=root, capture_output=True, text=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
