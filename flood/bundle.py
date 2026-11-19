"""Load a saved flood model bundle and refuse it if it does not match this code's input contract."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import contract
from .data import ContractError
from .model import build


def latest(root="data/flood/bundles"):
    found = sorted(p for p in Path(root).glob("flood-unet-*") if (p / "bundle.json").exists())
    if not found:
        raise FileNotFoundError(f"no bundle under {root}; run python -m flood.train")
    return found[-1]


def load(path, device="cpu"):
    path = Path(path)
    meta = json.loads((path / "bundle.json").read_text())
    if meta.get("hazard") != contract.HAZARD or meta.get("task") != contract.TASK:
        raise ContractError(f"bundle is for {meta.get('hazard')}/{meta.get('task')}, not flood/detection")
    if meta["preprocessing_version"] != contract.PREPROCESSING_VERSION:
        raise ContractError(f"bundle preprocessing {meta['preprocessing_version']} != code {contract.PREPROCESSING_VERSION}")
    if meta["input_schema"]["bands"] != contract.INPUT_SCHEMA["bands"]:
        raise ContractError(f"bundle bands {meta['input_schema']['bands']} != {contract.INPUT_SCHEMA['bands']}")
    weights = (path / meta["weights_file"]).read_bytes()
    if hashlib.sha256(weights).hexdigest() != meta["weights_sha256"]:
        raise ContractError("weights.pt does not match the checksum in bundle.json")
    model = build(meta["architecture"]).to(device)
    model.load_state_dict(torch.load(path / meta["weights_file"], map_location=device, weights_only=True))
    model.eval()
    mean = np.array(meta["normalization"]["mean"], "float32")
    std = np.array(meta["normalization"]["std"], "float32")
    return model, meta, mean, std


def device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
