"""Download one Next Day Wildfire Spread shard and print its structure.

The Kaggle release is a zip of TFRecord files. Each example is a 64 by 64 km
patch. Twelve grids are inputs and FireMask is the next-day label. This script
fetches only the first shard, then shows the feature list and a few samples.
"""

import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/fantineh/next-day-wildfire-spread"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ndws"
TILE_SIZE = 64
SAMPLE_COUNT = 5

FEATURE_INFO = {
    "elevation": ("input", "Elevation, meters"),
    "th": ("input", "Wind direction, degrees clockwise from north"),
    "vs": ("input", "Wind speed, meters per second"),
    "tmmn": ("input", "Minimum temperature, kelvin"),
    "tmmx": ("input", "Maximum temperature, kelvin"),
    "sph": ("input", "Specific humidity"),
    "pr": ("input", "Precipitation, millimeters"),
    "pdsi": ("input", "Palmer Drought Severity Index"),
    "NDVI": ("input", "Vegetation index, stored scaled rather than -1 to 1"),
    "population": ("input", "Population density"),
    "erc": ("input", "Energy release component, BTU per square foot"),
    "PrevFireMask": ("input", "Yesterday's fire: 1 fire, 0 no fire, -1 uncertain"),
    "FireMask": ("label", "Tomorrow's fire: 1 fire, 0 no fire, -1 uncertain"),
}


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk:
            raise EOFError(f"Download ended {remaining} bytes early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def zip64_sizes(extra):
    index = 0
    while index + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, index)
        payload = extra[index + 4 : index + 4 + data_size]
        index += 4 + data_size
        if header_id == 1:
            uncompressed, compressed = struct.unpack_from("<QQ", payload)
            return uncompressed, compressed
    raise ValueError("Zip member is missing a ZIP64 size field")


def download_first_shard():
    """Save the first TFRecord inside the Kaggle zip."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["curl", "-fsSL", "-A", "Mozilla/5.0", DATASET_URL],
        stdout=subprocess.PIPE,
    )
    stream = process.stdout
    try:
        header = read_exact(stream, 30)
        _, _, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack(
            "<IHHHHHIIIHH", header
        )
        name = read_exact(stream, name_len).decode()
        extra = read_exact(stream, extra_len)
        if flags != 0 or method != 8:
            raise ValueError(f"Unexpected zip member {name}: flags={flags} method={method}")
        uncompressed_size, compressed_size = zip64_sizes(extra)
        destination = DATA_DIR / name
        if destination.exists() and destination.stat().st_size == uncompressed_size:
            process.terminate()
            print(f"Using existing {destination}")
            return destination

        print(f"Downloading {name} ({compressed_size / 1e6:.0f} MB compressed)")
        decompressor = zlib.decompressobj(-15)
        remaining = compressed_size
        written = 0
        with destination.open("wb") as outfile:
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise EOFError(f"Download ended {remaining} bytes early")
                remaining -= len(chunk)
                piece = decompressor.decompress(chunk)
                outfile.write(piece)
                written += len(piece)
            outfile.write(decompressor.flush())
        if written != uncompressed_size:
            raise ValueError(f"Expected {uncompressed_size} bytes, wrote {written}")
        print(f"Saved {destination}")
        return destination
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait()


def read_varint(data, index):
    result = 0
    shift = 0
    while True:
        byte = data[index]
        index += 1
        result |= (byte & 0x7F) << shift
        if byte < 128:
            return result, index
        shift += 7


def parse_fields(data):
    fields = []
    index = 0
    while index < len(data):
        key, index = read_varint(data, index)
        field_number = key >> 3
        wire_type = key & 7
        if wire_type == 0:
            value, index = read_varint(data, index)
        elif wire_type == 2:
            length, index = read_varint(data, index)
            value = data[index : index + length]
            index += length
        elif wire_type == 5:
            value = data[index : index + 4]
            index += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")
        fields.append((field_number, wire_type, value))
    return fields


def parse_float_list(data):
    values = []
    for field_number, wire_type, value in parse_fields(data):
        if field_number != 1:
            continue
        if wire_type == 5:
            values.append(np.frombuffer(value, dtype="<f4"))
        elif wire_type == 2:
            values.append(np.frombuffer(value, dtype="<f4"))
    if not values:
        return np.array([], dtype=np.float32)
    return np.concatenate(values)


def parse_example(blob):
    """Return one TF Example as a dict of feature name to a 64 by 64 array."""
    features_message = next(
        value for field_number, _, value in parse_fields(blob) if field_number == 1
    )
    parsed = {}
    for field_number, _, entry in parse_fields(features_message):
        if field_number != 1:
            continue
        entry_fields = parse_fields(entry)
        key = next(value for number, _, value in entry_fields if number == 1).decode()
        feature = next(value for number, _, value in entry_fields if number == 2)
        float_list = next(
            (value for number, _, value in parse_fields(feature) if number == 2),
            None,
        )
        if float_list is None:
            raise ValueError(f"{key} is not stored as a float list")
        grid = parse_float_list(float_list)
        if grid.size != TILE_SIZE * TILE_SIZE:
            raise ValueError(f"{key} has {grid.size} values, expected {TILE_SIZE * TILE_SIZE}")
        parsed[key] = grid.reshape(TILE_SIZE, TILE_SIZE)
    return parsed


def read_examples(path, limit):
    examples = []
    with path.open("rb") as record_file:
        while len(examples) < limit:
            length_bytes = record_file.read(8)
            if not length_bytes:
                break
            (length,) = struct.unpack("<Q", length_bytes)
            record_file.read(4)
            blob = read_exact(record_file, length)
            record_file.read(4)
            examples.append(parse_example(blob))
    return examples


def feature_table(examples):
    keys = list(FEATURE_INFO)
    for example in examples:
        for key in example:
            if key not in keys:
                keys.append(key)

    rows = []
    for key in keys:
        role, meaning = FEATURE_INFO.get(key, ("unknown", ""))
        grids = [example[key] for example in examples if key in example]
        stacked = np.stack(grids)
        rows.append(
            {
                "feature": key,
                "role": role,
                "shape": f"{TILE_SIZE}x{TILE_SIZE}",
                "meaning": meaning,
                "min": stacked.min(),
                "max": stacked.max(),
                "mean": stacked.mean(),
            }
        )
    table = pd.DataFrame(rows)
    for column in ("min", "max", "mean"):
        table[column] = table[column].map(lambda value: round(float(value), 4))
    return table


def sample_table(examples):
    rows = []
    for index, example in enumerate(examples):
        row = {"sample": index}
        for key, grid in example.items():
            row[f"{key}_mean"] = round(float(grid.mean()), 4)
        for mask_name in ("PrevFireMask", "FireMask"):
            mask = example[mask_name]
            row[f"{mask_name}_fire_pixels"] = int((mask == 1).sum())
            row[f"{mask_name}_clear_pixels"] = int((mask == 0).sum())
            row[f"{mask_name}_uncertain_pixels"] = int((mask == -1).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    shard = download_first_shard()
    examples = read_examples(shard, SAMPLE_COUNT)
    print(f"\nRead {len(examples)} samples from {shard.name}")
    print(f"Keys in the first sample: {list(examples[0])}")
    print("\nFeature structure")
    print(feature_table(examples).to_string(index=False))
    print("\nFirst samples")
    print(sample_table(examples).to_string(index=False))


if __name__ == "__main__":
    main()
