# Flood detection (T5)

Segments **observed flood extent** in one post-event Sentinel-1 GRD image (VV, VH): background,
permanent water or flooded water per 10 m pixel, at the image's acquisition time. It is not a
flood forecast (that is T6) and a historical evaluation does not establish live coverage.

| file | role |
|---|---|
| `contract.py` | input schema, label semantics, preprocessing version |
| `geoid.py` | **provisional** GEOID-Flood adapter: event-split manifest + selective tile download |
| `data.py` | validated loader: CRS/transform/shape/band/unit/label checks, invalid pixels kept |
| `model.py` | the wildfire detector's light U-Net with a 3-class head |
| `train.py` | train on `train`, select on `val`, save a versioned bundle |
| `evaluate.py` | held-out event report with baselines, per-event table and examples |
| `predict.py` | batch inference → `classes.tif`, `flood.geojson`, `result.json` (shared result contract) |
| `bundle.py` | loads a bundle and rejects mismatched preprocessing, bands or weights |
| `tests/` | boundary tests on synthetic rasters (`python -m pytest flood/tests -q`) |

## Run

```bash
pip install -r requirements.txt
python -m flood.geoid          # ~1.7 GB: manifest.csv + 450 selected tiles into data/flood/geoid/
python -m flood.geoid --name manifest_v2 --val-events 40 --pin-eval data/flood/geoid/manifest.csv --train 300 --val 150
python -m flood.train --manifest data/flood/geoid/manifest_v2.csv --name flood-unet-v2
python -m flood.evaluate --bundles data/flood/bundles/flood-unet-v2-<time> --manifest data/flood/geoid/manifest_v2.csv
python -m flood.predict --sample-id <a test sample_id> --manifest data/flood/geoid/manifest_v2.csv
python -m pytest flood/tests -q
```

`data/` is git-ignored. Weights and data are not committed.

## Data and split

Source: [GEOID-Flood v1.1](https://huggingface.co/datasets/links-ads/geoid-flood) (CC BY 4.0;
modified Copernicus Sentinel-1 data; CEMS Rapid Mapping labels © European Union), the dataset
Ariana's flood notebook (`origin/flood_test`) already uses. **The dataset decision belongs to T2**
(Lokesh, with Ariana). If T2 chooses another source, add an adapter that writes the same manifest
columns; the loader, model, evaluation and inference stay as they are.

GEOID-Flood's own split is per area of interest, so 50 of its 210 events have tiles in more than one
of train/val/test (EMSR712-10 is train, EMSR712-1…9 are test). `geoid.py` instead groups tiles by
the connected components of flood event and the dataset's touching-AoI cluster, then assigns
whole groups 70/10/20 by tile count. It verifies no tile footprint is shared across splits.
The dataset's held-out tree (EMSR857-871, January-March 2026) is a later-in-time test; held-out
tiles that overlap any main-tree ground are dropped. Normalization is fitted on our train tiles,
not taken from the dataset card (those statistics include pixels from our test events).

Training tiles are 75% flooded; evaluation tiles 60% flooded and 40% dry, so dry-scene false
alarms can be measured. Pooled pixel metrics therefore describe this enriched sample.

## Results: `flood-unet-v2-20261119T034731Z-b5de5fba`

Trained on 300 tiles / 109 events (`manifest_v2.csv`), best val checkpoint at epoch 34.
Choices made on val only (38 events): U-Net alone beat U-Net + permanent-water reference, with the
flood threshold at 47/255 chosen for the best F2 (recall weighted 2x). The test and 2026 tiles are
the same pinned tiles as the first baseline (PR #10), scored once.
Full report: [examples/evaluation/report.md](examples/evaluation/report.md). Weights are not committed.

| held-out set | method | recall | precision | flood IoU | accuracy* | dry-scene alarms |
|---|---|---|---|---|---|---|
| test: 110 tiles, 47 events | **U-Net v2** | **72.0%** | 40.9% | **0.353** | 95.2% | 27/44 |
| | first baseline (PR #10) | 70.5% | 31.3% | 0.276 | n/a | 31/44 |
| | VH < -22.5 dB + reference | 57.1% | 35.2% | 0.278 | 94.6% | 43/44 |
| | never flood | 0% | n/a | 0 | 96.4% | 0/44 |
| 2026 floods: 50 tiles, 7 events | **U-Net v2** | **70.5%** | 58.9% | **0.472** | 97.0% | 7/20 |
| | first baseline (PR #10) | 69.5% | 51.9% | 0.423 | n/a | 12/20 |
| | VH < -22.5 dB + reference | 59.9% | 57.3% | 0.414 | 96.8% | 18/20 |

\*Per labelled pixel, flood vs not. About 96% of pixels are not flood, so "never flood" scores 96.4%:
**accuracy is not a useful measure here**. Read recall, precision and IoU.

Compared with the first baseline, test IoU rose from 0.276 to 0.353, test precision from 31% to 41%, and
dry-scene false alarms fell. Recall is about 71-72%: the model finds roughly 7 of every 10 flooded
pixels and still over-maps (453 km² predicted vs 257 km² labelled on test), mostly by calling
permanent water flood (40% of permanent-water pixels).

Higher recall is a policy choice: lowering the threshold trades precision for recall. That target
should be set as a rule on val (e.g. "recall >= 85% on val"), not tuned on test.
Not yet tried: the permanent-water reference as an input channel (`train.py --permwater-input`),
pre/post-event change detection, and more training tiles (limited here by 8 GB RAM and disk).

## Example result for the dashboard

[examples/EMSR501-1-9/](examples/EMSR501-1-9/) holds one held-out test prediction in the shared
result contract (Albania, acquired 2021-02-12, 28.8 km² flooded water predicted, tile flood IoU 0.84):
`result.json` (provenance, decision rule, acquisition time, valid coverage, area, WGS84 bounds) and
`flood.geojson` (polygons simplified to 10 m). Ready for T7 (map layer) and T12 (fixtures).
