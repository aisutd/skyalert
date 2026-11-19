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
python -m flood.geoid          # ~1.7 GB: manifest + 450 selected tiles into data/flood/geoid/
python -m flood.train          # bundle into data/flood/bundles/flood-unet-<UTC time>/
python -m flood.evaluate       # <bundle>/evaluation/report.md
python -m flood.predict --sample-id <a test sample_id from the manifest>
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

## Results: baseline `flood-unet-20261119T025417Z-107bfbad`

Trained on 240 tiles / 130 events, stopped at epoch 21 (best val flood IoU 0.260 at epoch 13),
about 14 minutes on an M-series laptop (MPS). Full report:
[examples/evaluation/report.md](examples/evaluation/report.md); bundle metadata:
[examples/evaluation/bundle.json](examples/evaluation/bundle.json). Weights are not committed.

| held-out set | method | flood IoU | Dice | precision | recall | dry-scene alarms | permanent water called flood |
|---|---|---|---|---|---|---|---|
| test: 110 tiles, 47 events | **U-Net (selected on val)** | 0.276 | 0.433 | 31.3% | 70.5% | 31/44 | 52.9% |
| | U-Net + permanent-water reference | 0.372 | 0.542 | 43.4% | 72.3% | 28/44 | 1.9% |
| | VH < −22 dB + reference (tuned on val) | 0.264 | 0.418 | 32.2% | 59.7% | 44/44 | 0.8% |
| 2026 floods: 50 tiles, 7 events | **U-Net (selected on val)** | 0.423 | 0.594 | 51.9% | 69.5% | 12/20 | 35.3% |
| | U-Net + permanent-water reference | 0.501 | 0.667 | 62.1% | 72.2% | 9/20 | 0.0% |
| | VH < −22 dB + reference | 0.402 | 0.574 | 53.0% | 62.5% | 18/20 | 0.0% |

Reading these honestly:

- The U-Net only slightly beats a tuned radar threshold on the main test events (0.276 vs 0.264).
- It over-maps flood: 580 km² predicted vs 257 km² labelled on test. The main error is calling permanent water flood (53% of permanent-water pixels), which also drives most dry-scene false alarms (31 of 44 dry tiles).
- Val (13 events) picked "U-Net alone", but both held-out sets favour adding the permanent-water reference. The variant was **not** switched after seeing test, because that would select on test. The next run needs a larger val split (more events) so this choice can be made on val.
- Treat this as a working baseline, not a deployable detector.

Next steps, all to be decided on val, not test: a larger val split; the permanent-water reference as an input channel or the default post-processing; pre/post-event change detection (doubles radar download); and rerunning on the T2 dataset decision.

## Example result for the dashboard

[examples/EMSR501-1-9/](examples/EMSR501-1-9/) holds one held-out test prediction in the shared result contract. It covers Albania, acquired 2021-02-12, with 24.8 km² of flooded water predicted and a tile flood IoU of 0.66:

- `result.json`: provenance, acquisition time, valid coverage, area and WGS84 bounds.
- `flood.geojson`: polygons, simplified to 10 m.

It is ready for T7 (map layer) and T12 (fixtures). No dashboard code was changed in this branch.
