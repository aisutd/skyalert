# Retraining the spread model for live use: plan

Option 1 from the failed gate (`docs/spread_forecast_plan.md`, "Gate result"): train a new model
on tiles built exactly the way the live forecast builds them, so training matches live use.
Branch `feat-spread-forecast`. Every builder works from this file; if code and this file
disagree, stop and fix one of them.

## Goal

Forecast where fire will be burning the next day, including where it spreads, better than simple
rules. Success on held-out years (test split, scored once at the end):
1. Higher IoU than **every** baseline below, with the 95% bootstrap interval of the difference
   above 0.
2. Real recall of **new fire**: pixels burning on t+1 that weren't burning on t. Persistence scores
   0 here by construction; the model must be clearly above 0.
If that fails, we stop and decide the next step with the team (no shipping a model that doesn't
pass).

## Mac safety (hard rules)

The user asked that this never strain the Mac.
- **One heavy job at a time.** Never run data building and training together, or two trainings.
- **Low priority:** every long job runs under `nice -n 10`.
- **CPU:** `torch.set_num_threads(4)`; NumPy/BLAS threads capped at 4 (`OMP_NUM_THREADS=4`,
  `VECLIB_MAXIMUM_THREADS=4`).
- **GPU (MPS):** batch size at most 64; `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5`, so PyTorch can't take
  more than half the GPU memory. Also set `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4`: the high ratio alone
  makes PyTorch refuse to start (its default low ratio, 1.4, is above 0.5). CPU fallback is fine.
- **Memory:** process memory under about 6 GB (the Mac has 19 GB). Datasets are float16 memory-mapped
  files on disk, never loaded whole into RAM.
- **Disk:** everything under `data/` (gitignored), at most about 25 GB total. Check free space first;
  stop if fewer than 50 GB would remain.
- **Time:** a single training run at most about 30 minutes; at most about 8 training runs overall.
  Each run logs its wall time.
- **Network politeness:** at most 2 concurrent downloads per service; FIRMS limit 5000
  transactions per 10 minutes (stay far below); never print the FIRMS key.
- Stop and report rather than push past any of these.

## What the new model sees and predicts

Forecast setup, the same in training and live:
- **Day t** = the latest complete UTC day of fire detections. The forecast is for **t+1**.
- **Fire source: VIIRS NOAA-20 only** (`VIIRS_NOAA20_SP` until 2026-07-31, `VIIRS_NOAA20_NRT`
  after).
  - Why VIIRS: it is live until well past MODIS's end in 2027, and its 375 m pixels give much
    sharper fire masks.
  - Why NOAA-20 only: NOAA-21 starts in 2024, so mixing them would change how much fire the
    labels contain from year to year.
  - Rasterized with `fire_points.footprint_mask` at scale 1.0 (VIIRS pixels are about 0.4 to
    0.8 km, so this mostly marks one cell).
- **Weather: what live will have.** gridMET is about 2 days behind, so training uses the mean of
  days t−2 and t−1 (`live` lag), not t−1 and t.

Channels (all at the existing `TileGrid`: south-up, degree pixels, 64 × 64):

| Channel | Source | Notes |
|---|---|---|
| fire_t | VIIRS N20, day t | 0/1 footprint mask |
| fire_t1 | VIIRS N20, day t−1 | fire history: growth direction and speed |
| fire_t2 | VIIRS N20, day t−2 | |
| frp_t | VIIRS N20, day t | log1p of summed FRP per pixel (MW) |
| elevation | Terrarium (terrain.py) | |
| wind_u, wind_v | gridMET th + vs | eastward and northward wind components, so wind has no 0/360 jump and flips/rotations stay physical |
| tmmn, tmmx, sph, pr, erc | gridMET (t−2, t−1) mean | same recipe as now (weather_layer) |
| pdsi | gridMET drought | latest pentad on or before t |

NDVI and population are dropped (no live source). Label: `fire_t+1`, the VIIRS N20 footprint mask.
No pixel is "uncertain": FIRMS has no cloud mask, so a cloudy pixel looks like no fire (a known limit).

## Data

- **Seasons:** fire seasons (April to November) 2018–2025 (VIIRS N20 archive starts 2018-04-01).
- **Splits by year, so the same fire never lands in two splits:** train 2018–2022, val 2023,
  test 2024–2025. Test is scored once, at the end.
- **Tiles:**
  - Centred on a day-t VIIRS fire pixel, as when a user clicks a live fire.
  - To get many tiles without duplicates: per day, cluster fire pixels and take one tile per cluster,
    with a random offset of up to ±16 pixels so the fire isn't always dead centre.
  - Tiles whose fire dies out by t+1 are kept: the model must learn that too.
  - Target about 25,000 train tiles and about 4,000 each for val and test (fewer if the data runs out).
- **Weather source for bulk building:** whole yearly gridMET files downloaded once
  (`/thredds/fileServer/MET/{var}/{var}_{YYYY}.nc`, HDF5, 80–150 MB each) and read with `h5py`. The
  layers must be **identical** to `gridmet.weather_layer` (same unpacking, node grid anchored at
  gridMET's corner, bicubic): a test builds sample tiles both ways and requires agreement to
  float precision. About 7 vars × 8 years × about 120 MB ≈ 7 GB.
- **FIRMS:** VIIRS N20 SP for CONUS, DAY_RANGE 5 per request, cached on disk as CSV (about 500 requests).
- **Storage:** shards of float16 arrays (`inputs (n, C, 64, 64)`, `label (n, 64, 64)` as uint8) plus
  a `tiles.csv` index (date, centre, split, fire pixel counts), under `data/retrain/`.
- **Dataset card:** per-channel stats from train only, used for normalization (saved like
  `spread_feature_stats.csv`).

## Training

- **Code:** `backend/spread_model_v2.py` (model, channels, normalization) and
  `training/train_spread_v2.py` (training loop). The U-Net Light architecture is unchanged (about
  1.9 million parameters); only the input channels differ.
- **Loss:** BCE with positive weighting, compared against BCE + Dice, because the old model
  over-forecast about 4×.
- **Augmentation:** the 8 flips and rotations, with wind_u and wind_v transformed to stay physical.
- **Threshold:** chosen on val to maximize IoU (the metric we judge by), never on test.
- **Variants:** a handful only (Mac budget):
  - loss type
  - positive weight
  - with and without fire history
  - optionally fine-tuning from the NDWS weights for the channels they share
- **Early stopping and selection:** by val IoU at the threshold re-chosen on val each epoch
  (PR-AUC is logged too); test is opened once for the winner.

## Baselines (all on the same tiles and masks)

1. Persistence: tomorrow = today.
2. Persistence dilated by 1 and by 2 pixels.
3. Persistence plus downwind growth: today's fire plus copies shifted 1..r pixels along the unit
   vector of the mean (wind_u, wind_v) over today's fire pixels (r = 1, 2); plain persistence
   below 0.5 m/s.
4. The old NDWS model fed the matching live inputs (MODIS-trained, for reference).

## Steps

| Step | What | Owner |
|---|---|---|
| R1 | gridMET bulk reader (`backend/gridmet_bulk.py` or a data-prep module) with the equality test against `weather_layer` | data agent |
| R2 | Tile sampler and dataset builder (`training/build_spread_dataset.py`), quick sample first, then full | data agent |
| R3 | Model v2, training loop, baselines, evaluation (`training/`), built and tested on the quick sample while R2 runs | training agent |
| R4 | Full training runs (after R2 finishes; one at a time) | training agent |
| R5 | Independent review of R1 to R4 | reviewer |
| R6 | Test-split scoring of the winner and the baselines, then report to the team | lead |
| Gate | If the goal is met: build the API and UI (old plan steps 4 to 6) with the new model | |

## Rules for builders

- Never guess a unit, variable name, scale factor, or orientation. Reuse the verified backend
  code (spread_grid, terrain, gridmet recipe, fire_points) instead of re-implementing it.
- Missing data is NaN, never a made-up number.
- Own only your assigned files. Don't commit; the lead commits.
- Follow the Mac safety rules above at all times.
