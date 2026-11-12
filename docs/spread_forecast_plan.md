# Next-day spread forecast: build plan

The reference for building the live spread forecast on branch `feat-spread-forecast`.
Every builder works from this file. If code and this file disagree, stop and fix one of them.

## Goal

Click a fire on the Live Map and see where it is likely to be burning the next day: the model's
64 by 64 pixel tile around the fire (about 1 km per pixel), coloured by the trained spread model's
probability, with a line at the model's fire threshold (0.621). Every input layer is labelled
live, lagged, or filled, and the model's test scores are shown next to the forecast.

## Decisions (agreed 2026-11-11)

| Question | Decision |
|---|---|
| Fix the empty VIIRS S-NPP feed | On this branch, first. Done in `dc58603`: the map uses NOAA-20 + NOAA-21. NASA retired S-NPP fire data on 2026-11-02. |
| Fire source for the model | MODIS Terra (the training masks are Terra MOD14A1), VIIRS NOAA-20/21 as fallback. MODIS ends 2027. |
| Backtest gate | Stop after the backtest (step 3) and show the numbers before building the API and UI. |
| NDVI and population | Filled: NaN, which the model's scaling turns into the train mean. Labelled "filled". No Earthdata login. |

## Status

| Step | What | State |
|---|---|---|
| 0 | Dashboard uses NOAA-20 + NOAA-21; cleaning handles missing FRP | Done, `dc58603` |
| 1 | `backend/spread_model.py`, parity tests, demo uses it | Done, `bc8da3a` |
| 2 | Input builder: grid, terrain, gridMET, FIRMS fires, assembly | Done |
| 2b | Reconstruction check: rebuild located training tiles channel by channel | Done: `docs/spread_reconstruction_report.md` |
| 3 | `demos/backtest_spread.py`: score live-style inputs on past fires | Done: `docs/spread_backtest_report.md` |
| Review | Independent correctness review of the step 2 code | Done: 1 critical (weather node anchor), 1 major (fire layer on FIRMS failure), 3 minor; all fixed |
| Gate | Show backtest numbers to the team | **Failed** 2026-11-12: model level with persistence. Steps 4 to 6 on hold pending a decision |
| 4 | `GET /api/spread-forecast` | After the gate |
| 5 | Map overlay, popup button, legend | After the gate |
| 6 | README, APIS.md, credits | After the gate |

## Gate result (2026-11-12): failed

336 tiles centred on a day-t Terra fire, 168 fire-season days 2018–2024, all regions. Inputs built
with `5bef496`, fire masks with `cb082c5`. Live setting (weather lag t−3..t−2, MODIS Terra fire):

| Masks (inputs; labels) | Model IoU | Persistence IoU | Difference, 95% CI |
|---|---|---|---|
| footprint ×0.5; footprint ×0.5 (closest to training) | 0.141 | 0.150 | −0.009 [−0.030, 0.008] |
| centre pixel; centre pixel | 0.075 | 0.073 | +0.002 [−0.013, 0.014] |

The lead recomputed the centre-pixel row from the saved tiles and got the same numbers.
- The model forecasts about 4 times more pixels than burn (precision 15%, recall 65%). Even at the
  best threshold for these labels its IoU only reaches about 0.08 (centre-pixel labels).
- Without weather the IoU rises (0.164) because the forecast shrinks, so the weather adds little.
- Almost all the signal is in the West.
- For scale, the model only beat persistence by 0.017 IoU on its own NDWS test split (0.200 vs 0.183),
  and by 0.035 on the 19 reconstructed tiles. The backtest can't show an edge that small, and on
  live-style tiles there isn't one.
- Likely reasons the live setting is harder: tiles are centred on today's fire (180 of 336 had no
  Terra fire the next day; NDWS tiles were centred on next-day fire), and the labels have no cloud
  mask (NDWS left cloudy pixels out).

## Resume here

Paused on 2026-11-11 so the work could move. To continue:
1. Rerun the independent review of the step 2 code (`ffaf108`) and fix anything it confirms.
2. Finish and run `demos/reconstruct_spread_tiles.py` (step 2b). The draft is uncommitted; check it
   before trusting it. Caches are in `data/cache/reconstruction/`.
3. Finish and run `demos/backtest_spread.py` (step 3), quick mode first. Caches are in `data/cache/backtest/`.
4. Stop at the gate and show the backtest numbers before steps 4 to 6.

## How the training tiles were made

From the authors' export code (google-research `simulation_research/next_day_wildfire_spread`,
commit `cf61877d4c`) and checks on our local shards. Full notes with quotes:
`research/ndws_export_spec.md` (scratchpad). D is the label day (t+1); t = D−1 is "today".

| Channel | Source | Time window | Resampling to the tile | Units |
|---|---|---|---|---|
| elevation | SRTM `USGS/SRTMGL1_003` | static | Earth Engine default (pyramid, then nearest) | m |
| th, vs, tmmn, tmmx, sph, pr, erc | gridMET `IDAHO_EPSCOR/GRIDMET` | days t−1 and t, `.median()` (of two values) | reprojected to a 20 km grid (`atScale(20000)`), then bicubic | deg from (clockwise from N); m/s; K; K; kg/kg; mm; NFDRS index |
| pdsi | `GRIDMET/DROUGHT` | [D−5, D) median, one pentad image | gridMET 1/24° grid, then bicubic | unitless |
| NDVI | VIIRS `VNP13A1` | [D−8, D) | about 1 km nearest neighbour in the released data | NDVI × 10000 |
| population | GPWv4.11 density | static | nearest | people/km² |
| PrevFireMask | MODIS **Terra** `MOD14A1` `FireMask`, day t only | one UTC day | nearest | classes 7, 8, 9 → 1; 3, 5 → 0; everything else → −1 |
| FireMask (label) | same, day t+1 | one UTC day | nearest | same |

Tile geometry, from the released data:
- Pixels are 0.0089831528° (= 1000 / 111319.49 m) on **both** axes, so a tile is about 64 km
  north–south but only about 49 km east–west at 40° N. The export used Earth Engine's default
  WGS84 degree grid at scale 1000.
- **Stored arrays are south-up: row 0 is the south edge, column 0 the west edge, no transpose.**
  Confirmed by matching 44 training tiles' elevation against a real DEM: 43 matched best only
  under a vertical flip (NCC 0.96 to 0.998; every other transform at most 0.93). Our own
  TileGrid + terrain rebuild of three located tiles correlates 0.997 to 0.999 with the stored
  arrays, with no flipping (`tests/test_spread_alignment.py`).
- Degree pixels confirmed the same way: a longitude step scaled by 1.00 fits best in all 10 tiles
  tested; square-km pixels fit clearly worse. Pixel edges sit on multiples of 0.0089831528° from 0°.
- Elevation behaves like a 1 km area mean, not a point sample, which is what terrain.py does.
- Tiles were centred on a label-day fire pixel at array index [31, 31] and kept only if
  PrevFireMask had fire. No clipping or scaling was done before the TFRecords were written.

Unsettled points, all tested in the backtest instead of guessed:
- Public code vs. the release: with `lag=1` every non-label window shifts a day earlier. The
  release has 12.4% of tiles with no label fire, which the public code can't produce.
- Median of two gridMET days vs. "the day t average" in the paper. We use the mean of the two days.
- Wind direction was averaged and interpolated as plain numbers (no 0/360 wrap). We do the same,
  to match training.

## How we rebuild each channel from live data

Verified by probing on 2026-11-11 (`research/data_sources.md` in the scratchpad).

| Channel | Live source | Our recipe | Status shown |
|---|---|---|---|
| elevation | AWS Terrarium tiles, zoom 10 (about 120 m pixels at 38° N) | Mean of the zoom-10 pixels whose centres fall in each tile pixel; negative values → 0 (SRTM has no ocean values; Death Valley's −86 m becoming 0 doesn't matter after scaling) | static |
| th, vs, tmmn, tmmx, sph, pr, erc | gridMET over THREDDS OPeNDAP, yearly files `{var}_{YYYY}.nc` | Fetch the chosen days, unpack, average the days per gridMET cell, block-mean onto the 20 km node grid anchored at gridMET's corner, bicubic to the tile's pixel centres | lagged (gridMET runs about 2 days behind) |
| pdsi | gridMET `MET/pdsi/pdsi.nc` | Latest pentad on or before day t, bicubic from the 1/24° grid to the pixel centres | lagged |
| NDVI | none | NaN | filled |
| population | none | NaN | filled |
| PrevFireMask | FIRMS MODIS (SP until 2026-08-31, NRT after), Terra only, one UTC day | 1 in every tile pixel covered by a detection's footprint (0.5 × scan by 0.5 × track km), else 0. No −1 (we can't see clouds) | live |
| PrevFireMask fallback | FIRMS VIIRS NOAA-20 / NOAA-21 | Same, used only when MODIS has no Terra detection in the tile or the request fails | live (fallback) |

### gridMET facts (verified)

- Base `https://www.reacchpna.org/thredds/dodsC/MET`. Mirrors with identical answers:
  `https://thredds.northwestknowledge.net/thredds/dodsC/MET`, `https://tds-proxy.nkn.uidaho.edu/thredds/dodsC/MET`.
- Request: `{base}/{var}/{var}_{YYYY}.nc.ascii?{name}[d0:1:d1][i0:1:i1][j0:1:j1]`. Ranges are
  inclusive. Index order is **[day][lat][lon]** (the `dimensions` attribute saying "lon lat time" is wrong).
- Values are packed UInt16: `physical = raw × scale_factor + add_offset`; raw 32767 = missing.
  **Read scale and offset from each file's `.das`, never by variable name**: tmmn and tmmx share
  the name `air_temperature` but have offsets 210 and 220.

| short | name in file | scale | offset |
|---|---|---|---|
| th | `wind_from_direction` | 1.0 | 0 |
| vs | `wind_speed` | 0.1 | 0 |
| tmmn | `air_temperature` | 0.1 | 210 |
| tmmx | `air_temperature` | 0.1 | 220 |
| sph | `specific_humidity` | 1e-5 | 0 |
| pr | `precipitation_amount` | 0.1 | 0 |
| erc | `energy_release_component-g` | 1.0 | 0 |
| pdsi | `pdsi` in `MET/pdsi/pdsi.nc` | 0.01 | −15 |

- Grid: 585 × 1386 cells at 1/24°. Latitude **descends** from 49.400000000000006;
  longitude ascends from −124.76666663333334. `i = round((49.4 − lat) × 24)`,
  `j = round((lon + 124.76666663333334) × 24)` (cell centres). Read the coordinate arrays once
  rather than trusting these constants.
- Time: `day` is days since 1900-01-01. The latest day is the last value in the file's `day`
  array (2026-11-09 on 2026-11-11). `{var}_{next year}.nc` returns 404 until it exists, so
  windows that span January 1 read two files.
- PDSI: one file, 5-day pentads from 1980-01-05 to 2026-11-06. Whether a pentad's date is its
  start or its end is unverified: take the latest pentad dated on or before day t.
- Yearly files for 2012–2025 exist for all seven variables (needed for the backtest).
- About 0.3 s per small request.

### FIRMS facts (verified)

- `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/{SOURCE}/{w,s,e,n}/{DAY_RANGE}/{YYYY-MM-DD}`.
  With a date, DAY_RANGE counts **forward** from it. DAY_RANGE must be 1–5 (0 returns an empty CSV silently).
- Coverage: MODIS_SP 2000-11-01 to 2026-08-31; MODIS_NRT 2026-09-01 to now;
  VIIRS_NOAA20_SP 2018-04-01 to 2026-07-31; VIIRS_NOAA20_NRT 2026-08-01 to now;
  VIIRS_NOAA21_NRT 2024-01-17 to now. Read `/api/data_availability/csv/{KEY}/all` rather than hard-coding.
- MODIS columns: `latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight[,type]`.
  `satellite` is `Terra`/`Aqua` in SP; check the NRT spelling (it may be `T`/`A`) and accept both.
  Confidence is 0–100. `acq_time` is HHMM without padding. `acq_date` is UTC.
- Limit: 5000 transactions per 10 minutes.

### Elevation tiles (verified)

`https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png`, XYZ Web Mercator,
256 × 256 RGB, `elevation = R×256 + G + B/256 − 32768`. Zoom 10: about 9 tiles per tile area.
Zoom 10 and below show sea depths offshore; we clamp negatives to 0.

## Module layout and contracts

Every layer function takes a `TileGrid` and returns a float32 `(64, 64)` array in **model
orientation**, with NaN where data is missing. Orientation and geometry exist only in `TileGrid`;
no other module flips or transposes anything.

- `backend/spread_grid.py`
  - `PIXEL_DEG = 1000 / 111319.49079327357`, `TILE_SIZE = 64`, `CENTER_INDEX = 31`, `ROW0_IS_SOUTH = True`.
  - `TileGrid.around(lat, lon, geometry="degree" | "km")` makes a tile whose pixel `[31, 31]`
    contains the point. Pixels sit on the global grid anchored at (0°, 0°) with size `PIXEL_DEG`
    (degree), or `PIXEL_DEG / cos(lat)` in longitude (km).
  - Attributes `lats`, `lons` (pixel centres, 64 × 64, model orientation), `lat_step`, `lon_step`,
    and `bounds` (south, west, north, east of the outer pixel edges).
  - `cell_of(lats, lons)` gives row and column arrays (−1 outside); `rasterize(lats, lons)` gives a
    0/1 float32 mask.
  - `to_north_up(array)` for display only.
- `backend/terrain.py`: `elevation(grid)` from Terrarium zoom 10, with an in-memory tile cache.
- `backend/gridmet.py`
  - `fetch(var, start_day, end_day, south, west, north, east)` returns physical values plus the
    cell-centre lats and lons.
  - `latest_day(var)`.
  - `weather_layer(grid, var, days)` builds a layer with the 20 km node block mean + bicubic.
  - `pdsi_layer(grid, day)`.
  - Mirror fallback, retries, an in-memory cache keyed by (var, year, box, days).
- `backend/fire_points.py`: `fire_points(south, west, north, east, day, family="MODIS" | "VIIRS")`
  returns detections for one UTC day, choosing SP or NRT by date, with a normalized `satellite`
  column. `fire_mask(grid, day, prefer="MODIS")` returns the mask plus which source was used.
  Reuses `backend/firms_client.py` for key handling and errors.
- `backend/spread_inputs.py`: `build_inputs(lat, lon, fire_day, weather_days=None, geometry="degree", fire_source="MODIS")`
  returns the `(12, 64, 64)` array in `INPUT_FEATURES` order plus per-channel metadata
  (source, dates used, status, notes) and the grid.
- `demos/backtest_spread.py` (step 3).
- `tests/`: pytest. Network tests use `@pytest.mark.network` and run only with `--network`.

### Weather node grid

The export reprojected gridMET with `GRIDMET_proj.atScale(20000)`, which changes the scale but
keeps gridMET's origin. So node edges sit at whole multiples of `NODE_DEG = 20000 / 111319.49°`
from gridMET's outer corner: the north edge of its first row (49.4 + 1/48) and the west edge of its
first column (−124.7667 − 1/48), **not** from (0°, 0°).

Evidence: bicubic leaves faint "knots" at node centres. In the stored training weather (three
located tiles, five variables) they fall at global pixel phase 11.0 (lat) and about 18.5 (lon)
mod 20. Running our node_grid + bicubic on random fields with each candidate origin and comparing
the knot pattern (L1 distance, lat / lon): (0°, 0°) 1.03 / 1.99; Earth Engine's asset corner
49.41667 0.67 / 0.03; gridMET file corner 49.42083 0.07 / 0.03. Locked in by
`tests/test_spread_alignment.py::test_weather_node_grid_lines_up_with_training_knots`.
Found by the independent review on 2026-11-11 (the first version used (0°, 0°)).

### Bicubic

Separable cubic convolution (Keys, a = −0.5) on the regular source grid, evaluated at the tile's
pixel centres. Earth Engine's exact kernel is undocumented; small differences are smoothed away by
the 20 km source grid. Missing source values: use only the valid taps and divide by the sum of
their weights; if that sum is below 0.5 (or there are no valid taps) the output is NaN, which
later becomes the train mean. This keeps weather near coastlines instead of losing it to one
missing ocean node.

## Backtest design (step 3)

- **Cases.** Past fire days in the fire season (June to October), 2018–2024, contiguous US.
  Tiles are centred on a day-t Terra fire pixel (as a person clicking a live fire would see it).
  The label is day t+1 Terra detections, rasterized the same way. A few hundred tiles, spread over
  many days and regions, with a fixed seed.
- **Scores.** Pixel precision, recall, and IoU at threshold 0.621, plus PR-AUC, for the model and
  for persistence (tomorrow = today) on the same tiles. Bootstrap 95% intervals over tiles.
- **Variants.** One change at a time from the baseline (south-up, degree grid, weather days t−1 and t,
  MODIS Terra, point rasterization):
  - Orientation: all 8 flips and rotations. The true one should score best.
  - Geometry: degree vs km pixels.
  - Weather days: (t−1, t), t only, (t−2, t−1), and the live lag (the latest two gridMET days
    on or before t−2).
  - Fire source: Terra only, Terra + Aqua, VIIRS NOAA-20.
  - Ablation: all weather channels NaN, to see how much weather matters.
- **Pass mark for the gate.** The model must beat persistence on IoU, with the live-lag weather
  and live fire source, and the bootstrap interval must say whether that win is real.

## Reconstruction check (step 2b): results

`demos/reconstruct_spread_tiles.py`, full report in `docs/spread_reconstruction_report.md`.
43 DEM-located training tiles were dated; 19 confidently (weather match and FIRMS fire overlap).
- **Weather window: mean of gridMET days t−1 and t** (public code, `lag=0`). Lowest error for every
  variable (e.g. wind speed MAE 0.074 m/s vs 0.21 for day t only and 0.36 for t−2, t−1).
- **PDSI: latest pentad dated on or before t** reproduces the stored channel exactly.
- **Wind direction:** no convention error (median 1.8° on tiles away from north).
- **Rebuilt channels vs stored** (median corr): elevation 0.992, pdsi 1.000, tmmn 0.963, tmmx 0.944,
  vs 0.963, sph 0.945, pr 0.960, erc 0.904, th 0.943.
- **Model cost** (pooled IoU against the stored next-day label, threshold 0.621): stored inputs
  0.208; NDVI and population filled 0.205; our full rebuild 0.201; our rebuild with live-lag
  weather 0.186; persistence 0.166.
- **Fire mask:** centre pixels caught only about 40% of the MOD14A1 fire pixels (IoU 0.34).
  Marking each detection's footprint at half its scan × track size raises that to 0.63 and the
  model's IoU on stored inputs to 0.209. Now the default (`fire_points.FOOTPRINT_SCALE = 0.5`).
  Chosen on these 19 tiles, so the backtest re-checks it on independent tiles.

What the check was designed to settle:

Three training tiles have known locations (DEM match). For each, find its fire day t: the
stored PrevFireMask should match FIRMS MODIS Terra detections rasterized on day t, and the
stored weather should match gridMET. Then rebuild all 12 channels with `build_inputs` and compare
each with the stored channel (correlation, mean absolute difference), and compare the model's
forecast from stored vs rebuilt inputs. This settles:
- which gridMET days the release used (t−1 and t, or one day earlier: the `lag` question)
- whether the 20 km block mean + bicubic recipe reproduces the stored weather texture
- how close FIRMS-based fire masks are to MOD14A1

## Fire layer when FIRMS can't be reached

If every FIRMS request fails, `fire_mask` raises instead of returning an empty mask: not knowing
today's fire is different from "no fire today". `build_inputs` then marks PrevFireMask "missing"
(strict mode raises). The API (step 4) must refuse to forecast without a fire layer.

## Open checks

- PDSI pentad date convention (the reconstruction check can settle it).
- Closed: orientation and geometry (DEM match). Closed: MODIS_NRT spells satellites "Terra"/"Aqua", like SP.

## Rules for builders

- Never guess a unit, variable name, scale factor, or orientation. Check it against this file, the
  research notes, or a live response, and say in a comment where it came from.
- Missing data is NaN, never a made-up number.
- No new dependencies (numpy, pandas, requests, Pillow, torch are available; there is no scipy).
- Own only the files you're assigned. Don't commit; the lead commits each step.
- Never print, log, or write the FIRMS key.

## Sources

- Huot et al., Next Day Wildfire Spread: https://arxiv.org/abs/2112.02447
- Export code: https://github.com/google-research/google-research/tree/master/simulation_research/next_day_wildfire_spread
- gridMET THREDDS: https://www.reacchpna.org/thredds/catalog/MET/erc/catalog.html
- NASA on retiring S-NPP fire data: https://www.earthdata.nasa.gov/news/blog/transitioning-from-s-npp-viirs-omps-what-nasa-firms-users-need-know
- MODIS end of mission: https://ladsweb.modaps.eosdis.nasa.gov/learn/modis-to-viirs-transition
- Terrarium tiles: https://registry.opendata.aws/terrain-tiles/
