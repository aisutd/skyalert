"""The live-trained next-day spread model (v2): channels and dataset format.

See docs/spread_retrain_plan.md. This module is the single source of truth for the channel
order, shared by the dataset builder (training/build_spread_dataset.py) and the training code.
The model class, normalization and prediction are added by the training step (R3).
"""

# Input channels, in order. All are (64, 64) on backend.spread_grid.TileGrid (south-up rows,
# degree pixels), raw units, NaN = missing.
CHANNELS = (
    "fire_t",  # VIIRS NOAA-20 fire on day t: 0/1 footprint mask (fire_points.footprint_mask, scale 1.0)
    "fire_t1",  # same, day t-1
    "fire_t2",  # same, day t-2
    "frp_t",  # log1p(summed FRP in MW of day-t detections whose centre is in the pixel), 0 where none
    "elevation",  # metres, terrain.elevation
    "wind_u",  # eastward wind (m/s): -vs*sin(th), per gridMET cell and day, then the weather recipe
    "wind_v",  # northward wind (m/s): -vs*cos(th), same
    "tmmn",  # K, gridMET recipe (weather_layer) over the weather days
    "tmmx",  # K
    "sph",  # kg/kg
    "pr",  # mm
    "erc",  # NFDRS energy release component
    "pdsi",  # gridMET drought, latest pentad on or before t
)
LABEL = "fire_t+1"  # VIIRS NOAA-20 fire on day t+1, same mask recipe as fire_t

# Weather days relative to the fire day t: the mean of t-2 and t-1, what live data has
# (gridMET runs about two days behind).
WEATHER_OFFSETS = (-2, -1)
FIRE_FOOTPRINT_SCALE = 1.0  # VIIRS pixels are 0.4 to 0.8 km, so mostly one 1 km cell
VIIRS_SATELLITES = ("N20",)

# Dataset files under data/retrain/ (written by training/build_spread_dataset.py):
#   {split}_inputs.npy  float16 (n, len(CHANNELS), 64, 64), NaN kept for missing (opened with mmap)
#   {split}_labels.npy  uint8 (n, 64, 64), 1 = fire on t+1
#   tiles.csv           one row per tile: split, index (row in the split arrays), fire_day,
#                       center_lat, center_lon, fire_t_px, label_px, new_fire_px, region
#   channel_stats.csv   per-channel low/high (0.1/99.9 percentiles), mean, std from train only
SPLIT_YEARS = {"train": range(2018, 2023), "val": (2023,), "test": (2024, 2025)}
