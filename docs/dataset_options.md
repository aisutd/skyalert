# Dataset Options for Fire Prediction

## What we need

SkyAlert's main goal is to predict where active wildfire will be, across the United States. Mapping land that has already burned is a second phase.

Training needs labeled examples. Each example has two parts:

- **Input:** a patch of land, as one or more grids of pixels.
- **Label:** the correct answer for that patch. For fire prediction, this is a mask of where fire is the next day. For burned-area mapping, it is a mask of which pixels burned.

A dataset supports our task only if its labels answer the question we want the model to answer.

### Why NASA FIRMS is not a candidate

FIRMS is what `main.py` already uses for the live fire map. Each row is a single fire point: latitude, longitude, brightness, confidence, date, and fire radiative power. FIRMS has no image patch around the point, no weather or vegetation conditions from before the fire, and no record of where the fire went next. It records detections, not training examples, so it stays the live map feed.

## Option 1: Next Day Wildfire Spread

- **Paper:** [Next Day Wildfire Spread (Huot et al.)](https://arxiv.org/abs/2112.02447)
- **Download:** [Kaggle: fantineh/next-day-wildfire-spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread)
- **Code:** [Google Research loader and export code](https://github.com/google-research/google-research/tree/master/simulation_research/next_day_wildfire_spread)
- **License:** CC-BY 4.0

### Bands

Each sample is a 64 by 64 km patch at 1 km resolution, so every grid is 64 by 64 pixels. There are 12 input grids:

| Name | What it is |
|---|---|
| `elevation` | Elevation, meters |
| `th` | Wind direction, degrees clockwise from north |
| `vs` | Wind speed, meters per second |
| `tmmn` | Minimum temperature, kelvin |
| `tmmx` | Maximum temperature, kelvin |
| `sph` | Specific humidity |
| `pr` | Precipitation, millimeters |
| `pdsi` | Palmer Drought Severity Index |
| `NDVI` | Vegetation index, stored scaled rather than -1 to 1 |
| `population` | Population density |
| `erc` | Energy release component, BTU per square foot |
| `PrevFireMask` | Fire on day t |

Most of these are weather, terrain, and population layers rather than satellite spectral bands. Only `NDVI` and the fire masks come from satellites. The dataset has no red, near-infrared, or shortwave infrared bands, so NBR cannot be computed from it.

### Labels

`FireMask` is the fire on day t+1 for the same patch. Each pixel is `1` for fire, `0` for no fire, or `-1` for uncertain, usually because cloud blocked the satellite. The fire masks come from MODIS satellite fire detections.

### Size

- 18,545 samples across the contiguous United States, from 2012 to 2020.
- Split into training, validation, and test sets, about 80, 10, and 10 percent.
- About 2 GB on Kaggle, as 19 TFRecord files.

### Does it support our task?

**Yes, for fire prediction.** The input is today's conditions plus today's fire, and the label is tomorrow's fire. That is the question SkyAlert wants to answer, and the coverage is the whole contiguous US, which matches our live map.

We checked this by downloading the first file, `next_day_wildfire_spread_eval_00.tfrecord`, and loading it in `notebooks/ndws.ipynb`. Every sample had the 12 inputs above plus `FireMask`, each a 64 by 64 grid. In sample 1, yesterday had 18 fire pixels and tomorrow had 39. Some of tomorrow's fire was not near any fire from yesterday, so the model has to learn new ignitions, not only spread.

Limits:

- At 1 km per pixel, it shows roughly where fire is, not exact fire edges.
- It is a one-day forecast, not same-day detection from an image.
- Pixels labeled `-1` must be left out of training and scoring rather than counted as no fire.
- A few precipitation values are slightly below zero. The authors' code clips them to zero.

## Option 2: CaBuAr (California Burned Areas)

- **Paper:** [CaBuAr: California Burned Areas dataset for delineation](https://arxiv.org/abs/2401.11519)
- **Download:** [Hugging Face: DarthReca/california_burned_areas](https://huggingface.co/datasets/DarthReca/california_burned_areas)
- **Loader:** [TorchGeo CaBuAr](https://docs.torchgeo.org/en/stable/api/datasets/cabuar.html)

We did not download CaBuAr. The details below come from the paper and the dataset pages.

### Bands

12 Sentinel-2 Level-2A bands, resampled to 20 m per pixel: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, and B12.

These include red (B04), near-infrared (B08 or B8A), and shortwave infrared (B12), so both engineered features can be computed:

- NDVI = (NIR − Red) / (NIR + Red)
- NBR = (NIR − SWIR) / (NIR + SWIR)

When a pre-fire image is present, dNBR = NBR before − NBR after measures how badly each pixel burned.

### Labels

A binary mask for each post-fire image: `1` for burned area and `0` for no change. The masks are drawn from fire perimeters published by the California Department of Forestry and Fire Protection.

### Size

- 340 Sentinel-2 scenes of 5490 by 5490 pixels, covering California fires from 2015 to 2022.
- The ready-made version is 424 pairs of 512 by 512 patches, each with a pre-fire image, a post-fire image, and a mask.
- The data is stored as HDF5 files, with five folds plus a separate test set from the ChaBuD 2023 challenge.

### Does it support our task?

**Not for fire prediction. Yes for burned-area mapping.** The label is land that has already burned, not where fire will be next, so it cannot train the main SkyAlert model. It does fit the second phase: a U-Net that outlines burned land, using NDVI and NBR as features.

Limits:

- It covers California only, so a model trained on it may not transfer to the vegetation in other states.
- The ready-made 512 by 512 patches only include patches with at least one burned pixel. A fire or no-fire classifier would need unburned patches cut from the full scenes.
- Some events have no pre-fire image, so dNBR is not available for every sample.

## Recommendation

**Use Next Day Wildfire Spread to train the fire prediction model.** It is the only option whose labels answer our main question, it covers the whole contiguous US, and we have confirmed its structure by loading it.

**Keep CaBuAr for the burned-area phase.** It has the spectral bands needed for NDVI and NBR and pixel-level burn labels, which Next Day Wildfire Spread does not provide.

**Keep FIRMS as the live map** of fires burning now.

## How we will evaluate

For each pixel the model predicts fire or no fire, and we compare that with the label:

- **Correct fire:** the model and the label both say fire.
- **False alarm:** the model says fire and the label says no fire. Responders would be sent somewhere that is not burning.
- **Miss:** the label says fire and the model says no fire. A real fire goes unwarned.
- Pixels labeled `-1` (uncertain) are skipped.

We will report:

- **Intersection over union (IoU):** correct fire pixels divided by all pixels where either the model or the label says fire. This measures how well the predicted fire area overlaps the real one. 1 is a perfect match.
- **Recall:** the share of real fire pixels the model caught. A miss is the more dangerous error for first responders, so recall is reported alongside IoU rather than replaced by it.
- **Precision:** the share of predicted fire pixels that were real. This tracks false alarms.

Overall accuracy is not a useful score here. Most pixels are not on fire, so a model that always predicts no fire would look accurate while missing every fire.
