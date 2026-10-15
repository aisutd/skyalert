# Dataset Options for Fire Detection

## What we need

SkyAlert's fire model detects active fire in a satellite image taken today. A ResNet classifier flags whether a patch contains fire, and a U-Net marks which pixels are on fire.

Training needs labeled examples. Each example has two parts:

- **Input:** a satellite image patch, with spectral bands.
- **Label:** the correct answer for that patch, as a mask of which pixels are on fire. The ResNet label comes from the same mask: a patch is "fire" if any pixel in it is on fire.

A dataset supports our task only if:

1. Its inputs are satellite spectral bands, not weather or map layers.
2. Its labels mark active fire in the same image, not burned land and not tomorrow's fire.
3. Its bands include red, near-infrared, and shortwave infrared, so we can compute NDVI and NBR:
   - NDVI = (NIR − Red) / (NIR + Red), which measures living vegetation.
   - NBR = (NIR − SWIR) / (NIR + SWIR), which drops where land has burned.

### Where NASA FIRMS fits

FIRMS is what `main.py` uses for the live fire map. Each row is a single fire point, with no image around it, so FIRMS cannot be a training input. It can be a label: TS-SatFire, below, builds its training labels from the VIIRS active fire detections that FIRMS publishes.

## Option 1: Landsat-8 Active Fire

- **Paper:** [Active Fire Detection in Landsat-8 Imagery (Pereira et al., 2021)](https://arxiv.org/abs/2101.03409)
- **Download and code:** [GitHub: pereira-gha/activefire](https://github.com/pereira-gha/activefire)
- **License:** CC-BY 4.0

### Bands

10 Landsat-8 bands per patch, at 30 m per pixel. Landsat-8 band 8, the 15 m panchromatic band, is left out.

| Band | What it sees |
|---|---|
| 1 | Coastal aerosol |
| 2, 3, 4 | Blue, green, red |
| 5 | Near-infrared |
| 6, 7 | Shortwave infrared |
| 9 | Cirrus cloud |
| 10, 11 | Thermal infrared |

Red (4), near-infrared (5), and shortwave infrared (7) give NDVI and NBR. Shortwave infrared is also where a burning fire shows up brightest.

### Labels

A fire mask for every patch: each pixel is fire or no fire. There are two kinds of masks:

- **Algorithm masks** for all patches, made by three established fire detection methods (Schroeder, Murphy, and Kumar-Roy), plus masks where two or all three agree.
- **Hand-labeled masks** for a separate test set of 9,044 patches. These let us measure the model against people, not against the algorithms it learned from.

### Size

- 146,214 patches of 256 by 256 pixels, from 8,194 Landsat-8 scenes taken worldwide in August and September 2020.
- About 192 GB of images plus about 22 GB of masks.
- Split into 11 regions, roughly by continent, so we can download only North America.

### Does it support our task?

**Yes.** The input is a satellite image with real spectral bands, the label is active fire in that same image, and the patches are already cut to a size a U-Net can take. The authors trained U-Net models on it. Their best combination found 92.4 percent of the hand-labeled fire pixels, and 87.2 percent of the pixels it marked as fire were real.

Limits:

- Most labels come from detection algorithms rather than people, so the model learns their mistakes too. The hand-labeled test set is the honest score.
- It only covers two months of 2020.
- Landsat-8 photographs the same place only every 16 days, so a model trained on it can confirm and map fires but cannot watch a fire day to day.
- Most patches have very few fire pixels, so the model has to be trained to not ignore them.

## Option 2: TS-SatFire

- **Paper:** [TS-SatFire (Scientific Data, 2025)](https://www.nature.com/articles/s41597-025-06271-3)
- **Download:** [Kaggle: z789456sx/ts-satfire](https://www.kaggle.com/datasets/z789456sx/ts-satfire)
- **Code:** [GitHub: zhaoyutim/TS-SatFire](https://github.com/zhaoyutim/TS-SatFire)

### Bands

Six VIIRS bands. VIIRS is the same sensor behind the FIRMS feed in `main.py`.

| Band | What it sees | Pixel size |
|---|---|---|
| I1 | Red | 375 m |
| I2 | Near-infrared | 375 m |
| I3 | Shortwave infrared | 375 m |
| I4 | Mid-infrared, where fire heat peaks | 375 m |
| I5 | Thermal infrared | 375 m |
| M11 | Shortwave infrared | 750 m |

Red (I1), near-infrared (I2), and shortwave infrared (M11) give NDVI and NBR. Each fire has daytime and nighttime images. The dataset also includes weather, terrain, land cover, and fuel layers, but the detection task uses the satellite bands.

### Labels

Each image has two masks, one for active fire and one for burned area.

- **Training labels** for active fire come from the NASA VIIRS active fire product, which FIRMS serves. The authors checked them by eye and removed areas where the labels did not match the bright spots in the image.
- **Test labels** were drawn by hand for 17 fires around the world.

### Size

- 179 wildfires, mostly in the contiguous US, from January 2017 to October 2021.
- 3,552 images, following each fire from start to finish, as GeoTIFF files.
- About 71 GB in total.

### Does it support our task?

**Yes, with a resolution trade-off.** The inputs are real spectral bands, the labels are active fire in the same image, and the fires are in the US. It uses the same satellite as our live map, so a model trained on it could later run on fresh VIIRS images, which arrive twice a day.

Limits:

- A 375 m pixel is about 12 Landsat pixels across, so fire edges are coarse.
- The authors found that image-only models like U-Net did worse at active fire detection than models that also see earlier images of the same place, because fire pixels are so sparse at this resolution.
- Only 179 fires, and the files are organized by fire and date, not as ready-made patches.

## Recommendation

**Train the fire detector on Landsat-8 Active Fire.** It matches the plan most directly: real spectral bands, active fire labels in the same image, ready-made 256 by 256 patches for the ResNet and U-Net, enough bands for NDVI and NBR, and a hand-labeled test set for an honest score. We can start with the North America region instead of the full 192 GB.

**Keep TS-SatFire as the next step** if we want the detector to run on the same twice-daily VIIRS images that feed our live map. It would need models that also look at earlier images of the same place.

We did not download either dataset. The numbers above come from the papers and dataset pages.

### Other datasets considered

- **Next Day Wildfire Spread** forecasts tomorrow's fire rather than detecting today's, and 10 of its 12 inputs are weather, terrain, and population layers. We inspected it in `notebooks/ndws.ipynb` before ruling it out.
- **WildfireSpreadTS** has some VIIRS reflectance bands, but it also forecasts tomorrow's fire rather than detecting today's.
- **CaBuAr** labels land that has already burned, not active fire. It may be useful for a later burned-area phase.

## How we will evaluate

For each pixel the model predicts fire or no fire, and we compare that with the label:

- **Correct fire:** the model and the label both say fire.
- **False alarm:** the model says fire and the label says no fire. Responders would be sent somewhere that is not burning.
- **Miss:** the label says fire and the model says no fire. A real fire goes unreported.

We will report:

- **Intersection over union (IoU):** correct fire pixels divided by all pixels where either the model or the label says fire. This measures how well the predicted fire area overlaps the real one. 1 is a perfect match.
- **Recall:** the share of real fire pixels the model caught. A miss is the more dangerous error for first responders, so recall is reported alongside IoU rather than replaced by it.
- **Precision:** the share of predicted fire pixels that were real. This tracks false alarms.

Overall accuracy is not a useful score here. Almost every pixel is not on fire, so a model that always says no fire would look accurate while missing every fire.

Scores on the hand-labeled test set count. Scores on algorithm-made labels only show how well the model copies those algorithms.
