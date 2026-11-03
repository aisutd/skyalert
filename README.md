# SkyAlert

## Description

SkyAlert is a computer vision system for detecting wildfires and floods from satellite imagery. It uses a classifier to flag affected regions, a segmentation model to map damage at pixel level, and engineered satellite features to provide actionable information through an interactive dashboard.

## Planned Technologies

- Python and JavaScript
- Streamlit or Gradio with Folium
- PyTorch, ResNet, U-Net, Rasterio, and scikit-learn
- NASA FIRMS, DeepGlobe, Google Earth Engine, and Copernicus EMS

## Fire Map Setup

`main.py` downloads active fire detections for the contiguous United States from NASA FIRMS and saves an interactive map to `fires.html`. The map has a switch between clickable fire dots and a fire density heatmap.

1. Get a free NASA FIRMS map key at [firms.modaps.eosdis.nasa.gov/api/map_key](https://firms.modaps.eosdis.nasa.gov/api/map_key/).

2. Copy the example environment file and put your key in it:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env`:

   ```
   FIRMS_MAP_KEY=your_nasa_firms_map_key
   ```

   `.env` is listed in `.gitignore`, so your key stays on your machine and is never pushed. Do not put the key directly in `main.py` or the notebook.

3. Create a virtual environment and install the dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. Build and open the map:

   ```bash
   python main.py
   open fires.html
   ```

   Wait for the `Done!` line before opening the map. If the script is stopped while saving, `fires.html` will be empty.

`notebooks/fires.ipynb` loads the same data into a pandas dataframe for exploring the columns. To run it in Cursor or VS Code, select the `.venv` Python kernel.

## Training Data

`docs/dataset_options.md` compares candidate training datasets. Landsat-8 Active Fire trains the fire detector, and Next Day Wildfire Spread trains the next-day spread model.

`notebooks/ndws.ipynb` loads a sample of Next Day Wildfire Spread and shows its inputs, labels, and fire masks. On first run it downloads one 120 MB file into `data/ndws/`, which is not committed. Select the `.venv` kernel and run all cells. To print the same tables in the terminal instead:

```bash
python notebooks/inspect_ndws.py
```

`notebooks/landsat.ipynb` loads the Landsat-8 Active Fire sample, not a continent archive. On first run it downloads a 231 MB zip into `data/landsat8/`, which is not committed, and shows five image-and-mask pairs with their dimensions and band names. Select the `.venv` kernel and run all cells. To print the same tables in the terminal instead:

```bash
python notebooks/inspect_landsat.py
```

## Model Training

There are two U-Net models. Each notebook prepares its data, trains U-Net Light (about 1.9 million parameters, from Pereira et al. 2021), and scores it on the test split.

- `notebooks/train_unet.ipynb` detects active fire in Landsat-8 patches, from bands 7, 6, and 2. It uses the Landsat-8 Active Fire North America datasets, about 23 GB. It writes `index.csv`, the weights `detection_unet.pt`, `detection_history.csv` per epoch, and `detection_results.csv`.
- `notebooks/train_spread.ipynb` predicts where a fire will be tomorrow from today's fire, weather, fuel, and terrain. It uses Next Day Wildfire Spread, about 4 GB. It writes `spread_index.csv`, `spread_feature_stats.csv` for input scaling, `spread_baseline.csv` (the score of predicting that tomorrow's fire is today's fire), the weights `spread_unet.pt`, `spread_history.csv`, and `spread_results.csv`.

Training runs on Kaggle, where the datasets are mounted and nothing is downloaded. It trains for up to 50 epochs and stops early once val stops improving. Run locally, each notebook downloads a small part of its data into `data/` and trains for 2 epochs, only to check the code: about 100 MB for detection and about 570 MB for spread. Local outputs go to `data/training/local/`, so they never overwrite results downloaded from Kaggle.

1. Create a Kaggle API token at [kaggle.com/settings](https://www.kaggle.com/settings). Save it in `~/.kaggle/access_token`. The notebooks also read `~/.kaggle/kaggle.json`.
2. Run a notebook locally with the `.venv` kernel to check it.
3. Upload it to Kaggle as a private GPU notebook with its datasets attached:

   ```bash
   kaggle kernels push -p kaggle/detection
   kaggle kernels push -p kaggle/spread
   ```

   Each `kernel-metadata.json` holds the notebook id, the GPU setting, and the dataset list. Change the id to your own Kaggle username before pushing.

4. Check progress, then download the weights and scores when the status is complete:

   ```bash
   kaggle kernels status notlokii/skyalert-fire-spread-unet
   kaggle kernels output notlokii/skyalert-fire-spread-unet -p data/training
   ```

### Results

Scores on the test split, which the models never trained on.

| Model | Precision | Recall | IoU | Compared with |
|---|---|---|---|---|
| Fire detection | 82.4% | 93.3% | 0.778 | Pereira et al. U-Net Light on the voting mask: 90.2%, 96.5%, 0.873, with a random patch split rather than our split by location |
| Next-day spread | 24.5% | 52.5% | 0.200 | Copying today's fire: 35.7%, 27.3%, 0.183 |

Detection precision and recall count fire pixels against the voting mask. Spread scores count 1 km pixels that burn the next day, leaving out uncertain pixels.

## Demos

The demo scripts run the trained models on test data and save figures for presentations to `docs/figures/`. They need the weights and stats downloaded from Kaggle into `data/training/`.

```bash
python demos/detector_california.py
python demos/spread_examples.py
```

- `detector_california.py` runs the detector on two California fires from August 2020 that it never trained on, in the southern Sierra Nevada and the Trinity Alps. It downloads the 38 patches it needs, about 50 MB, and writes `docs/figures/detector_042035.png` and `detector_045032.png`, plus a map and per-patch tables in `data/demos/california/`. Each row of a figure is one 7.7 km patch: the satellite image, the model's fire in red, and the label in yellow, all from the same moment.
- `spread_examples.py` shows the spread model on test fires and writes `docs/figures/spread_examples.png`. Each row is one 64 km area: today's fire, the model's forecast for tomorrow, and what burned the next day. The first three rows are the best forecasts among test fires with at least 30 km² burning the next day, and the last row is the median one.

## Project Layout

```
main.py                       FIRMS live fire map, writes fires.html
notebooks/fires.ipynb         FIRMS data exploration
notebooks/ndws.ipynb          Next Day Wildfire Spread exploration
notebooks/inspect_ndws.py     Download and parse helper for ndws.ipynb
notebooks/landsat.ipynb       Landsat-8 Active Fire sample
notebooks/inspect_landsat.py  Download and parse helper for landsat.ipynb
notebooks/train_unet.ipynb    Fire detection U-Net training
notebooks/train_spread.ipynb  Next-day fire spread U-Net training
kaggle/detection/             Kaggle upload settings for train_unet.ipynb
kaggle/spread/                Kaggle upload settings for train_spread.ipynb
demos/                        Run the trained models and save figures
docs/dataset_options.md       Training dataset comparison
docs/figures/                 Demo figures for presentations
data/                         Downloaded datasets and trained weights, not committed
```

## Docker Setup

Make sure Docker Desktop is installed and running.

```bash
docker compose up -d
docker compose down
docker compose ps
```

This is a generic starter configuration. The team can add project-specific dependencies and startup commands later.
