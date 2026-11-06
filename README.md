# SkyAlert

## Description

SkyAlert is a computer vision system for detecting wildfires and floods from satellite imagery. It uses a classifier to flag affected regions, a segmentation model to map damage at pixel level, and engineered satellite features to provide actionable information through an interactive dashboard.

## Technologies

- **Backend:** Python, FastAPI, pandas
- **Frontend:** React + React Leaflet (the UI first built as the HTML prototype in `prototype/`)
- **Models:** PyTorch U-Nets, trained on Kaggle GPUs
- **Live data:** NASA FIRMS (VIIRS NOAA-20), Open-Meteo, GRIDMET, OpenStreetMap Nominatim, Esri map tiles
- **Training data:** Landsat-8 Active Fire and Next Day Wildfire Spread (flood: Sen1Floods11 planned)

## Quick start

First time only: put your NASA FIRMS key in `.env` (see step 3 below).

Start the backend first, then the frontend, then open the browser. Both terminals start in the `skyalert` folder, not inside `backend/` or `frontend/`.

**Terminal 1, backend** (in the `skyalert` folder):

```bash
python3 -m venv .venv              # first time only
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # first time only
python -m backend.dashboard        # API on http://127.0.0.1:8000
```

**Terminal 2, frontend** (in the `skyalert` folder):

```bash
cd frontend
npm install                        # first time only
npm run dev                        # prints the address to open
```

Wait for Terminal 1 to print `Uvicorn running on http://127.0.0.1:8000`, then open the address Terminal 2 printed (usually **http://localhost:5173**). Full details and troubleshooting are below.

## Live prototype (single page, no npm)

`prototype/index.html` is the design prototype (overview that opens into the map) running on live data. The backend serves it, so only one terminal is needed:

```bash
source .venv/bin/activate
python -m backend.dashboard
```

Then open **http://127.0.0.1:8000/prototype/**. Open it through this address, not by double-clicking the file: the page gets its data from `/api` on the same server, so from `file://` it falls back to sample data and shows a banner saying so.

The first load can take several seconds while the backend downloads the fires from NASA and the weather for each one. After that the fires are cached for 10 minutes.

The page has two views. The **overview** shows the riskiest fire in California as one big number, with the map in a box on the right. **Focus on the map** opens the map full screen, with three floating panels: the summary (top left), the selected fire and the list of fires (right) and the weather (bottom). **Overview**, or `Esc`, goes back.

- **Fire** is live: today's NASA FIRMS detections across the US (red dots) and the 8 strongest fires in California, numbered on the map. Each one gets its place name, weather and the Fosberg fire-weather index from `/api/fire-context`, and the list is sorted by that index. Detections within 15 km of each other count as one fire. Click a fire dot or any other spot on the map to add it to the list and see its weather.
- **Flood** is still sample data.
- **Model on a past fire** (button on the overview, or the switch in the map view) draws the fire detector's output on the map at its real location: 4 Landsat-8 tiles in Sequoia National Forest, Aug 27, 2020, placed by matching them against Landsat scene 042/035 from that day. The spread predictor's examples are shown in the bottom panel, not on the map, because their test data has no locations.
- **Model results** (chart button) has the same results as a popup, plus the training-data stats. Images are in `prototype/model_results/`.
- **Spread model on today's data:** put `spread_unet.pt` and `spread_feature_stats.csv` from Kaggle in `data/training/` and restart the backend. The page then shows the model's chance of fire tomorrow for each fire (big number, list and a heat-map layer) instead of the fire-weather index. `backend/spread_forecast.py` builds the model's 12 inputs from FIRMS, GRIDMET and elevation tiles, and loads the model code from `notebooks/train_spread.ipynb`, like `demos/spread_examples.py`. NDVI and population aren't filled yet (they need a NASA Earthdata login), so they get the training average, as the training code does for missing pixels. Without the files, the page says it's waiting for the weights.

## Running SkyAlert from scratch

SkyAlert has two parts that run at the same time:

- **Backend** (Python, data only): `backend/firms_client.py` fetches fire detections from NASA FIRMS, `backend/data_cleaning.py` cleans them, `main.py` chains the two, `backend/fire_context.py` gets the weather and place name for each fire, `backend/spread_forecast.py` runs the spread model on live inputs, and `backend/dashboard.py` (FastAPI) serves it all as JSON on port 8000. It also serves the live prototype at `/prototype/`.
- **Frontend** (React, all the UI): `frontend/` runs on port 5173. It opens on the overview (today's top fire, its fire-weather index and weather); "Focus on the map" morphs into the full map view with the fire list, weather panel and the model on a past fire. Flood shows sample data until there's a flood model. Every backend call is in `frontend/src/lib/api.js`.

### 1. Install the tools (once per computer)

- **Python 3.10 or newer** (tested on 3.12). Check with `python3 --version`.
- **Node.js 20.19+ or 22.12+** (needed by Vite). Check with `node --version`.
- **Git**

### 2. Get the code

```bash
git clone https://github.com/aisutd/skyalert.git
cd skyalert
```

### 3. Add your NASA FIRMS key

1. Get a free map key at [firms.modaps.eosdis.nasa.gov/api/map_key](https://firms.modaps.eosdis.nasa.gov/api/map_key/).
2. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

3. Open `.env` and put your key in it:

   ```
   FIRMS_MAP_KEY=your_nasa_firms_map_key
   ```

`.env` is listed in `.gitignore`, so your key stays on your machine and is never pushed. Never put the key directly in code or in the notebook.

### 4. Set up the backend (once)

From the `skyalert` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Set up the frontend (once)

```bash
cd frontend
npm install
cd ..
```

### 6. Run it (every time)

You need **two terminals**, both starting in the `skyalert` folder. Go in this order:

1. Start the backend (Terminal 1) and wait until it's ready.
2. Start the frontend (Terminal 2).
3. Open the dashboard in your browser.

**Terminal 1, backend:**

```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m backend.dashboard
```

Run this from the `skyalert` folder: your prompt should end in `skyalert %`, not `backend %`. After activating, the prompt should start with `(.venv)`. If it shows `(base)` instead, you're on Anaconda's Python, so run `conda deactivate` and then activate again.

Leave it running. It's ready when it prints `Uvicorn running on http://127.0.0.1:8000`.

**Terminal 2, frontend:**

```bash
cd frontend
npm run dev
```

Then open the address it prints, usually **http://localhost:5173**. If another project is already using 5173, Vite picks the next free port (5174, 5175, …), so always use the one in the terminal. The first load takes a few seconds while the backend downloads the latest fires from NASA. After that, the data is cached for 10 minutes.

To stop, press `Ctrl+C` in each terminal.

### Check the data without the dashboard

```bash
source .venv/bin/activate
python main.py
```

This fetches and cleans the fires and prints the first few rows. No server or browser is needed.

`notebooks/fires.ipynb` loads the same data into a pandas dataframe for exploring the columns. To run it in Cursor or VS Code, select the `.venv` Python kernel.

### Troubleshooting

| Problem | Fix |
|---|---|
| `FIRMS_MAP_KEY is not set` | Do step 3: create `.env` in the `skyalert` folder with your key. |
| Dashboard opens but shows "Couldn't load stats" or no fire dots | The backend isn't running. Start Terminal 1 and refresh. |
| `ModuleNotFoundError: No module named 'backend'` | You're inside the `backend/` folder. Run `cd ..` to get back to `skyalert`, then `python -m backend.dashboard`. |
| `python dashboard.py` says "No such file" | The server moved to `backend/`. Use `python -m backend.dashboard` from the `skyalert` folder. |
| Map shows a 502 error for fire detections | The frontend can't reach the backend. Start Terminal 1 (or check it didn't crash), wait for the Uvicorn line, then refresh. |
| `Address already in use` (port 8000 or 5173) | The backend or frontend is already running in another terminal. Use that one or stop it first. |
| `npm run dev` complains about the Node version | Install Node.js 20.19+ or 22.12+. |
| Prototype shows "Live data isn't available… sample data" | The backend isn't running, or the page was opened as a file. Start the backend and open http://127.0.0.1:8000/prototype/. |
| Prototype says "Waiting for weights" | Expected until `spread_unet.pt` and `spread_feature_stats.csv` are in `data/training/`. Everything else works without them. |
| Prototype says "Inputs unavailable" | The weights are there, but GRIDMET didn't answer. Try again later; the fire-weather index is still shown. |
| 0 fires, or far fewer than usual | Check `FIRMS_SOURCE` in `main.py`. A satellite can stop sending data (SNPP did after 2026-11-02); switch to another VIIRS source such as `VIIRS_NOAA20_NRT` or `VIIRS_NOAA21_NRT`. |

See `APIS.md` for every API the project calls and exposes.

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
main.py                       Fetch and clean FIRMS fires without the dashboard
backend/                      FIRMS client, cleaning, fire context, spread forecast, FastAPI dashboard API
prototype/index.html          Live single-page prototype, served at /prototype/
prototype/model_results/      Detector and spread-model images used by the prototype
frontend/                     React app: src/App.jsx (state + overview ⇄ map morph), src/components/, src/lib/ (API, live data, sample data)
frontend/public/model_results/ Detector and spread-model images used by the React app
APIS.md                       Every API the project calls and exposes
docs/integration.md           How fetch, cleaning, map, and dashboard fit together
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

## Docker (not needed yet)

You do not need Docker to run anything in this repo. Use the Quick start steps above instead.

`docker-compose.yml` is an empty placeholder. It starts a bare Linux container with no Python and no project code running, so it cannot run the backend, the frontend, or the notebooks. It is kept so the team can turn it into a real setup later.
