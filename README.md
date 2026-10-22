# SkyAlert

## Description

SkyAlert is a computer vision system for detecting wildfires and floods from satellite imagery. It uses a classifier to flag affected regions, a segmentation model to map damage at pixel level, and engineered satellite features to provide actionable information through an interactive dashboard.

## Planned Technologies

- Python and JavaScript
- Streamlit or Gradio with Folium
- PyTorch, ResNet, U-Net, Rasterio, and scikit-learn
- NASA FIRMS, DeepGlobe, Google Earth Engine, and Copernicus EMS

## Running SkyAlert from scratch

SkyAlert has two parts that run at the same time:

- **Backend** (Python, data only): `src/firms_client.py` fetches fire detections from NASA FIRMS, `src/data_cleaning.py` cleans them, `main.py` chains the two, and `dashboard.py` (FastAPI) serves the result as JSON on port 8000.
- **Frontend** (React, all the UI): `frontend/` draws the dashboard on port 5173: stat cards, the fire map with a popup and heatmap, and the Analytics table. It gets every piece of data from the backend through `frontend/src/lib/api.js`.

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

You need **two terminals**, both starting in the `skyalert` folder.

**Terminal 1, backend:**

```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python dashboard.py
```

Leave it running. It's ready when it prints `Uvicorn running on http://127.0.0.1:8000`.

**Terminal 2, frontend:**

```bash
cd frontend
npm run dev
```

Then open **http://localhost:5173** in your browser. The first load takes a few seconds while the backend downloads the latest fires from NASA. After that, the data is cached for 10 minutes.

To stop, press `Ctrl+C` in each terminal.

### Check the data without the dashboard

```bash
source .venv/bin/activate
python main.py
```

This fetches and cleans the fires and prints the first few rows. No server or browser is needed.

`fires.ipynb` loads the same data into a pandas dataframe for exploring the columns. To run it in Cursor or VS Code, select the `.venv` Python kernel.

### Troubleshooting

| Problem | Fix |
|---|---|
| `FIRMS_MAP_KEY is not set` | Do step 3: create `.env` in the `skyalert` folder with your key. |
| Dashboard opens but shows "Couldn't load stats" or no fire dots | The backend isn't running. Start Terminal 1 and refresh. |
| `Address already in use` (port 8000 or 5173) | The backend or frontend is already running in another terminal. Use that one or stop it first. |
| `npm run dev` complains about the Node version | Install Node.js 20.19+ or 22.12+. |

See `APIS.md` for every API the project calls and exposes.

## Docker Setup

Make sure Docker Desktop is installed and running.

```bash
docker compose up -d
docker compose down
docker compose ps
```

This is a generic starter configuration. The team can add project-specific dependencies and startup commands later.
