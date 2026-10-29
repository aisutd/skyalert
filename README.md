# SkyAlert

## Description

SkyAlert is a computer vision system for detecting wildfires and floods from satellite imagery. It uses a classifier to flag affected regions, a segmentation model to map damage at pixel level, and engineered satellite features to provide actionable information through an interactive dashboard.

## Planned Technologies

- Python and JavaScript
- Streamlit or Gradio with Folium
- PyTorch, ResNet, U-Net, Rasterio, and scikit-learn
- NASA FIRMS, DeepGlobe, Google Earth Engine, and Copernicus EMS

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

## Running SkyAlert from scratch

SkyAlert has two parts that run at the same time:

- **Backend** (Python, data only): `backend/firms_client.py` fetches fire detections from NASA FIRMS, `backend/data_cleaning.py` cleans them, `main.py` chains the two, `backend/fire_context.py` gets the weather and place name for the fire popup, and `backend/dashboard.py` (FastAPI) serves it all as JSON on port 8000.
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

`fires.ipynb` loads the same data into a pandas dataframe for exploring the columns. To run it in Cursor or VS Code, select the `.venv` Python kernel.

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

See `APIS.md` for every API the project calls and exposes.

## Docker Setup

Make sure Docker Desktop is installed and running.

```bash
docker compose up -d
docker compose down
docker compose ps
```

This is a generic starter configuration. The team can add project-specific dependencies and startup commands later.
