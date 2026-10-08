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

`fires.ipynb` loads the same data into a pandas dataframe for exploring the columns. To run it in Cursor or VS Code, select the `.venv` Python kernel.

## Docker Setup

Make sure Docker Desktop is installed and running.

```bash
docker compose up -d
docker compose down
docker compose ps
```

This is a generic starter configuration. The team can add project-specific dependencies and startup commands later.
