import os
import requests
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from io import StringIO
from dotenv import load_dotenv

load_dotenv()
MAP_KEY = os.environ["FIRMS_MAP_KEY"]

# Contiguous United States: west, south, east, north
US_BOUNDS = "-125,24,-66,50"

# MODIS reports confidence as 0-100; VIIRS_SNPP_NRT detects more fires but only reports l/n/h
SOURCE = "MODIS_NRT"

url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{US_BOUNDS}/2"

response = requests.get(url, timeout=60)
print("Status code:", response.status_code)

# Parse into dataframe
df = pd.read_csv(StringIO(response.text))
print(f"Found {len(df)} active fires")
print(df[['latitude', 'longitude', 'brightness', 'confidence', 'acq_date', 'frp']].head())

ZOOM_START = 4
HEAT_RADIUS = 14
# Keep blur below the radius; a wider blur flattens every peak into pale blue
HEAT_BLUR = 8


def percentile_density_weights(lats, lons, zoom, radius):
    """Weight each fire so a heatmap cell sums to the percentile rank of its fire count.

    Leaflet.heat bins points into radius/2-pixel cells and colors each cell by
    its summed weight divided by the busiest cell. With raw counts, a few dense
    hotspots push almost everything else into blue. Ranking cells by count
    keeps the real ordering of densities but spreads it across the full color
    scale: red is the densest areas, blue the sparsest.
    """
    world_px = 256 * 2 ** zoom
    lat_rad = np.radians(lats)
    x = (lons + 180) / 360 * world_px
    y = (1 - np.log(np.tan(lat_rad) + 1 / np.cos(lat_rad)) / np.pi) / 2 * world_px
    cell = radius / 2
    cell_ids = pd.Series(list(zip((x // cell).astype(int), (y // cell).astype(int))))
    cell_counts = cell_ids.value_counts()
    cell_rank = cell_counts.rank(method="max", pct=True)
    counts = cell_ids.map(cell_counts).to_numpy()
    return cell_ids.map(cell_rank).to_numpy() / counts


# Build the interactive map
m = folium.Map(location=[39, -98], zoom_start=ZOOM_START, tiles=None)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    control=False,
).add_to(m)

# overlay=False makes these radio buttons, so only one layer shows at a time
dots = folium.FeatureGroup(name="Fire dots", overlay=False).add_to(m)
heat = folium.FeatureGroup(name="Fire density heatmap", overlay=False, show=False).add_to(m)

weights = percentile_density_weights(df['latitude'].to_numpy(), df['longitude'].to_numpy(), ZOOM_START, HEAT_RADIUS)
heat_points = np.column_stack([df['latitude'], df['longitude'], weights]).tolist()

HeatMap(
    heat_points,
    radius=HEAT_RADIUS,
    blur=HEAT_BLUR,
    min_opacity=0.3,
    gradient={0.3: 'blue', 0.5: 'cyan', 0.7: 'lime', 0.85: 'yellow', 1.0: 'red'},
).add_to(heat)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=1,
        color='red',
        fill=True,
        popup=(
            f"Brightness: {row['brightness']}<br>"
            f"Confidence: {row['confidence']}%<br>"
            f"Date: {row['acq_date']}<br>"
            f"FRP: {row['frp']}"
        )
    ).add_to(dots)

folium.LayerControl(collapsed=False).add_to(m)

# Save and open fires.html in your browser to see the map
print("Saving fires.html...")
m.save("fires.html")
print(f"Done! Map saved with {len(df)} fire markers. Open fires.html in your browser.") 
