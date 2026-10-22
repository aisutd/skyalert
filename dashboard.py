"""SkyAlert dashboard backend.

Serves the fire map and its summary stats to the React frontend (frontend/).
Run it with:  python dashboard.py   (then open http://localhost:5173 from the frontend)
"""
import os
import threading
import time

import folium
import xyzservices
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from fire_context import get_fire_context
from main import DAY_RANGE, FIRMS_SOURCE, REGION_NAME, US_BOUNDS, build_map, fetch_fires

# static/ holds the fire information panel's JavaScript and CSS (fire-panel.js/.css)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Styles and a small script added to the map page so the map and its controls
# (zoom buttons, dots/heatmap switch, popups) match the dashboard's light or dark theme.
# The React app picks the theme: it loads the map with ?theme=dark|light, then
# sends a message to the map whenever the user flips the toggle.
MAP_THEME_HEAD = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  /* Light theme colours (default) */
  :root {
    --ctl-bg: #ffffff;
    --ctl-text: #1b1d22;
    --ctl-muted: #6b7080;
    --ctl-border: rgba(20, 22, 28, 0.10);
    --ctl-hover: #f4f5f7;
    --ctl-shadow: 0 4px 16px rgba(20, 22, 28, 0.12);
    --accent: #f26b1d;
    /* Same look as the dashboard's stat boxes (frontend/src/index.css) */
    --card-bg: rgba(255, 255, 255, 0.95);
    --card-border: rgba(20, 22, 28, 0.08);
    --card-shadow: 0 4px 16px rgba(20, 22, 28, 0.12);
  }
  /* Dark theme colours, used when the dashboard sets data-theme="dark" */
  html[data-theme="dark"] {
    --ctl-bg: #1b1e26;
    --ctl-text: #eef0f3;
    --ctl-muted: #9aa0ad;
    --ctl-border: rgba(255, 255, 255, 0.10);
    --ctl-hover: #262a34;
    --ctl-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);
    --card-bg: rgba(27, 30, 38, 0.94);
    --card-border: rgba(255, 255, 255, 0.08);
    --card-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);
  }

  html .leaflet-container { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
  /* Colour shown behind the tiles while they load */
  html[data-theme="dark"] .leaflet-container { background: #1c1f24; }


  /* Zoom + / - buttons, the sound button and the dots/heatmap box look like the stat boxes.
     The .leaflet-touch selectors are needed to beat Leaflet's own 2px border on touch screens */
  html .leaflet-bar,
  html .leaflet-touch .leaflet-bar,
  html .leaflet-control-layers,
  html .leaflet-touch .leaflet-control-layers {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 6px;
    box-shadow: var(--card-shadow);
    backdrop-filter: blur(6px);
    overflow: hidden;
  }

  /* Zoom buttons */
  html .leaflet-bar a,
  html .leaflet-touch .leaflet-bar a {
    width: 36px; height: 36px; line-height: 36px;
    background: transparent;
    color: var(--ctl-text);
    border-bottom: 1px solid var(--card-border);
  }
  html .leaflet-bar a:last-child { border-bottom: none; }
  html .leaflet-bar a:hover { background: var(--ctl-hover); color: var(--accent); }
  html .leaflet-bar a.leaflet-disabled { background: transparent; color: var(--ctl-muted); opacity: 0.5; }

  /* Dots / heatmap switch (top right) */
  html .leaflet-control-layers-expanded { padding: 10px 14px 10px 10px; color: var(--ctl-text); }
  html .leaflet-control-layers label { font-size: 13px; padding: 3px 0; cursor: pointer; }
  html .leaflet-control-layers input { accent-color: var(--accent); margin: 0 6px 0 2px; cursor: pointer; }
  html .leaflet-control-layers-separator { border-top-color: var(--ctl-border); }

  /* Popups when you click a fire dot */
  html .leaflet-popup-content-wrapper,
  html .leaflet-popup-tip { background: var(--ctl-bg); color: var(--ctl-text); box-shadow: var(--ctl-shadow); }
  html .leaflet-popup-content-wrapper { border-radius: 10px; }
  html .leaflet-container a.leaflet-popup-close-button { color: var(--ctl-muted); }
</style>
<script>
  // Starting theme comes from the URL (?theme=dark), so there's no flash of the wrong colours
  document.documentElement.dataset.theme =
    new URLSearchParams(window.location.search).get('theme') === 'dark' ? 'dark' : 'light';

  // Later theme changes come from the dashboard as a message
  window.addEventListener('message', function (event) {
    if (event.origin !== window.location.origin) return; // ignore messages from other sites
    if (event.data && event.data.type === 'skyalert-theme') {
      var theme = event.data.theme === 'dark' ? 'dark' : 'light';
      document.documentElement.dataset.theme = theme;
      // Swap the map tiles too (set up in MAP_TILE_SWAP_SCRIPT once the map exists)
      if (window.skyalertSetMapTheme) window.skyalertSetMapTheme(theme);
    }
  });
</script>
"""

# __MAP__, __LIGHT_TILES__ and __DARK_TILES__ are replaced with the JavaScript names
# Folium gave the map and its two tile layers. Folium puts this code *before* the
# code that creates the map, so it waits for DOMContentLoaded (fired after all of
# the page's scripts have run) to make sure the map exists first.
MAP_TILE_SWAP_SCRIPT = """
document.addEventListener('DOMContentLoaded', function () {
  var map = __MAP__;
  var lightTiles = __LIGHT_TILES__;
  var darkTiles = __DARK_TILES__;

  window.skyalertSetMapTheme = function (theme) {
    var show = theme === 'dark' ? darkTiles : lightTiles;
    var hide = theme === 'dark' ? lightTiles : darkTiles;
    map.removeLayer(hide);
    show.addTo(map);
  };

  // Use whatever theme the page started with (?theme=dark in the URL)
  window.skyalertSetMapTheme(document.documentElement.dataset.theme);
});
"""


def add_theme_support(m):
    """Let the dashboard switch this map's controls and tiles between light and dark."""
    m.get_root().header.add_child(folium.Element(MAP_THEME_HEAD))

    # The Esri street map build_map() added, used in light mode
    light_tiles = next(c for c in m._children.values() if isinstance(c, folium.TileLayer))
    # Folium's built-in Stadia "Alidade Smooth Dark" map, used in dark mode.
    # (Folium's other built-in dark map, CartoDB Dark Matter, now needs an API key.)
    # show=False keeps it hidden until the theme script switches to it, and
    # control=False keeps it out of the dots/heatmap switch.
    dark_tiles = folium.TileLayer(
        xyzservices.providers.Stadia.AlidadeSmoothDark, control=False, show=False
    ).add_to(m)

    script = (
        MAP_TILE_SWAP_SCRIPT
        .replace("__MAP__", m.get_name())
        .replace("__LIGHT_TILES__", light_tiles.get_name())
        .replace("__DARK_TILES__", dark_tiles.get_name())
    )
    m.get_root().script.add_child(folium.Element(script))


# Starts the fire information panel (static/fire-panel.js) once the map exists.
# __MAP__ and __DOTS__ are the JavaScript names of the map and the "Fire dots" layer.
FIRE_PANEL_INIT_SCRIPT = """
document.addEventListener('DOMContentLoaded', function () {
  // Remove the credit strip in the corner (Leaflet / Esri / Stadia / OpenStreetMap)
  __MAP__.attributionControl.remove();
  if (window.SkyAlertFirePanel) window.SkyAlertFirePanel.init(__MAP__, __DOTS__);
});
"""


def add_fire_panel(m):
    """Replace the fire dots' plain popups with the animated information panel."""
    # __ASSET_VERSION__ is swapped for the files' last modified time on every request
    # (see fire_map), so the cached map page never points at an old copy of the files
    m.get_root().header.add_child(folium.Element(
        '<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">'
        '<link rel="stylesheet" href="/api/static/fire-panel.css?v=__ASSET_VERSION__">'
        # defer: runs after the page is parsed but before DOMContentLoaded, so it's
        # ready by the time FIRE_PANEL_INIT_SCRIPT calls it
        '<script defer src="/api/static/fire-panel.js?v=__ASSET_VERSION__"></script>'
    ))
    dots = next(
        c for c in m._children.values()
        if isinstance(c, folium.FeatureGroup) and c.layer_name == "Fire dots"
    )
    script = FIRE_PANEL_INIT_SCRIPT.replace("__MAP__", m.get_name()).replace("__DOTS__", dots.get_name())
    m.get_root().script.add_child(folium.Element(script))


# FIRMS only updates every few hours, so reuse the data for a while instead of
# downloading it again on every page load
CACHE_SECONDS = 10 * 60

_cache = {"df": None, "map_html": None, "fetched_at": 0.0}
# Stops two requests that arrive at the same time from both downloading the data
_cache_lock = threading.Lock()


def get_data():
    """Return the cached fire data and map, refreshing them when they're too old."""
    with _cache_lock:
        if _cache["df"] is None or time.time() - _cache["fetched_at"] > CACHE_SECONDS:
            df = fetch_fires()
            _cache["df"] = df
            m = build_map(df)
            add_theme_support(m)
            add_fire_panel(m)
            # render() gives the full map page as an HTML string, same as fires.html
            _cache["map_html"] = m.get_root().render()
            _cache["fetched_at"] = time.time()
        return _cache


app = FastAPI(title="SkyAlert")
# OUR API: /api/static/... serves the popup's JS and CSS (fire-panel.js, fire-panel.css) to the map page
app.mount("/api/static",StaticFiles(directory=STATIC_DIR), name="static")


# OUR API: /api/summary. Feeds the three stat cards at the top (detections, region, dates). Called by the React app (StatCards.jsx)
@app.get("/api/summary")
def summary():
    """Numbers shown in the dashboard's stat cards."""
    data = get_data()
    df = data["df"]
    west, south, east, north = (float(v) for v in US_BOUNDS.split(","))
    return {
        "detections": len(df),
        "region": {"name": REGION_NAME, "bounds": {"west": west, "south": south, "east": east, "north": north}},
        # acq_date is already "YYYY-MM-DD" text, so min/max give the earliest and latest day
        "date_range": {
            "start": df["acq_date"].min() if len(df) else None,
            "end": df["acq_date"].max() if len(df) else None,
        },
        "source": FIRMS_SOURCE,
        "day_range": DAY_RANGE,
        "fetched_at": data["fetched_at"],
    }


# Columns sent to the Analytics table (FIRMS has a few more, like scan/track,
# that aren't useful to show)
TABLE_COLUMNS = [
    "latitude", "longitude", "bright_ti4", "bright_ti5", "frp",
    "confidence", "acq_date", "acq_time", "daynight", "satellite",
]


# OUR API: /api/fires. Every fire detection as JSON. Feeds the Analytics table, and the popup's FRP, confidence and time
@app.get("/api/fires")
def fires():
    """Every fire detection, one object per row, for the Analytics table."""
    df = get_data()["df"]
    # The row number becomes an "id" so React can tell rows apart
    table = df[TABLE_COLUMNS].rename_axis("id").reset_index()
    # to_json turns numpy numbers into plain JSON numbers (and missing values into null)
    return Response(table.to_json(orient="records"), media_type="application/json")


# OUR API: /api/fire-context?lat=&lon=. Feeds the popup: place name (Nominatim), weather (Open-Meteo) and the risk level. Called when a fire dot is clicked
@app.get("/api/fire-context")
def fire_context(lat: float = Query(ge=-90, le=90), lon: float = Query(ge=-180, le=180)):
    """Place name, weather and fire-weather risk for one spot, for the map's info panel."""
    return get_fire_context(lat, lon)


# OUR API: /api/map. The full Folium map page (dots, heatmap, popup). The React app shows it in an iframe
@app.get("/api/map", response_class=HTMLResponse)
def fire_map():
    """The Folium map page; the React app shows it inside an iframe."""
    # ?v=<last modified time> makes the browser fetch the panel files again after they change
    version = int(max(os.path.getmtime(os.path.join(STATIC_DIR, f)) for f in ("fire-panel.js", "fire-panel.css")))
    return get_data()["map_html"].replace("__ASSET_VERSION__", str(version))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dashboard:app", host="127.0.0.1", port=8000, reload=True)
