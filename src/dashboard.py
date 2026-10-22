import threading
import time

from fastapi import FastAPI, Query
from fastapi.responses import Response

from fire_context import get_fire_context
from main import CONFIDENCE_LABELS, DAY_RANGE, FIRMS_SOURCE, REGION_NAME, US_BOUNDS, fetch_fires

# FIRMS only updates every few hours, so reuse the data for a while instead of
# downloading it again on every page load
CACHE_SECONDS = 10 * 60

_cache = {"df": None, "fetched_at": 0.0}
# Stops two requests that arrive at the same time from both downloading the data
_cache_lock = threading.Lock()


def get_data():
    """Return the cached, cleaned fire data, refreshing it when it's too old."""
    with _cache_lock:
        if _cache["df"] is None or time.time() - _cache["fetched_at"] > CACHE_SECONDS:
            _cache["df"] = fetch_fires()
            _cache["fetched_at"] = time.time()
        return _cache


app = FastAPI(title="SkyAlert")


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


# Columns sent to the frontend (FIRMS has a few more, like scan/track,
# that aren't useful to show)
TABLE_COLUMNS = [
    "latitude", "longitude", "bright_ti4", "bright_ti5", "frp",
    "confidence", "acq_date", "acq_time", "daynight", "satellite",
]


# OUR API: /api/fires. Every fire detection as JSON. Feeds the map's dots and heatmap, the popup and the Analytics table
@app.get("/api/fires")
def fires():
    """Every fire detection, one object per row."""
    df = get_data()["df"]
    # The row number becomes an "id" so React can tell rows apart
    table = df[TABLE_COLUMNS].rename_axis("id").reset_index()
    # Ready-to-show values, so the popup and the Analytics table don't each work them out
    table["confidence_label"] = table["confidence"].map(CONFIDENCE_LABELS).fillna(table["confidence"])
    hhmm = table["acq_time"].astype(int).astype(str).str.zfill(4)  # FIRMS acq_time: 815 -> "0815"
    table["acq_time_utc"] = hhmm.str[:2] + ":" + hhmm.str[2:]
    # to_json turns numpy numbers into plain JSON numbers (and missing values into null)
    return Response(table.to_json(orient="records"), media_type="application/json")


# OUR API: /api/fire-context?lat=&lon=. Feeds the popup: place name (Nominatim), weather (Open-Meteo) and the risk level. Called when a fire dot is clicked
@app.get("/api/fire-context")
def fire_context(lat: float = Query(ge=-90, le=90), lon: float = Query(ge=-180, le=180)):
    """Place name, weather and fire-weather risk for one spot, for the map's popup."""
    return get_fire_context(lat, lon)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.dashboard:app", host="127.0.0.1", port=8000, reload=True)
