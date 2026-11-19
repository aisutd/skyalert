import datetime as dt
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from backend import spread_forecast
from backend.fire_context import get_fire_context
from main import CONFIDENCE_LABELS, DAY_RANGE, FIRMS_SOURCES, REGION_NAME, US_BOUNDS, fetch_fires

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


# OUR API: /api/summary. Detection count, region and dates. Not used by the React app right now (its stat cards were replaced)
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
        "sources": list(FIRMS_SOURCES),
        "day_range": DAY_RANGE,
        "fetched_at": data["fetched_at"],
    }


# Columns sent to the frontend (FIRMS has a few more, like scan/track,
# that aren't useful to show)
TABLE_COLUMNS = [
    "latitude", "longitude", "bright_ti4", "bright_ti5", "frp",
    "confidence", "acq_date", "acq_time", "daynight", "satellite",
]


# OUR API: /api/fires. Every fire detection as JSON. Feeds the map's fire dots and the fire list (frontend/src/lib/live.js)
@app.get("/api/fires")
def fires():
    """Every fire detection, one object per row."""
    df = get_data()["df"]
    # The row number becomes an "id" so React can tell rows apart
    table = df[TABLE_COLUMNS].rename_axis("id").reset_index()
    # Ready-to-show values, so the frontend doesn't have to work them out
    table["confidence_label"] = table["confidence"].map(CONFIDENCE_LABELS).fillna(table["confidence"])
    hhmm = table["acq_time"].astype(int).astype(str).str.zfill(4)  # FIRMS acq_time: 815 -> "0815"
    table["acq_time_utc"] = hhmm.str[:2] + ":" + hhmm.str[2:]
    # to_json turns numpy numbers into plain JSON numbers (and missing values into null)
    return Response(table.to_json(orient="records"), media_type="application/json")


# OUR API: /api/fire-context?lat=&lon=. Feeds each fire in the list (and any spot clicked on the map): place name (Nominatim), weather (Open-Meteo) and the risk level. Called once per fire, cached
@app.get("/api/fire-context")
def fire_context(lat: float = Query(ge=-90, le=90), lon: float = Query(ge=-180, le=180)):
    """Place name, weather and fire-weather risk for one spot, for the fire list and weather panel."""
    return get_fire_context(lat, lon)


# OUR API: /api/model-status. Whether the spread model can run (models/spread_v2, committed to the repo)
@app.get("/api/model-status")
def spread_model_status():
    return spread_forecast.model_status()


# OUR API: /api/spread-forecast?lat=&lon=[&day=]. The spread model's chance of fire on the next day, per
# 1 km pixel, for the 64 x 64 pixel area around a spot (backend/spread_forecast.py: NOAA-20 fire detections,
# gridMET weather, AWS terrain). Always answers 200 with a "status": "ok", or "inputs_unavailable" with a
# "note" saying why (the frontend's contract, frontend/src/lib/api.js)
@app.get("/api/spread-forecast")
def spread_forecast_endpoint(
    lat: float = Query(ge=24, le=50),
    lon: float = Query(ge=-125, le=-66),
    day: dt.date | None = Query(None, description="Start from fires seen on this past UTC day (default: the latest complete day)"),
):
    """Next-day fire spread forecast for the 64 x 64 pixel tile around one spot.

    With `day`, the forecast starts from that past day and also returns what actually burned
    the next day, so the forecast can be checked (the past-fire examples).
    """
    latest = spread_forecast.latest_fire_day()
    if day is not None and not (dt.date(2018, 4, 3) <= day <= latest):
        raise HTTPException(status_code=422, detail=f"day must be between 2018-04-03 (NOAA-20 archive) and {latest.isoformat()}")
    status = spread_forecast.model_status()
    if not status["ready"]:
        return {"status": "not_ready", **status}
    try:
        return spread_forecast.forecast(lat, lon, fire_day=day)
    except spread_forecast.ForecastUnavailable as error:
        return {"status": "inputs_unavailable", "note": str(error)}


# The standalone HTML prototype (prototype/index.html). Served from here so the page
# can call /api on the same address: no CORS, and the FIRMS key stays on the server
PROTOTYPE_DIR = Path(__file__).resolve().parent.parent / "prototype"
if PROTOTYPE_DIR.is_dir():
    app.mount("/prototype", StaticFiles(directory=PROTOTYPE_DIR, html=True), name="prototype")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.dashboard:app", host="127.0.0.1", port=8000, reload=True)
