"""Fetch active-fire detections from NASA FIRMS (VIIRS and MODIS, near-real-time and archive)."""

import os
import threading
import time
from datetime import date as Date
from io import StringIO

import pandas as pd
import requests
from dotenv import load_dotenv

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_AVAILABILITY_URL = "https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv"
# Contiguous United States: west, south, east, north (same box as main.py)
US_BOUNDS = "-125,24,-66,50"
# Near-real-time VIIRS sources, used by the dashboard through fetch_viirs()
VIIRS_SOURCES = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
# Every source fetch_area() accepts. NRT = near real time (recent days), SP = standard
# processing (the archive). Date ranges per source come from data_availability().
AREA_SOURCES = (
    "MODIS_NRT", "MODIS_SP",
    "VIIRS_SNPP_NRT", "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_NRT", "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_NRT", "VIIRS_NOAA21_SP",
)
# The columns FIRMS returns for VIIRS detections, in order (SP sources add "type")
VIIRS_COLUMNS = [
    "latitude", "longitude", "bright_ti4", "scan", "track", "acq_date", "acq_time",
    "satellite", "instrument", "confidence", "version", "bright_ti5", "frp", "daynight",
]
# The columns FIRMS returns for MODIS detections, in order (SP sources add "type").
# Seen in a MODIS_SP response on 2026-11-11 (research/data_sources.md, B.2).
MODIS_COLUMNS = [
    "latitude", "longitude", "brightness", "scan", "track", "acq_date", "acq_time",
    "satellite", "instrument", "confidence", "version", "bright_t31", "frp", "daynight",
]
# FIRMS accepts 1 to 5 days per request; 0 silently returns nothing
DAY_RANGES = (1, 2, 3, 4, 5)
AVAILABILITY_CACHE_SECONDS = 6 * 60 * 60

_availability = {"table": None, "fetched_at": 0.0}
_availability_lock = threading.Lock()


class FirmsError(Exception):
    """Raised when FIRMS data can't be fetched."""

# get env stuff
def get_map_key():
    """Read the FIRMS map key from the .env file or the environment."""
    load_dotenv()
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not key:
        raise FirmsError(
            "FIRMS_MAP_KEY is not set. Copy .env.example to .env and put your key in it."
        )
    return key


def build_url(map_key, area=US_BOUNDS, days=2, source="VIIRS_NOAA20_NRT"):
    """Build the FIRMS area API URL for a near-real-time VIIRS request."""
    if source not in VIIRS_SOURCES:
        raise FirmsError(
            f"Unknown source {source!r}. Use one of: {', '.join(VIIRS_SOURCES)}"
        )
    if days not in DAY_RANGES:
        raise FirmsError(f"days must be a whole number from 1 to 5, got {days!r}")
    return f"{FIRMS_AREA_URL}/{map_key}/{source}/{area}/{int(days)}"


def _date_text(value):
    """A date or "YYYY-MM-DD" text as "YYYY-MM-DD", or FirmsError if it isn't one."""
    if isinstance(value, Date):
        return value.isoformat()
    try:
        return Date.fromisoformat(str(value)).isoformat()
    except ValueError:
        raise FirmsError(f"date must be a date or YYYY-MM-DD text, got {value!r}") from None


def build_area_url(map_key, source, area=US_BOUNDS, days=1, date=None):
    """Build the FIRMS area API URL for any source, optionally from a start date.

    With a date, FIRMS returns `days` days counting forward from that date (UTC).
    Without one, it returns the last `days` days up to today.
    """
    if source not in AREA_SOURCES:
        raise FirmsError(f"Unknown source {source!r}. Use one of: {', '.join(AREA_SOURCES)}")
    if days not in DAY_RANGES:
        raise FirmsError(f"days must be a whole number from 1 to 5, got {days!r}")
    url = f"{FIRMS_AREA_URL}/{map_key}/{source}/{area}/{int(days)}"
    if date is not None:
        url += f"/{_date_text(date)}"
    return url


def _get_csv(url, map_key, timeout, empty_columns, expect_header="latitude"):
    """GET a FIRMS CSV and return it as a DataFrame. Messages never include the URL (it holds the key)."""
    try:
        response = requests.get(url, timeout=timeout)
    except requests.exceptions.Timeout:
        raise FirmsError(f"FIRMS did not respond within {timeout} seconds.") from None
    except requests.exceptions.ConnectionError:
        raise FirmsError("Could not connect to FIRMS. Check your internet connection.") from None
    except requests.exceptions.RequestException as error:
        raise FirmsError(f"The request to FIRMS failed ({type(error).__name__}).") from None

    text = response.text.strip()
    message = text[:200].replace(map_key, "<key>")
    if response.status_code != 200:
        raise FirmsError(f"FIRMS returned HTTP {response.status_code}: {message}")
    if not text:
        return pd.DataFrame(columns=empty_columns)
    if not text.startswith(expect_header):
        # FIRMS reports problems such as an invalid key as plain text
        raise FirmsError(f"FIRMS did not return fire data: {message}")

    try:
        return pd.read_csv(StringIO(text))
    except pd.errors.ParserError:
        raise FirmsError("FIRMS returned fire data that could not be read as CSV.") from None


def fetch_viirs(area=US_BOUNDS, days=2, source="VIIRS_NOAA20_NRT", timeout=60, map_key=None):
    """Fetch VIIRS detections for an area and return them as a DataFrame.

    area is "west,south,east,north" in degrees, or "world". days is 1 to 5.
    Returns an empty DataFrame with the usual VIIRS columns when FIRMS has no
    detections. Raises FirmsError if the key is missing or rejected, the
    request fails, or FIRMS answers with something other than fire data.
    """
    if map_key is None:
        map_key = get_map_key()
    url = build_url(map_key, area, days, source)
    return _get_csv(url, map_key, timeout, VIIRS_COLUMNS)


def fetch_area(source, area=US_BOUNDS, days=1, date=None, timeout=60, map_key=None):
    """Fetch detections from any FIRMS source (MODIS or VIIRS, NRT or SP) as a DataFrame.

    area is "west,south,east,north" in degrees. days is 1 to 5. date (a date or
    "YYYY-MM-DD") makes FIRMS return `days` days counting forward from it; without
    it, the last `days` days. Columns are whatever FIRMS sends for that source
    (MODIS has brightness/bright_t31, VIIRS has bright_ti4/bright_ti5, SP adds type).
    Returns an empty DataFrame with the usual columns when there are no detections.
    Raises FirmsError like fetch_viirs.
    """
    if map_key is None:
        map_key = get_map_key()
    url = build_area_url(map_key, source, area, days, date)
    empty_columns = MODIS_COLUMNS if source.startswith("MODIS") else VIIRS_COLUMNS
    return _get_csv(url, map_key, timeout, empty_columns)


def data_availability(map_key=None, timeout=30):
    """Date range of every FIRMS source, as a DataFrame with data_id, min_date, max_date (dates).

    FIRMS answers from /api/data_availability/csv/{KEY}/all. Kept for 6 hours, since the
    ranges move at most once a day.
    """
    with _availability_lock:
        cached = _availability["table"]
        if cached is not None and time.time() - _availability["fetched_at"] < AVAILABILITY_CACHE_SECONDS:
            return cached.copy()
        if map_key is None:
            map_key = get_map_key()
        table = _get_csv(
            f"{FIRMS_AVAILABILITY_URL}/{map_key}/all", map_key, timeout,
            ["data_id", "min_date", "max_date"], expect_header="data_id",
        )
        for column in ("min_date", "max_date"):
            table[column] = pd.to_datetime(table[column], errors="coerce").dt.date
        _availability["table"] = table
        _availability["fetched_at"] = time.time()
        return table.copy()


if __name__ == "__main__":
    fires = fetch_viirs()
    print(f"Fetched {len(fires)} detections")
    print(fires.head())

    # A patch of open ocean has no fires, so this should come back empty
    empty = fetch_viirs(area="-150,-60,-149,-59", days=1)
    print(f"Open ocean: {len(empty)} detections, {len(empty.columns)} columns")
