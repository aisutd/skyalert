"""Fetch VIIRS active-fire detections from NASA FIRMS."""

import os
from io import StringIO

import pandas as pd
import requests
from dotenv import load_dotenv

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
# Contiguous United States: west, south, east, north (same box as main.py)
US_BOUNDS = "-125,24,-66,50"
VIIRS_SOURCES = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
# The columns FIRMS returns for VIIRS detections, in order
VIIRS_COLUMNS = [
    "latitude", "longitude", "bright_ti4", "scan", "track", "acq_date", "acq_time",
    "satellite", "instrument", "confidence", "version", "bright_ti5", "frp", "daynight",
]


class FirmsError(Exception):
    """Raised when FIRMS data can't be fetched."""


def get_map_key():
    """Read the FIRMS map key from the .env file or the environment."""
    load_dotenv()
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not key:
        raise FirmsError(
            "FIRMS_MAP_KEY is not set. Copy .env.example to .env and put your key in it."
        )
    return key


def build_url(map_key, area=US_BOUNDS, days=2, source="VIIRS_SNPP_NRT"):
    """Build the FIRMS area API URL for a VIIRS request."""
    if source not in VIIRS_SOURCES:
        raise FirmsError(
            f"Unknown source {source!r}. Use one of: {', '.join(VIIRS_SOURCES)}"
        )
    if days not in (1, 2, 3, 4, 5):
        raise FirmsError(f"days must be a whole number from 1 to 5, got {days!r}")
    return f"{FIRMS_AREA_URL}/{map_key}/{source}/{area}/{int(days)}"


def fetch_viirs(area=US_BOUNDS, days=2, source="VIIRS_SNPP_NRT", timeout=60, map_key=None):
    """Fetch VIIRS detections for an area and return them as a DataFrame.

    area is "west,south,east,north" in degrees, or "world". days is 1 to 5.
    Returns an empty DataFrame with the usual VIIRS columns when FIRMS has no
    detections. Raises FirmsError if the key is missing or rejected, the
    request fails, or FIRMS answers with something other than fire data.
    """
    if map_key is None:
        map_key = get_map_key()
    url = build_url(map_key, area, days, source)

    # These messages never include the URL, because the URL contains the key.
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
        return pd.DataFrame(columns=VIIRS_COLUMNS)
    if not text.startswith("latitude"):
        # FIRMS reports problems such as an invalid key as plain text
        raise FirmsError(f"FIRMS did not return fire data: {message}")

    try:
        return pd.read_csv(StringIO(text))
    except pd.errors.ParserError:
        raise FirmsError("FIRMS returned fire data that could not be read as CSV.") from None


if __name__ == "__main__":
    fires = fetch_viirs()
    print(f"Fetched {len(fires)} detections")
    print(fires.head())

    # A patch of open ocean has no fires, so this should come back empty
    empty = fetch_viirs(area="-150,-60,-149,-59", days=1)
    print(f"Open ocean: {len(empty)} detections, {len(empty.columns)} columns")