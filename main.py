import pandas as pd

from backend.data_cleaning import clean_viirs
from backend.firms_client import US_BOUNDS, FirmsError, fetch_viirs

# US_BOUNDS (the box we fetch) comes from backend/firms_client.py
REGION_NAME = "Contiguous United States"
# Both VIIRS satellites that are currently sending data. Suomi NPP (VIIRS_SNPP_NRT)
# returned no US detections at all from 2026-11-07 to 2026-11-11, so it is left out.
FIRMS_SOURCES = ("VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
DAY_RANGE = 2

# VIIRS gives confidence as one letter; these are the words shown to people
CONFIDENCE_LABELS = {"l": "Low", "n": "Nominal", "h": "High"}


def fetch_fires():
    """Download the latest US fire detections from every source in FIRMS_SOURCES and clean them.

    A source that fails is skipped, so one satellite's outage doesn't empty the map.
    Raises FirmsError only if every source fails.
    """
    frames, errors = [], []
    for source in FIRMS_SOURCES:
        # EXTERNAL API: NASA FIRMS, called through backend/firms_client.py (needs FIRMS_MAP_KEY in .env)
        try:
            frames.append(fetch_viirs(area=US_BOUNDS, days=DAY_RANGE, source=source))
        except FirmsError as error:
            errors.append(f"{source}: {error}")
    if not frames:
        raise FirmsError("Every FIRMS source failed. " + " ".join(errors))
    for message in errors:
        print(f"Skipped a FIRMS source. {message}")
    # Leave out empty frames, so concat keeps the real column types
    non_empty = [frame for frame in frames if len(frame)]
    raw = pd.concat(non_empty, ignore_index=True) if non_empty else frames[0]
    # backend/data_cleaning.py drops bad coordinates and duplicate detections
    return clean_viirs(raw)


# Only runs when you do `python main.py`, not when backend/dashboard.py imports this file
if __name__ == "__main__":
    df = fetch_fires()
    print(f"Found {len(df)} active fires")
    print(df["satellite"].value_counts().to_string())
    print(df[["latitude", "longitude", "bright_ti4", "confidence", "acq_date", "frp"]].head())
