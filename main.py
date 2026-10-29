from backend.data_cleaning import clean_viirs
from backend.firms_client import US_BOUNDS, fetch_viirs

# US_BOUNDS (the box we fetch) comes from backend/firms_client.py
REGION_NAME = "Contiguous United States"
FIRMS_SOURCE = "VIIRS_SNPP_NRT"
DAY_RANGE = 2

# VIIRS gives confidence as one letter; these are the words shown to people
CONFIDENCE_LABELS = {"l": "Low", "n": "Nominal", "h": "High"}


def fetch_fires():
    """Download the latest US fire detections and clean them, using the team's backend/ files."""
    # EXTERNAL API: NASA FIRMS, called through backend/firms_client.py (needs FIRMS_MAP_KEY in .env)
    raw = fetch_viirs(area=US_BOUNDS, days=DAY_RANGE, source=FIRMS_SOURCE)
    # backend/data_cleaning.py drops bad coordinates and duplicate detections
    return clean_viirs(raw)


# Only runs when you do `python main.py`, not when dashboard.py imports this file
if __name__ == "__main__":
    df = fetch_fires()
    print(f"Found {len(df)} active fires")
    print(df[["latitude", "longitude", "bright_ti4", "confidence", "acq_date", "frp"]].head())
