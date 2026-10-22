# SkyAlert Integration

**Flow:** user input → Fetch → Cleaning → Map → Dashboard

> Function names and columns below are proposed. Confirm them with each owner before integrating.

## Fetch
The component that talks to NASA FIRMS. You give it a region (coordinates) and how many days back to look. It calls the FIRMS API and returns the raw fire detections as a pandas table. Each row is one satellite hotspot, with columns like latitude, longitude, acquisition date and time, brightness, confidence, and fire radiative power. If the API fails or returns old or incorrect data, it raises an error, since the data cannot be used properly.

```python
def fetch_fires(bbox: tuple[float, float, float, float], days: int = 1) -> pd.DataFrame:
    """bbox = (west, south, east, north). Returns raw FIRMS detections. Raises on failure."""
```

## Cleaning
Takes the raw table and makes it safe for everything downstream to use. It:
- drops rows with missing or incorrect coordinates
- removes duplicates
- merges the separate date and time columns into one timestamp
- converts the confidence values into one consistent format
- makes sure the numeric columns are actually numbers

It returns a new, tidy table so the map and dashboard always work with the same, consistent data.

```python
def clean_viirs(raw_df):
    """Takes the raw VIIRS table from fetch and returns a cleaned copy."""
```

## Map
Takes the clean table and returns an interactive map with one marker per detection. Color or size can show the confidence level or strength of the detection, and clicking a marker shows its time, power, and other details. It also handles the case where there are no detections. Later, this is where the ResNet flags and U-Net damage masks get layered on top.

```python
def build_map(df: pd.DataFrame) -> folium.Map:
    """One marker per detection. Returns a default map if df is empty."""
```

## Dashboard
The only piece that knows about the others. It is the frontend where a user picks a region and date range, then runs fetch, cleaning, and the map in order. It also handles errors: a message is displayed if fetch fails, and a "no detections found" message is displayed if the table is empty. It caches results so Streamlit doesn't re-call the NASA API every time someone moves a slider.

```python
@st.cache_data(ttl=600)
def load_data(bbox, days):
    return clean_viirs(fetch_fires(bbox, days))

try:
    df = load_data(bbox, days)
except Exception as e:
    st.error(f"Could not load fire data: {e}")
    st.stop()

if df.empty:
    st.info("No detections found.")
else:
    st_folium(build_map(df))
```
