# SkyAlert APIs

Everything the project calls, and everything it exposes.

## External APIs (data we pull in)

| API | Used for | Key needed? | Limits / notes | Where in code |
|---|---|---|---|---|
| **NASA FIRMS** `firms.modaps.eosdis.nasa.gov/api/area/csv/` | Satellite fire detections (VIIRS_NOAA20_NRT; SNPP stopped after 2026-11-02): location, brightness, FRP, confidence, date and time | Yes, `FIRMS_MAP_KEY` in `.env` | Free key. Too many requests returns plain-text errors instead of CSV | `backend/firms_client.py` → `fetch_viirs()` (called by `main.py` → `fetch_fires()`) |
| **Open-Meteo** `api.open-meteo.com/v1/forecast` | Current temperature, humidity and wind; the 24 h hourly forecast; past week's rain. Feeds the weather tiles and the fire-weather index | No | Free for non-commercial use. Modeled forecast data, not a station at the fire | `backend/fire_context.py` → `fetch_weather()` |
| **GRIDMET** `thredds.northwestknowledge.net` | Daily wind, temperature, humidity, rain, fuel dryness (ERC) and drought index: the spread model's weather inputs | No | Usually 1–2 days behind. Sometimes unreachable; the forecast then says its inputs are unavailable | `backend/spread_forecast.py` → `_gridmet()` |
| **AWS terrain tiles** `s3.amazonaws.com/elevation-tiles-prod` | Elevation for the spread model | No | Cached in `data/cache/elevation/` | `backend/spread_forecast.py` → `_elevation()` |
| **OpenStreetMap Nominatim** `nominatim.openstreetmap.org/reverse` | Place name from a lat/lon, e.g. "Poinsett County, Arkansas" (each fire's name) | No | Max 1 request per second, and must send a `User-Agent`. Both are handled in `reverse_geocode()` | `backend/fire_context.py` → `reverse_geocode()` |

## Map tiles and fonts (loaded by the browser)

| Service | Used for | Where in code |
|---|---|---|
| **Esri World Street Map** `server.arcgisonline.com` | Light map background | `frontend/src/components/bits.jsx` → `TILES` |
| **Esri World Dark Gray Base** `server.arcgisonline.com` | Dark map background | `frontend/src/components/bits.jsx` → `TILES` |
| **Google Fonts** `fonts.googleapis.com` | Archivo (text and numbers), Space Grotesk (the SkyAlert wordmark), IBM Plex Mono (coordinates) | `frontend/index.html` |

## Our own API (FastAPI, `backend/dashboard.py`, port 8000)

The API is data only (JSON); all UI is in `frontend/src`. The React app reaches these through the Vite proxy (`/api` → `127.0.0.1:8000`).

| Endpoint | Returns | Used by |
|---|---|---|
| `GET /api/summary` | Detection count, region, date range | Not used by the UI right now |
| `GET /api/fires` | Every fire detection as JSON (cleaned, with ready-to-show `confidence_label` and `acq_time_utc`) | Loaded once by `App.jsx`: the map's fire dots, and the fire list (nearby detections grouped, `lib/live.js`) |
| `GET /api/fire-context?lat=&lon=` | Place name, weather and risk level for one spot | Each fire in the list, and any spot clicked on the map |
| `GET /api/model-status` | Whether the spread model's weights are in `data/training/` and it can run | `frontend/src/lib/api.js`, `prototype/index.html` |
| `GET /api/spread-forecast?lat=&lon=` | The spread model's chance of fire tomorrow per 1 km square, 64 × 64 km around a spot, plus the input dates | `frontend/src/lib/api.js`, `prototype/index.html` |

## Caching

- Fire data: 10 minutes (`backend/dashboard.py`).
- Popup data (weather + place): 30 minutes, per ~1 km (`backend/fire_context.py`).

## Credits

The on-map credit strip has been removed. The data and map providers above ask for credit,
so keep them credited here (and in any public release):
NASA FIRMS, Open-Meteo, © OpenStreetMap contributors (Nominatim and map data),
Esri, Stadia Maps and OpenMapTiles.
