# SkyAlert APIs

Everything the project calls, and everything it exposes.

## External APIs (data we pull in)

| API | Used for | Key needed? | Limits / notes | Where in code |
|---|---|---|---|---|
| **NASA FIRMS** `firms.modaps.eosdis.nasa.gov/api/area/csv/` | Satellite fire detections (VIIRS_SNPP_NRT): location, brightness, FRP, confidence, date and time | Yes, `FIRMS_MAP_KEY` in `.env` | Free key. Too many requests returns plain-text errors instead of CSV | `src/firms_client.py` → `fetch_viirs()` (called by `main.py` → `fetch_fires()`) |
| **Open-Meteo** `api.open-meteo.com/v1/forecast` | Current temperature, humidity and wind; the 24 h hourly forecast; past week's rain. Feeds the popup's weather tiles and the risk level | No | Free for non-commercial use. Modeled forecast data, not a station at the fire | `fire_context.py` → `fetch_weather()` |
| **OpenStreetMap Nominatim** `nominatim.openstreetmap.org/reverse` | Place name from a lat/lon, e.g. "Poinsett County, Arkansas" (popup title) | No | Max 1 request per second, and must send a `User-Agent`. Both are handled in `reverse_geocode()` | `fire_context.py` → `reverse_geocode()` |

## Map tiles and fonts (loaded by the browser)

| Service | Used for | Where in code |
|---|---|---|
| **Esri World Street Map** `server.arcgisonline.com` | Light map background | `frontend/src/components/FireMap.jsx` → `TILES` |
| **Stadia Maps, Alidade Smooth Dark** `tiles.stadiamaps.com` | Dark map background | `frontend/src/components/FireMap.jsx` → `TILES` |
| **Google Fonts** `fonts.googleapis.com` | Archivo (all text) and JetBrains Mono (numbers in the popup) | `frontend/index.html` |

## Our own API (FastAPI, `dashboard.py`, port 8000)

The API is data only (JSON); all UI is in `frontend/src`. The React app reaches these through the Vite proxy (`/api` → `127.0.0.1:8000`).

| Endpoint | Returns | Used by |
|---|---|---|
| `GET /api/summary` | Detection count, region, date range | The 3 stat cards (`StatCards.jsx`) |
| `GET /api/fires` | Every fire detection as JSON (cleaned, with ready-to-show `confidence_label` and `acq_time_utc`) | Loaded once by `App.jsx`: the map's dots and heatmap, the popup, and the Analytics table |
| `GET /api/fire-context?lat=&lon=` | Place name, weather and risk level for one spot | The popup, on each dot click |

## Caching

- Fire data: 10 minutes (`dashboard.py`).
- Popup data (weather + place): 30 minutes, per ~1 km (`fire_context.py`).

## Credits

The on-map credit strip has been removed. The data and map providers above ask for credit,
so keep them credited here (and in any public release):
NASA FIRMS, Open-Meteo, © OpenStreetMap contributors (Nominatim and map data),
Esri, Stadia Maps and OpenMapTiles.
