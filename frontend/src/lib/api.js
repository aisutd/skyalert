// Every call from the frontend to the backend lives here.
//
// The backend is the FastAPI server in backend/dashboard.py (run: python -m backend.dashboard),
// on http://127.0.0.1:8000. The Vite dev server forwards "/api/..." there (vite.config.js), so no port appears below.
//
//   frontend function      ->  FastAPI endpoint            ->  Python that makes the data
//   getFires()             ->  GET /api/fires              ->  main.fetch_fires() (NASA FIRMS, cleaned)
//   getModelStatus()       ->  GET /api/model-status       ->  backend/spread_forecast.model_status()
//   getFireContext(lat,lon)->  GET /api/fire-context       ->  backend/fire_context.get_fire_context() (Open-Meteo + OSM)
//   getForecast(lat,lon,day)-> GET /api/spread-forecast    ->  backend/spread_forecast.forecast() (models/spread_v2)

async function getJSON(url, ms) {                                 // GET with a timeout; throws on an error status
  const c = new AbortController(), t = setTimeout(() => c.abort(), ms)
  try {
    const r = await fetch(url, { signal: c.signal })
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
    return await r.json()
  } finally { clearTimeout(t) }
}

export const getFires = () => getJSON('/api/fires', 90000)
export const getModelStatus = () => getJSON('/api/model-status', 20000).catch(() => ({ ready: false }))

const ctxCache = new Map(), fcCache = new Map()                   // one request per ~1 km spot, shared by everything that asks
const spot = (lat, lon) => `${lat.toFixed(2)},${lon.toFixed(2)}`

export function getFireContext(lat, lon) {
  const k = spot(lat, lon)
  if (!ctxCache.has(k)) ctxCache.set(k, getJSON(`/api/fire-context?lat=${lat.toFixed(4)}&lon=${lon.toFixed(4)}`, 30000)
    .catch((err) => { ctxCache.delete(k); return { location: null, weather: null, reasons: [], errors: [String(err)] } }))
  return ctxCache.get(k)
}

// The spread model's next-day forecast for the 64 × 64 pixel area around a spot. day ("YYYY-MM-DD",
// optional) starts from a past day; the answer then also has `actual` (what burned) and `check` (its scores).
// Past examples pass their exact tile centre, so lat/lon keep full precision there.
export function getForecast(lat, lon, day = null) {
  const k = day ? `${lat},${lon},${day}` : spot(lat, lon)
  const where = day ? `lat=${lat}&lon=${lon}&day=${day}` : `lat=${lat.toFixed(4)}&lon=${lon.toFixed(4)}`
  if (!fcCache.has(k)) fcCache.set(k, getJSON(`/api/spread-forecast?${where}`, 90000)
    .catch((err) => { fcCache.delete(k); return { status: 'inputs_unavailable', note: String(err) } }))
  return fcCache.get(k)
}
