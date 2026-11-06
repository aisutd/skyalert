// Every call from the frontend to the backend lives here.
//
// The backend is the FastAPI server in backend/dashboard.py (run: python -m backend.dashboard),
// on http://127.0.0.1:8000. The Vite dev server forwards "/api/..." there (vite.config.js), so no port appears below.
//
//   frontend function      ->  FastAPI endpoint            ->  Python that makes the data
//   getFires()             ->  GET /api/fires              ->  main.fetch_fires() (NASA FIRMS, cleaned)
//   getModelStatus()       ->  GET /api/model-status       ->  backend/spread_forecast.model_status()
//   getFireContext(lat,lon)->  GET /api/fire-context       ->  backend/fire_context.get_fire_context() (Open-Meteo + OSM)
//   getForecast(lat,lon)   ->  GET /api/spread-forecast    ->  backend/spread_forecast.forecast()

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

export function getForecast(lat, lon) {                           // the spread model's 64 × 64 km forecast around a spot
  const k = spot(lat, lon)
  if (!fcCache.has(k)) fcCache.set(k, getJSON(`/api/spread-forecast?lat=${lat.toFixed(4)}&lon=${lon.toFixed(4)}`, 90000)
    .catch((err) => { fcCache.delete(k); return { status: 'inputs_unavailable', note: String(err) } }))
  return fcCache.get(k)
}
