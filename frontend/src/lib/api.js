// Every call from the frontend to the backend lives here.
//
// The backend is the FastAPI server in backend/dashboard.py (run: python -m backend.dashboard),
// listening on http://127.0.0.1:8000. The frontend asks for "/api/...", and the Vite dev
// server forwards it there (see the proxy in vite.config.js), so no port appears below.
//
//   frontend function    ->  FastAPI endpoint (backend/dashboard.py)  ->  Python that makes the data
//   getSummary()         ->  GET /api/summary                 ->  main.fetch_fires()
//   getFires()           ->  GET /api/fires                   ->  main.fetch_fires()
//                                                                 = src/firms_client.fetch_viirs()
//                                                                   + src/data_cleaning.clean_viirs()
//   getFireContext()     ->  GET /api/fire-context            ->  backend/fire_context.get_fire_context()

const API_BASE = '/api' // forwarded to the FastAPI server by vite.config.js

// GET one FastAPI endpoint and return its JSON; throws if the server answers with an error
async function getJson(path) {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) throw new Error(`Server returned ${res.status}`)
  return res.json()
}

// Numbers for the 3 stat cards: detection count, region, date range
export function getSummary() {
  return getJson('/summary')
}

// Every cleaned fire detection (one object per fire), for the map, popup and Analytics table
export function getFires() {
  return getJson('/fires')
}

// Place name, weather and fire-weather risk for one spot, for the popup
export function getFireContext(lat, lon) {
  return getJson(`/fire-context?lat=${lat}&lon=${lon}`)
}
