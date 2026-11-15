// Turns the latest FIRMS detections into the list of fires the UI shows: cluster nearby detections in the
// area on the map, look up each fire's place + weather, and the spread model's chance of fire on the next day.
import L from 'leaflet'
import { getFireContext, getForecast } from './api.js'
import { RISK, compass } from './format.js'

export const US = [[24.4, -125.0], [49.5, -66.9]]                // first view: the contiguous US (the fires and the model cover it all)
const LEVEL = { low: 1, moderate: 2, high: 3, critical: 4 }
export const CLUSTER_KM = 15                                      // detections this close count as one fire
const CANDIDATES = 8                                              // the strongest 8 fires in the area get weather and a forecast

export const kmBetween = (a, b) => Math.hypot((a[0] - b[0]) * 111.32, (a[1] - b[1]) * 111.32 * Math.cos((a[0] * Math.PI) / 180))

export function clustersIn(fires, bounds) {                       // strongest detection first; nearby ones join it
  const b = L.latLngBounds(bounds)
  const inside = fires.filter((f) => b.contains([f.latitude, f.longitude])).sort((a, z) => z.frp - a.frp)
  const out = []
  for (const f of inside) {
    const p = [f.latitude, f.longitude], seen = `${f.acq_date} ${f.acq_time_utc}`
    const c = out.find((x) => kmBetween(x.latlng, p) < CLUSTER_KM)
    if (c) { c.count += 1; if (seen > c.last) c.last = seen }
    else if (out.length < CANDIDATES) out.push({ latlng: p, frp: f.frp, count: 1, last: seen })
  }
  return { clusters: out, total: inside.length }
}

export function clusterAround(fires, f) {                         // one clicked fire dot plus the detections near it (from the prototype)
  const p = [f.latitude, f.longitude], near = fires.filter((x) => kmBetween(p, [x.latitude, x.longitude]) < CLUSTER_KM)
  return { latlng: p, frp: Math.max(...near.map((x) => x.frp ?? 0)), count: near.length, last: near.map((x) => `${x.acq_date} ${x.acq_time_utc}`).sort().pop(), fire: true }
}

const STATES = { Alabama: 'AL', Alaska: 'AK', Arizona: 'AZ', Arkansas: 'AR', California: 'CA', Colorado: 'CO', Connecticut: 'CT', Delaware: 'DE', Florida: 'FL', Georgia: 'GA',
  Hawaii: 'HI', Idaho: 'ID', Illinois: 'IL', Indiana: 'IN', Iowa: 'IA', Kansas: 'KS', Kentucky: 'KY', Louisiana: 'LA', Maine: 'ME', Maryland: 'MD',
  Massachusetts: 'MA', Michigan: 'MI', Minnesota: 'MN', Mississippi: 'MS', Missouri: 'MO', Montana: 'MT', Nebraska: 'NE', Nevada: 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
  'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', Ohio: 'OH', Oklahoma: 'OK', Oregon: 'OR', Pennsylvania: 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
  'South Dakota': 'SD', Tennessee: 'TN', Texas: 'TX', Utah: 'UT', Vermont: 'VT', Virginia: 'VA', Washington: 'WA', 'West Virginia': 'WV', Wisconsin: 'WI', Wyoming: 'WY', 'District of Columbia': 'DC' }
function placeName(loc) {                                         // "Madera County, California" -> "Madera County, CA"
  if (!loc || !loc.name) return null
  if (loc.country !== 'United States') return loc.name
  const parts = loc.name.split(', '), last = parts[parts.length - 1]
  if (STATES[last]) parts[parts.length - 1] = STATES[last]
  return parts.join(', ')
}

function toItem(c, ctx, picked) {
  const r = ctx.weather && ctx.weather.risk
  return {
    id: `${c.latlng[0].toFixed(3)}, ${c.latlng[1].toFixed(3)}`,
    name: placeName(ctx.location) || (picked ? 'Picked spot' : 'Unknown area'),
    risk: r ? LEVEL[r.key] || 1 : 1, prob: r ? Math.round(r.index) : 0, fwi: r ? Math.round(r.index) : 0, conf: 0,
    metric: picked || c.frp == null ? '—' : `${c.frp.toFixed(1)} MW`,
    evidence: picked ? 'No detection here' : `${c.count} detection${c.count === 1 ? '' : 's'}`,
    window: picked ? (c.clickedSpot ? 'Weather for the spot you clicked' : 'No fires detected in view: weather at the middle of the map') : `Last detected ${c.last} UTC`,
    latlng: c.latlng, ctx, reasons: ctx.reasons || [], picked: !!picked,
  }
}

const chanceLevel = (p) => (p >= 62 ? 4 : p >= 40 ? 3 : p >= 20 ? 2 : 1)   // calibrated chance of fire within 2 km (models/spread_v2/calibration.json)
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
export const shortDay = (iso) => { const [, m, d] = iso.split('-'); return `${MONTHS[+m - 1]} ${+d}` }   // "2026-11-14" -> "Nov 14"
async function applyForecast(it) {                                // the model's chance replaces the fire-weather index
  it.fc = await getForecast(...it.latlng)
  if (it.fc.status !== 'ok') return
  it.prob = Math.round(it.fc.chance_near * 100)
  it.risk = chanceLevel(it.prob)
}

function measOf(it) {                                             // the four weather tiles for one fire
  const w = it && it.ctx.weather
  if (!w) return ['Temperature', 'Wind', 'Humidity', 'Rain, past week'].map((l) => ({ l, v: '—', u: '', s: 'Weather unavailable' }))
  const c = w.current, x = w.window, r = w.rain_past_week_in
  return [
    { l: 'Temperature', v: `${Math.round(c.temperature_f)}`, u: '°F', s: `Up to ${Math.round(x.max_temp_f)}°F in the next 24 h` },
    { l: 'Wind', v: `${Math.round(c.wind_mph)}`, u: 'mph', s: `From ${compass(c.wind_direction)}${x.max_gust_mph != null ? `, gusts ${Math.round(x.max_gust_mph)}` : ''}` },
    { l: 'Humidity', v: `${Math.round(c.humidity)}`, u: '%', s: `Drops to ${Math.round(x.min_humidity)}% in the next 24 h` },
    { l: 'Rain, past week', v: r.toFixed(2), u: 'in', s: r < 0.1 ? 'Very dry' : r >= 0.5 ? 'Wet' : 'Some rain' },
  ]
}
function seriesOf(it) {                                           // hour-by-hour fire-weather index, 24 h back and 24 h ahead
  const h = (it && it.ctx.weather && it.ctx.weather.series) || []
  const obs = h.filter((x) => x.past).map((x) => x.ffwi), pred = h.filter((x) => !x.past).map((x) => x.ffwi)
  return { name: 'Fire-weather index at this spot', max: 100, threshold: 50, live: true, obs: obs.length ? obs : pred.slice(0, 1), pred }
}
export const measFor = (D, sel) => (D.live ? measOf(D.items[sel]) : D.meas)
export const seriesFor = (D, sel) => (D.live ? seriesOf(D.items[sel]) : D.series)

// the sentence for one fire, shown in the right box above its details
export function selHeadline(D, it) {
  return D.modelReady ? `Chance of fire on ${D.forecastDay}: ${it.prob}% near ${it.name}` : `Fire weather is ${RISK[it.risk].toLowerCase()} near ${it.name}`
}

function liveData(items, total, modelStatus) {
  const top = items.reduce((a, b) => (b.prob > a.prob ? b : a), items[0])
  const anyFire = items.some((x) => !x.picked)
  const ran = items.find((x) => x.fc && x.fc.status === 'ok'), m = modelStatus.ready && !!ran   // the model ran on the latest data
  // The forecast is for the day after the latest complete day of satellite data: usually today, so say the date
  const forecastDay = ran ? shortDay(ran.fc.forecast_day) : null
  return {
    key: 'fire', live: true, modelReady: m, unit: m ? '%' : '/100', count: total, items, forecastDay,
    region: 'In view', window: m ? `For ${forecastDay}` : 'Next 24 h',
    model: m ? 'SkyAlert spread model, on NOAA-20 fires of the last 3 days and gridMET weather' : 'Fosberg fire-weather index, from Open-Meteo weather',
    headline: !anyFire ? 'No fires detected in this area in the last 2 days'
      : m ? `Highest chance of fire on ${forecastDay}: ${top.prob}% near ${top.name}` : `Fire weather is ${RISK[top.risk].toLowerCase()} near ${top.name}`,
    topName: anyFire ? top.name : null,   // the left box shows just this place; the sentence moves to the right box (selHeadline)
    risk: top.risk, prob: top.prob,
    probLabel: m ? `Chance of fire within 2 km on ${forecastDay}, from the SkyAlert spread model` : 'Fire-weather index (0–100), worst hour in the next 24 h',
    regionLabel: m ? 'highest chance' : 'highest index',
    metricName: 'Fire power (FRP)', listTitle: m ? 'Fires in view, highest chance first' : 'Fires in view, highest index first',
    layers: { observed: 'Fire detections, last 2 days', predicted: m ? `Model: chance of fire on ${forecastDay}` : modelStatus.ready ? 'Model predictions (inputs unavailable)' : 'Model predictions (model not loaded)', context: '' },
  }
}

// pick: a clicked fire (clusterAround) or spot ({ latlng, clickedSpot: true }); it joins the list, marked `clicked`
export async function buildLive({ fires, bounds, modelStatus, withModel, pick = null }) {
  const { clusters, total } = clustersIn(fires, bounds)
  let items = await Promise.all(clusters.map(async (c) => toItem(c, await getFireContext(...c.latlng))))
  if (pick) {                                                     // the spot or fire dot you clicked joins the list (as in the prototype)
    const it = toItem(pick, await getFireContext(...pick.latlng), !pick.fire)
    it.clicked = true
    items = items.filter((x) => kmBetween(x.latlng, it.latlng) >= CLUSTER_KM).concat(it)
  }
  if (!items.length) { const c = L.latLngBounds(bounds).getCenter(); items = [toItem({ latlng: [c.lat, c.lng] }, await getFireContext(c.lat, c.lng), true)] }
  if (withModel && modelStatus.ready) await Promise.all(items.map(applyForecast))
  items.sort((a, b) => b.prob - a.prob)                           // highest first
  return liveData(items, total, modelStatus)
}

const heatCache = new WeakMap()
export function heatURL(fc) {                                     // forecast grid -> see-through PNG, brighter = more likely
  if (heatCache.has(fc)) return heatCache.get(fc)
  const c = document.createElement('canvas'); c.width = c.height = 64
  const g = c.getContext('2d'), img = g.createImageData(64, 64)
  const stops = [[0, [247, 223, 166]], [0.35, [242, 193, 78]], [0.62, [245, 130, 58]], [1, [217, 87, 87]]]
  fc.probability.forEach((row, y) => row.forEach((p, x) => {
    if (p < 0.05) return
    const j = stops.findIndex(([at]) => p <= at), [a0, c0] = stops[Math.max(0, j - 1)], [a1, c1] = stops[Math.max(0, j)]
    const t = a1 === a0 ? 0 : (p - a0) / (a1 - a0), o = (y * 64 + x) * 4
    for (let k = 0; k < 3; k++) img.data[o + k] = Math.round(c0[k] + (c1[k] - c0[k]) * t)
    img.data[o + 3] = Math.round(255 * Math.min(0.85, 0.2 + 0.75 * p))
  }))
  g.putImageData(img, 0, 0)
  const url = c.toDataURL(); heatCache.set(fc, url); return url
}
