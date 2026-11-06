// Sample data, shown for Flood (no live flood source yet) and for Fire when the backend can't be reached

export const SAMPLE = {
  fire: {
    key: 'fire', region: 'Sample region R‑12', model: 'fire‑risk v0.3', window: 'Next 48 h',
    headline: 'Fire risk is predicted to stay high through Thursday',
    risk: 3, prob: 72, conf: 81, probLabel: 'Probability of significant fire activity', regionLabel: 'region probability',
    meas: [
      { l: 'Temperature', v: '34', u: '°C', s: '6° above normal' },
      { l: 'Wind', v: '24', u: 'km/h', s: 'From NW, gusts 41' },
      { l: 'Humidity', v: '12', u: '%', s: 'Very dry' },
      { l: 'Fuel moisture', v: '8', u: '%', s: 'Critical below 10 %' },
    ],
    metricName: 'Max FRP',
    series: {
      name: 'Fire risk index', max: 100,
      obs: [38, 41, 44, 47, 52, 55, 60, 63, 66, 69, 71, 72], pred: [74, 76, 78, 80, 79, 77, 74, 71, 68, 66, 63, 61],
      spread: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], bars: [3, 5, 4, 8, 11, 9, 14, 18, 16, 21, 17, 12], threshold: 70,
    },
    items: [
      { id: 'R‑12‑D', name: 'North Slope', risk: 4, prob: 86, conf: 84, metric: '61.0 MW', evidence: '23 detections', window: 'Prediction window Oct 28 12:00 – Oct 30 12:00 UTC', pos: [300, 64] },
      { id: 'R‑12‑A', name: 'Cedar Ridge', risk: 3, prob: 72, conf: 81, metric: '48.2 MW', evidence: '14 detections', window: 'Prediction window Oct 28 12:00 – Oct 30 12:00 UTC', pos: [212, 104] },
      { id: 'R‑12‑B', name: 'Juniper Flats', risk: 2, prob: 54, conf: 76, metric: '21.6 MW', evidence: '6 detections', window: 'Prediction window Oct 28 12:00 – Oct 30 12:00 UTC', pos: [128, 168] },
      { id: 'R‑12‑C', name: 'Ash Hollow', risk: 1, prob: 22, conf: 69, metric: '6.4 MW', evidence: '1 detection', window: 'Prediction window Oct 28 12:00 – Oct 30 12:00 UTC', pos: [76, 222] },
    ],
    layers: { observed: 'Detections', predicted: 'Predicted risk', context: 'Wind and terrain' },
  },
  flood: {
    key: 'flood', region: 'Sample basin B‑04', model: 'flood‑unet v0.1', window: 'Next 48 h',
    headline: 'Flood risk is predicted to peak tomorrow along Basin B‑04',
    risk: 3, prob: 63, conf: 74, probLabel: 'Probability of inundation', regionLabel: 'basin probability',
    meas: [
      { l: 'Rain intensity', v: '14', u: 'mm/h', s: 'Peak at 05:00' },
      { l: 'Accumulated rain', v: '86', u: 'mm', s: 'Last 24 h' },
      { l: 'Water level', v: '+2.1', u: 'm', s: 'Gauge G‑07, above flood stage' },
      { l: 'Soil saturation', v: '92', u: '%', s: 'Near capacity' },
    ],
    metricName: 'Inundation',
    series: {
      name: 'Water level, gauge G‑07', max: 3,
      obs: [0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.7, 1.9, 2.0, 2.1], pred: [2.2, 2.4, 2.5, 2.6, 2.6, 2.5, 2.3, 2.1, 1.9, 1.7, 1.5, 1.4],
      spread: [0.08, 0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.31, 0.34, 0.37, 0.4], bars: [2, 4, 6, 9, 12, 14, 11, 8, 6, 5, 3, 2], threshold: 2.0,
    },
    items: [
      { id: 'B‑04‑4', name: 'Mill Creek outlet', risk: 4, prob: 81, conf: 78, metric: '5.1 km²', evidence: '86 mm rain', window: 'Prediction window Oct 29 00:00 – Oct 30 00:00 UTC', pos: [332, 206] },
      { id: 'B‑04‑1', name: 'Lower Mill basin', risk: 3, prob: 68, conf: 74, metric: '3.4 km²', evidence: 'Gauge +2.1 m', window: 'Prediction window Oct 29 00:00 – Oct 30 00:00 UTC', pos: [250, 164] },
      { id: 'B‑04‑2', name: 'Riverside lowlands', risk: 2, prob: 51, conf: 71, metric: '1.2 km²', evidence: '61 mm rain', window: 'Prediction window Oct 29 00:00 – Oct 30 00:00 UTC', pos: [146, 112] },
      { id: 'B‑04‑3', name: 'Harbor shore', risk: 1, prob: 19, conf: 66, metric: '0.2 km²', evidence: 'Tide +0.4 m', window: 'Prediction window Oct 29 00:00 – Oct 30 00:00 UTC', pos: [64, 64] },
    ],
    layers: { observed: 'Gauges and rain', predicted: 'Predicted inundation', context: 'Rivers and terrain' },
  },
}

// ---------- the sample "model output" raster, drawn in 400 × 260 sketch units ----------
export const RIVER = [[0, 40], [52, 52], [104, 80], [150, 100], [200, 130], [250, 152], [300, 180], [346, 208], [400, 238]]
const hash = (x, y) => { const s = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453; return s - Math.floor(s) }
function riverDist(x, y) {
  let m = 1e9
  for (let i = 0; i < RIVER.length - 1; i++) {
    const [ax, ay] = RIVER[i], [bx, by] = RIVER[i + 1], dx = bx - ax, dy = by - ay
    const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
    m = Math.min(m, Math.hypot(ax + t * dx - x, ay + t * dy - y))
  }
  return m
}
const FIELD = {
  fire: (x, y) => {
    let v = 0
    for (const it of SAMPLE.fire.items) { const [ix, iy] = it.pos; v = Math.max(v, (it.risk / 4) * Math.exp(-((x - ix) ** 2 + (y - iy) ** 2) / (2 * 46 * 46))) }
    if (riverDist(x, y) < 16) v *= 0.25                           // the river breaks the fuel bed
    return v + 0.14 * (hash(x, y) - 0.5)
  },
  flood: (x, y) => {
    const rd = riverDist(x, y)
    if (rd < 6) return 0                                          // the permanent channel is not "flood"
    let v = Math.exp(-(rd * rd) / (2 * 24 * 24)) * 0.46 * (0.55 + (x / 400) * 0.7)
    for (const it of SAMPLE.flood.items) { const [ix, iy] = it.pos; v = Math.max(v, (it.risk / 4) * Math.exp(-((x - ix) ** 2 + (y - iy) ** 2) / (2 * 30 * 30)) * (rd < 40 ? 1 : 0.5)) }
    return v + 0.1 * (hash(x, y) - 0.5)
  },
}
export function rasterCells(key) {                                // [{x, y, lv}] squares of predicted risk, level 1-4
  const out = []
  for (let y = 0; y < 260; y += 13) for (let x = 0; x < 400; x += 13) {
    const v = FIELD[key](x + 6.5, y + 6.5), lv = v > 0.8 ? 4 : v > 0.56 ? 3 : v > 0.34 ? 2 : v > 0.17 ? 1 : 0
    if (lv) out.push({ x, y, lv })
  }
  return out
}
export const WIND = (() => { const out = []; for (let y = 26; y < 260; y += 62) for (let x = 30; x < 400; x += 74) out.push([x, y]); return out })()
export const POINTS = {
  fire: SAMPLE.fire.items.flatMap((it, i) => Array.from({ length: it.risk * 3 }, (_, k) => [it.pos[0] + (hash(i * 7 + k, 3) - 0.5) * 54, it.pos[1] + (hash(k, i * 5 + 1) - 0.5) * 44])),
  flood: [[246, 144], [158, 94], [318, 188], [92, 62], [372, 222]],
}

export const REG = { fire: { w: -120.95, n: 38.92, lon: 0.5, lat: 0.253 }, flood: { w: -121.95, n: 38.42, lon: 0.5, lat: 0.253 } }   // 400×260 sketch units → lat/lon box
export const sketchToLatLng = (hz, [x, y]) => { const r = REG[hz]; return [r.n - (y / 260) * r.lat, r.w + (x / 400) * r.lon] }
export const sketchBounds = (hz) => { const r = REG[hz]; return [[r.n - r.lat, r.w], [r.n, r.w + r.lon]] }
