// Past fires for showing the spread model's forecast next to what actually burned ("Spread on a past
// fire" on the map).
//
// All are from the 2024–2025 fire seasons, the model's held-out test years: it never trained on them
// and they weren't used to choose it. They were picked (docs/spread_retrain_plan.md, "Demo examples")
// as the test fires where the model caught the most real spread, so they show it at its best; the
// panel always shows the overall test scores next to them (lib/model.js, MODEL.spread.test).
// lat/lon are the exact tile centres from data/retrain/tiles.csv, so each forecast matches the test.
export const SPREAD_EXAMPLES = [
  { id: 'or-2024-09-04', label: 'Grant County, OR', day: '2024-09-04', lat: 44.2374468471588, lon: -119.48264576420598 },
  { id: 'id-2024-09-08', label: 'Adams County, ID', day: '2024-09-08', lat: 44.472282069888365, lon: -116.2498836943176 },
  { id: 'az-2025-07-28', label: 'Coconino County, AZ', day: '2025-07-28', lat: 36.48920152841195, lon: -111.95189270738209 },
  { id: 'ca-2025-08-11', label: 'San Luis Obispo County, CA', day: '2025-08-11', lat: 35.2564831528412, lon: -120.41904468125314 },
]

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
export const longDay = (iso) => { const [y, m, d] = iso.split('-'); return `${MONTHS[+m - 1]} ${+d}, ${y}` }   // "2024-09-04" -> "Sep 4, 2024"

// km² of one forecast pixel (pixels are 0.0089° square, so about 1 km north–south and less east–west)
export const pixelKm2 = (fc) => fc.pixel_km.north_south * fc.pixel_km.east_west

// The forecast drawn on a canvas, rows north to south (as the API sends them):
//   forecast = red where the model says likely, orange where its calibrated chance is at least 15%
//   today    = dark outline on each pixel burning on the fire day
//   actual   = blue square on each pixel that really burned the next day
const CELL = 10
export const POSSIBLE_FROM = 0.15
// One drawing for the map and the close-up. view = { r0, c0, size, cell, background, grid } crops and
// enlarges it (rows and columns of the 64 × 64 forecast, pixels per forecast square).
function drawGrid(fc, layers, theme, view = {}) {
  const { r0 = 0, c0 = 0, size = fc.size, cell = CELL, background = null, grid = false } = view
  const c = document.createElement('canvas')
  c.width = c.height = size * cell
  const g = c.getContext('2d')
  if (background) { g.fillStyle = background; g.fillRect(0, 0, c.width, c.height) }
  if (grid) {                                                     // faint 1 km grid, so the squares can be counted
    g.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,.08)' : 'rgba(0,0,0,.07)'
    g.lineWidth = 1
    for (let i = 1; i < size; i++) { g.beginPath(); g.moveTo(i * cell + 0.5, 0); g.lineTo(i * cell + 0.5, c.height); g.moveTo(0, i * cell + 0.5); g.lineTo(c.width, i * cell + 0.5); g.stroke() }
  }
  const each = (fn) => { for (let r = 0; r < size; r++) for (let k = 0; k < size; k++) fn(r0 + r, c0 + k, k * cell, r * cell) }
  if (layers.forecast) each((R, K, x, y) => {
    if (fc.likely[R][K]) g.fillStyle = 'rgba(229, 72, 58, 0.85)'
    else if (fc.probability[R][K] >= POSSIBLE_FROM) g.fillStyle = 'rgba(245, 160, 40, 0.5)'
    else return
    g.fillRect(x, y, cell, cell)
  })
  if (layers.today) {
    const w = Math.max(2, cell / 5)
    g.strokeStyle = theme === 'dark' ? '#FFFFFF' : '#25272C'
    g.lineWidth = w
    each((R, K, x, y) => { if (fc.fire_today[R][K]) g.strokeRect(x + w / 2 + 0.5, y + w / 2 + 0.5, cell - w - 1, cell - w - 1) })
  }
  if (layers.actual && fc.actual) {
    g.fillStyle = '#2a78d6'
    const m = cell * 0.3
    each((R, K, x, y) => { if (fc.actual[R][K]) g.fillRect(x + m, y + m, cell - 2 * m, cell - 2 * m) })
  }
  return c.toDataURL('image/png')
}
export const drawPast = (fc, layers, theme) => drawGrid(fc, layers, theme)

// An enlarged view of the forecast around the fire, for the panel: the squares that are burning, likely or
// possible, plus a margin, at least `min` squares across. Returns { url, size } (size in squares, ~1 km each).
export function drawCloseup(fc, theme, { px = 132, min = 13, margin = 3 } = {}) {
  const n = fc.size
  let r1 = n, r2 = -1, k1 = n, k2 = -1
  for (let r = 0; r < n; r++) for (let k = 0; k < n; k++) {
    if (fc.fire_today[r][k] || fc.likely[r][k] || fc.probability[r][k] >= POSSIBLE_FROM) { r1 = Math.min(r1, r); r2 = Math.max(r2, r); k1 = Math.min(k1, k); k2 = Math.max(k2, k) }
  }
  if (r2 < 0) { r1 = r2 = k1 = k2 = Math.floor(n / 2) }           // nothing burning or forecast: centre of the area
  const size = Math.min(n, Math.max(min, r2 - r1 + 1 + 2 * margin, k2 - k1 + 1 + 2 * margin))
  const clamp = (v) => Math.max(0, Math.min(n - size, v))
  const r0 = clamp(Math.round((r1 + r2) / 2 - (size - 1) / 2)), c0 = clamp(Math.round((k1 + k2) / 2 - (size - 1) / 2))
  const cell = Math.max(2, Math.floor(px / size))
  const background = theme === 'dark' ? '#2A2D33' : '#F3F1EC'
  return { url: drawGrid(fc, { today: true, forecast: true, actual: false }, theme, { r0, c0, size, cell, background, grid: cell >= 6 }), size }
}
