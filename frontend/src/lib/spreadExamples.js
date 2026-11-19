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
  const { r0 = 0, c0 = 0, size = fc.size, cell = CELL, background = null, grid = false, gridColor = null } = view
  const c = document.createElement('canvas')
  c.width = c.height = size * cell
  const g = c.getContext('2d')
  if (background) { g.fillStyle = background; g.fillRect(0, 0, c.width, c.height) }
  if (grid) {                                                     // faint 1 km grid, so the squares can be counted
    g.fillStyle = gridColor || (theme === 'dark' ? 'rgba(255,255,255,.1)' : 'rgba(0,0,0,.09)')
    const lw = Math.max(1, Math.round(cell / 24))                 // whole canvas pixels, so scaling the image up keeps lines sharp
    for (let i = 1; i < size; i++) { g.fillRect(i * cell, 0, lw, c.height); g.fillRect(0, i * cell, c.width, lw) }
  }
  const each = (fn) => { for (let r = 0; r < size; r++) for (let k = 0; k < size; k++) fn(r0 + r, c0 + k, k * cell, r * cell) }
  if (layers.forecast) each((R, K, x, y) => {
    if (fc.likely[R][K]) g.fillStyle = 'rgba(229, 72, 58, 0.85)'
    else if (fc.probability[R][K] >= POSSIBLE_FROM) g.fillStyle = 'rgba(245, 160, 40, 0.5)'
    else return
    g.fillRect(x, y, cell, cell)
  })
  if (layers.today) {                                             // a frame of four bars on whole pixels: a stroked rect lands on half
    const w = Math.max(2, Math.round(cell / 5)), m = Math.max(1, Math.round(cell / 10))   // pixels, which blur into bands when enlarged
    g.fillStyle = theme === 'dark' ? '#FFFFFF' : '#25272C'
    each((R, K, x, y) => {
      if (!fc.fire_today[R][K]) return
      const a = x + m, b = y + m, s = cell - 2 * m
      g.fillRect(a, b, s, w); g.fillRect(a, b + s - w, s, w); g.fillRect(a, b + w, w, s - 2 * w); g.fillRect(a + s - w, b + w, w, s - 2 * w)
    })
  }
  if (layers.actual && fc.actual) {
    g.fillStyle = '#2a78d6'
    const m = cell * 0.3
    each((R, K, x, y) => { if (fc.actual[R][K]) g.fillRect(x + m, y + m, cell - 2 * m, cell - 2 * m) })
  }
  return c.toDataURL('image/png')
}
export const drawPast = (fc, layers, theme) => drawGrid(fc, layers, theme)

// The part of the forecast worth looking at: this fire's squares (burning, likely or possible) plus a margin, at
// least `min` squares across. "This fire" is the group of such squares nearest the clicked spot, which is the
// centre of the 64 × 64 area (to within half a square); squares up to `gap` apart count as one fire. Other
// fires in the area (farm burns a few km away, say) stay on the map but don't widen the close-up.
// Returns r0, c0, size: rows (from the north) and columns of the forecast; bounds: where that is on the map
// ([[south, west], [north, east]], the squares' outer edges); counts: squares of each kind inside it;
// outside: how many burning/likely/possible squares of the area it leaves out.
export function forecastFocus(fc, { min = 9, margin = 2, gap = 2 } = {}) {
  const n = fc.size, mid = (n - 1) / 2
  const hot = (r, k) => fc.fire_today[r][k] || fc.likely[r][k] || fc.probability[r][k] >= POSSIBLE_FROM
  let seed = null, best = Infinity, total = 0
  for (let r = 0; r < n; r++) for (let k = 0; k < n; k++) {
    if (!hot(r, k)) continue
    total++
    const d = (r - mid) ** 2 + (k - mid) ** 2
    if (d < best) { best = d; seed = [r, k] }
  }
  let r1 = Math.floor(mid), r2 = r1, k1 = r1, k2 = r1             // nothing burning or forecast: centre of the area
  if (seed) {
    const seen = new Set([seed[0] * n + seed[1]]), queue = [seed]
    ;[r1, k1] = seed; [r2, k2] = seed
    while (queue.length) {
      const [r, k] = queue.pop()
      r1 = Math.min(r1, r); r2 = Math.max(r2, r); k1 = Math.min(k1, k); k2 = Math.max(k2, k)
      for (let dr = -gap; dr <= gap; dr++) for (let dk = -gap; dk <= gap; dk++) {
        const R = r + dr, K = k + dk, id = R * n + K
        if (R < 0 || K < 0 || R >= n || K >= n || seen.has(id) || !hot(R, K)) continue
        seen.add(id); queue.push([R, K])
      }
    }
  }
  const size = Math.min(n, Math.max(min, r2 - r1 + 1 + 2 * margin, k2 - k1 + 1 + 2 * margin))
  const clamp = (v) => Math.max(0, Math.min(n - size, v))
  const r0 = clamp(Math.round((r1 + r2) / 2 - (size - 1) / 2)), c0 = clamp(Math.round((k1 + k2) / 2 - (size - 1) / 2))
  const counts = { likely: 0, possible: 0, today: 0 }
  let inside = 0
  for (let r = r0; r < r0 + size; r++) for (let k = c0; k < c0 + size; k++) {
    if (fc.likely[r][k]) counts.likely++
    else if (fc.probability[r][k] >= POSSIBLE_FROM) counts.possible++
    if (fc.fire_today[r][k]) counts.today++
    if (hot(r, k)) inside++
  }
  const b = fc.bounds, dLat = (b.north - b.south) / n, dLon = (b.east - b.west) / n
  const north = b.north - r0 * dLat, west = b.west + c0 * dLon
  return { r0, c0, size, bounds: [[north - size * dLat, west], [north, west + size * dLon]], counts, outside: total - inside }
}

// An enlarged view of the forecast around the fire, for the panel (the focus area above). px is the canvas
// width, drawn at twice the size it's shown so it stays sharp. Returns { url, focus } (focus as forecastFocus gives it).
export function drawCloseup(fc, theme, { px = 480 } = {}) {
  const focus = forecastFocus(fc), { r0, c0, size } = focus
  const cell = Math.max(2, Math.floor(px / size))
  const background = theme === 'dark' ? '#2A2D33' : '#F3F1EC'
  return { url: drawGrid(fc, { today: true, forecast: true, actual: false }, theme, { r0, c0, size, cell, background, grid: cell >= 6 }), focus }
}

// Just the 1 km grid over the focus area, to lay on the map when you're zoomed in close enough to count squares
export function drawFocusGrid(fc, theme) {
  const { r0, c0, size } = forecastFocus(fc)
  return drawGrid(fc, {}, theme, { r0, c0, size, cell: 24, grid: true, gridColor: theme === 'dark' ? 'rgba(255,255,255,.24)' : 'rgba(37,39,44,.3)' })
}
