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
const POSSIBLE_FROM = 0.15
export function drawPast(fc, layers, theme) {
  const n = fc.size, c = document.createElement('canvas')
  c.width = c.height = n * CELL
  const g = c.getContext('2d')
  for (let r = 0; r < n && layers.forecast; r++) {
    for (let k = 0; k < n; k++) {
      if (fc.likely[r][k]) g.fillStyle = 'rgba(229, 72, 58, 0.78)'
      else if (fc.probability[r][k] >= POSSIBLE_FROM) g.fillStyle = 'rgba(245, 160, 40, 0.42)'
      else continue
      g.fillRect(k * CELL, r * CELL, CELL, CELL)
    }
  }
  if (layers.today) {
    g.strokeStyle = theme === 'dark' ? '#FFFFFF' : '#25272C'
    g.lineWidth = 2
    for (let r = 0; r < n; r++) for (let k = 0; k < n; k++) if (fc.fire_today[r][k]) g.strokeRect(k * CELL + 1.5, r * CELL + 1.5, CELL - 3, CELL - 3)
  }
  if (layers.actual && fc.actual) {
    g.fillStyle = '#2a78d6'
    for (let r = 0; r < n; r++) for (let k = 0; k < n; k++) if (fc.actual[r][k]) g.fillRect(k * CELL + 3, r * CELL + 3, CELL - 6, CELL - 6)
  }
  return c.toDataURL('image/png')
}
