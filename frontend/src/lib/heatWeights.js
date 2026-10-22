// Weight each fire so a heatmap cell sums to the percentile rank of its fire count.
//
// Leaflet.heat bins points into radius/2-pixel cells and colors each cell by its
// summed weight divided by the busiest cell. With raw counts, a few dense hotspots
// push almost everything else into blue. Ranking cells by count keeps the real
// ordering of densities but spreads it across the full color scale: red is the
// densest areas, blue the sparsest.
export function densityWeights(fires, zoom, radius) {
  const worldPx = 256 * 2 ** zoom
  const cell = radius / 2

  // Which heatmap cell each fire falls in (Web Mercator pixels at this zoom)
  const cellIds = fires.map((f) => {
    const latRad = (f.latitude * Math.PI) / 180
    const x = ((f.longitude + 180) / 360) * worldPx
    const y = ((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * worldPx
    return `${Math.floor(x / cell)},${Math.floor(y / cell)}`
  })

  // Fires per cell
  const counts = new Map()
  cellIds.forEach((id) => counts.set(id, (counts.get(id) ?? 0) + 1))

  // Percentile rank of each count: the share of cells with that many fires or fewer
  const sorted = [...counts.values()].sort((a, b) => a - b)
  const rankOf = new Map()
  sorted.forEach((count, i) => rankOf.set(count, (i + 1) / sorted.length)) // later (higher) i wins for ties

  return cellIds.map((id) => {
    const count = counts.get(id)
    return rankOf.get(count) / count
  })
}
