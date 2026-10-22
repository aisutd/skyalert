import { useEffect, useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, ArrowUpDown, ChevronLeft, ChevronRight, MapPin, Moon, Sun, X } from 'lucide-react'

const ROWS_PER_PAGE = 50
// "Open regional analysis" on the map shows detections within this distance of the clicked fire
const REGION_RADIUS_KM = 50

// Straight-line distance between two points on Earth (haversine formula), in km
function distanceKm(lat1, lon1, lat2, lon2) {
  const toRad = (d) => (d * Math.PI) / 180
  const dLat = toRad(lat2 - lat1)
  const dLon = toRad(lon2 - lon1)
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2
  return 6371 * 2 * Math.asin(Math.sqrt(a))
}

// Extra first column, only shown when the page is focused on one fire's region
const DISTANCE_COLUMN = {
  key: 'distance',
  label: 'Distance (km)',
  hint: 'How far this detection is from the fire you clicked on the map',
  numeric: true,
  sortValue: (f) => f.distance,
  render: (f) => f.distance.toFixed(1),
}

// VIIRS gives confidence as a letter. "rank" lets us sort it in a sensible order
// (low < nominal < high) instead of alphabetically.
const CONFIDENCE = {
  l: { label: 'Low', rank: 1 },
  n: { label: 'Nominal', rank: 2 },
  h: { label: 'High', rank: 3 },
}

// acq_time is a number like 815 meaning 08:15 UTC; turn it into "08:15"
function formatTime(acqTime) {
  const padded = String(acqTime).padStart(4, '0')
  return `${padded.slice(0, 2)}:${padded.slice(2)}`
}

// One entry per table column.
//   sortValue: what to compare when sorting by this column
//   numeric:   right-align it and sort biggest-first on the first click
const COLUMNS = [
  {
    key: 'detected',
    label: 'Detected (UTC)',
    sortValue: (f) => `${f.acq_date} ${formatTime(f.acq_time)}`,
    render: (f) => (
      <>
        {f.acq_date} <span className="muted">{formatTime(f.acq_time)}</span>
      </>
    ),
  },
  {
    key: 'frp',
    label: 'FRP (MW)',
    hint: 'Fire Radiative Power: how much heat the fire gives off',
    numeric: true,
    sortValue: (f) => f.frp,
    render: (f) => f.frp.toFixed(2),
  },
  {
    key: 'bright_ti4',
    label: 'Brightness (K)',
    hint: 'Pixel temperature from the I-4 band, in Kelvin',
    numeric: true,
    sortValue: (f) => f.bright_ti4,
    render: (f) => f.bright_ti4.toFixed(1),
  },
  {
    key: 'confidence',
    label: 'Confidence',
    hint: 'How sure NASA is that this is a real fire',
    sortValue: (f) => CONFIDENCE[f.confidence]?.rank ?? 0,
    render: (f) => (
      <span className={`pill pill-${f.confidence}`}>{CONFIDENCE[f.confidence]?.label ?? f.confidence}</span>
    ),
  },
  {
    key: 'daynight',
    label: 'Day / Night',
    sortValue: (f) => f.daynight,
    render: (f) =>
      f.daynight === 'D' ? (
        <span className="daynight"><Sun size={14} /> Day</span>
      ) : (
        <span className="daynight"><Moon size={14} /> Night</span>
      ),
  },
  { key: 'latitude', label: 'Latitude', numeric: true, sortValue: (f) => f.latitude, render: (f) => f.latitude.toFixed(4) },
  { key: 'longitude', label: 'Longitude', numeric: true, sortValue: (f) => f.longitude, render: (f) => f.longitude.toFixed(4) },
]

const FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'h', label: 'High' },
  { id: 'n', label: 'Nominal' },
  { id: 'l', label: 'Low' },
]

// Small number tiles above the table
function Kpi({ label, value }) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
    </div>
  )
}

// focus: { lat, lon, label } when opened from a fire on the map, otherwise null
export default function AnalyticsPage({ focus, onClearFocus }) {
  const [fires, setFires] = useState(null)
  const [error, setError] = useState(null)
  // Strongest fires first by default; nearest first when focused on a region
  const [sort, setSort] = useState(focus ? { key: 'distance', direction: 'asc' } : { key: 'frp', direction: 'desc' })
  const [confidenceFilter, setConfidenceFilter] = useState('all')
  const [page, setPage] = useState(0)

  // Load every detection from dashboard.py when the page opens
  useEffect(() => {
    fetch('/api/fires')
      .then((res) => {
        if (!res.ok) throw new Error(`Server returned ${res.status}`)
        return res.json()
      })
      .then(setFires)
      .catch((err) => setError(err.message))
  }, [])

  // When focused on a region: keep only detections near the clicked fire, with their distance
  const baseFires = useMemo(() => {
    if (!fires || !focus) return fires
    return fires
      .map((f) => ({ ...f, distance: distanceKm(focus.lat, focus.lon, f.latitude, f.longitude) }))
      .filter((f) => f.distance <= REGION_RADIUS_KM)
  }, [fires, focus])

  // useMemo keeps the same array between renders, so the sort below doesn't re-run for nothing
  const columns = useMemo(() => (focus ? [DISTANCE_COLUMN, ...COLUMNS] : COLUMNS), [focus])

  // If the region is cleared while sorted by distance, go back to the default sort
  useEffect(() => {
    if (!focus && sort.key === 'distance') setSort({ key: 'frp', direction: 'desc' })
  }, [focus, sort.key])

  // Filter, then sort. useMemo means this only re-runs when the data, filter or
  // sort changes, not on every render.
  const rows = useMemo(() => {
    if (!baseFires) return []
    const filtered = confidenceFilter === 'all' ? baseFires : baseFires.filter((f) => f.confidence === confidenceFilter)
    const column = columns.find((c) => c.key === sort.key) ?? COLUMNS[1]
    const flip = sort.direction === 'asc' ? 1 : -1
    // [...filtered] copies the array, because .sort() changes the array it's called on
    return [...filtered].sort((a, b) => {
      const x = column.sortValue(a)
      const y = column.sortValue(b)
      return (x < y ? -1 : x > y ? 1 : 0) * flip
    })
  }, [baseFires, confidenceFilter, sort, columns])

  // Clicking a header sorts by it; clicking the same header again flips the order
  const handleSort = (column) => {
    setSort((current) =>
      current.key === column.key
        ? { key: column.key, direction: current.direction === 'asc' ? 'desc' : 'asc' }
        : { key: column.key, direction: column.numeric ? 'desc' : 'asc' }
    )
    setPage(0)
  }

  const handleFilter = (id) => {
    setConfidenceFilter(id)
    setPage(0)
  }

  if (error) return <div className="page"><div className="page-error">Couldn't load fire data: {error}</div></div>
  if (!fires) return <div className="page"><div className="page-loading">Loading fire data…</div></div>

  const pageCount = Math.max(1, Math.ceil(rows.length / ROWS_PER_PAGE))
  const pageRows = rows.slice(page * ROWS_PER_PAGE, (page + 1) * ROWS_PER_PAGE)
  const firstShown = rows.length ? page * ROWS_PER_PAGE + 1 : 0
  const lastShown = Math.min((page + 1) * ROWS_PER_PAGE, rows.length)

  // Numbers for the tiles: all detections, or just the region's when focused
  // (the confidence filter doesn't change them)
  const highCount = baseFires.filter((f) => f.confidence === 'h').length
  const frpValues = baseFires.map((f) => f.frp)
  const maxFrp = frpValues.length ? Math.max(...frpValues) : 0
  const avgFrp = frpValues.length ? frpValues.reduce((sum, v) => sum + v, 0) / frpValues.length : 0

  // How many rows each filter button would show
  const countFor = (id) => (id === 'all' ? baseFires.length : baseFires.filter((f) => f.confidence === id).length)

  return (
    <div className="page">
      <header className="page-header">
        <h1>Analytics</h1>
        <p className="muted">
          {focus
            ? `Detections within ${REGION_RADIUS_KM} km of the fire you selected on the map.`
            : 'Every fire detection from the last few days.'}{' '}
          Click a column header to sort.
        </p>
        {focus && (
          <div className="region-chip">
            <MapPin size={14} />
            <span>
              Within {REGION_RADIUS_KM} km of <strong>{focus.label}</strong>
            </span>
            <button onClick={onClearFocus} aria-label="Show all detections" title="Show all detections">
              <X size={14} />
            </button>
          </div>
        )}
      </header>

      <div className="kpis">
        <Kpi label="Detections" value={baseFires.length.toLocaleString()} />
        <Kpi label="High confidence" value={highCount.toLocaleString()} />
        <Kpi label="Strongest fire (FRP)" value={`${maxFrp.toFixed(1)} MW`} />
        <Kpi label="Average FRP" value={`${avgFrp.toFixed(1)} MW`} />
      </div>

      <div className="table-card">
        <div className="table-toolbar">
          <div className="filter-group" role="group" aria-label="Filter by confidence">
            <span className="filter-label">Confidence</span>
            {FILTERS.map((f) => (
              <button
                key={f.id}
                className={`filter-chip${confidenceFilter === f.id ? ' active' : ''}`}
                onClick={() => handleFilter(f.id)}
              >
                {f.label} <span className="chip-count">{countFor(f.id).toLocaleString()}</span>
              </button>
            ))}
          </div>
          <span className="muted">
            {firstShown.toLocaleString()}–{lastShown.toLocaleString()} of {rows.length.toLocaleString()}
          </span>
        </div>

        <div className="table-scroll">
          <table className="fire-table">
            <thead>
              <tr>
                {columns.map((column) => {
                  const active = sort.key === column.key
                  const Arrow = !active ? ArrowUpDown : sort.direction === 'asc' ? ArrowUp : ArrowDown
                  return (
                    <th
                      key={column.key}
                      className={`${column.numeric ? 'num' : ''}${active ? ' sorted' : ''}`}
                      aria-sort={active ? (sort.direction === 'asc' ? 'ascending' : 'descending') : 'none'}
                    >
                      <button className="sort-button" onClick={() => handleSort(column)} title={column.hint}>
                        {column.label}
                        <Arrow size={13} />
                      </button>
                    </th>
                  )
                })}
              </tr>
            </thead>
            <tbody>
              {pageRows.map((fire) => (
                <tr key={fire.id}>
                  {columns.map((column) => (
                    <td key={column.key} className={column.numeric ? 'num' : ''}>
                      {column.render(fire)}
                    </td>
                  ))}
                </tr>
              ))}
              {pageRows.length === 0 && (
                <tr>
                  <td colSpan={columns.length} className="empty">No detections match this filter.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="pager">
          <button className="pager-button" onClick={() => setPage((p) => p - 1)} disabled={page === 0}>
            <ChevronLeft size={16} /> Previous
          </button>
          <span className="muted">Page {page + 1} of {pageCount}</span>
          <button className="pager-button" onClick={() => setPage((p) => p + 1)} disabled={page >= pageCount - 1}>
            Next <ChevronRight size={16} />
          </button>
        </div>
      </div>
    </div>
  )
}
