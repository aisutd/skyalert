import { useEffect, useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, ArrowUpDown, ChevronLeft, ChevronRight, Moon, Sun } from 'lucide-react'

const ROWS_PER_PAGE = 50

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

export default function AnalyticsPage() {
  const [fires, setFires] = useState(null)
  const [error, setError] = useState(null)
  // Strongest fires first by default
  const [sort, setSort] = useState({ key: 'frp', direction: 'desc' })
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

  // Filter, then sort. useMemo means this only re-runs when the data, filter or
  // sort changes, not on every render.
  const rows = useMemo(() => {
    if (!fires) return []
    const filtered = confidenceFilter === 'all' ? fires : fires.filter((f) => f.confidence === confidenceFilter)
    const column = COLUMNS.find((c) => c.key === sort.key)
    const flip = sort.direction === 'asc' ? 1 : -1
    // [...filtered] copies the array, because .sort() changes the array it's called on
    return [...filtered].sort((a, b) => {
      const x = column.sortValue(a)
      const y = column.sortValue(b)
      return (x < y ? -1 : x > y ? 1 : 0) * flip
    })
  }, [fires, confidenceFilter, sort])

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

  // Numbers for the tiles (always across all detections, not just the filter)
  const highCount = fires.filter((f) => f.confidence === 'h').length
  const frpValues = fires.map((f) => f.frp)
  const maxFrp = frpValues.length ? Math.max(...frpValues) : 0
  const avgFrp = frpValues.length ? frpValues.reduce((sum, v) => sum + v, 0) / frpValues.length : 0

  // How many rows each filter button would show
  const countFor = (id) => (id === 'all' ? fires.length : fires.filter((f) => f.confidence === id).length)

  return (
    <div className="page">
      <header className="page-header">
        <h1>Analytics</h1>
        <p className="muted">Every fire detection from the last few days. Click a column header to sort.</p>
      </header>

      <div className="kpis">
        <Kpi label="Detections" value={fires.length.toLocaleString()} />
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
                {COLUMNS.map((column) => {
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
                  {COLUMNS.map((column) => (
                    <td key={column.key} className={column.numeric ? 'num' : ''}>
                      {column.render(fire)}
                    </td>
                  ))}
                </tr>
              ))}
              {pageRows.length === 0 && (
                <tr>
                  <td colSpan={COLUMNS.length} className="empty">No detections match this filter.</td>
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
