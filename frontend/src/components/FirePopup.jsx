import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { X } from 'lucide-react'
import { getFireContext } from '../lib/api.js'

const PANEL_WIDTH = 360 // px, on wide screens
const GAP = 18 // px between the dot and the popup
const MARGIN = 16 // px kept free around the map's edges
const COMPACT_BELOW = 640 // map narrower than this: popup docks to the bottom

const COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
const compass = (deg) => COMPASS[Math.round(deg / 22.5) % 16]

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))

function formatCoord(lat, lon) {
  return `${Math.abs(lat).toFixed(4)}° ${lat >= 0 ? 'N' : 'S'}  ${Math.abs(lon).toFixed(4)}° ${lon >= 0 ? 'E' : 'W'}`
}

// Pick a spot for the popup next to the dot, inside the map, clear of the stat cards
function computePlace(map, fire, contentHeight, safeTop) {
  const size = map.getSize()
  const p = map.latLngToContainerPoint([fire.latitude, fire.longitude])
  const top = safeTop + MARGIN
  const maxHeight = size.y - top - MARGIN
  const height = Math.min(contentHeight || 420, maxHeight)

  if (size.x < COMPACT_BELOW) {
    const width = size.x - MARGIN * 2
    return { x: MARGIN, y: Math.max(top, size.y - height - MARGIN), width, height, maxHeight, compact: true }
  }

  const width = PANEL_WIDTH
  const roomRight = size.x - p.x - GAP - MARGIN
  const roomLeft = p.x - GAP - MARGIN
  const right = roomRight >= width || (roomLeft < width && roomRight >= roomLeft)
  const x = clamp(right ? p.x + GAP : p.x - GAP - width, MARGIN, size.x - width - MARGIN)
  const y = clamp(p.y - height * 0.3, top, Math.max(top, size.y - height - MARGIN))
  return { x, y, width, height, maxHeight, compact: false }
}

// Place name, weather and risk for the fire, from the FastAPI backend
// (lib/api.js -> backend/dashboard.py /api/fire-context -> backend/fire_context.py)
function useFireContext(fire) {
  const [state, setState] = useState({ status: 'loading', context: null })
  useEffect(() => {
    let stale = false // a different dot was picked before this answer arrived
    setState({ status: 'loading', context: null })
    getFireContext(fire.latitude, fire.longitude)
      .then((context) => !stale && setState({ status: 'ready', context }))
      .catch((err) => !stale && setState({ status: 'error', context: { error: err.message } }))
    return () => {
      stale = true
    }
  }, [fire])
  return state
}

function Skeleton({ width }) {
  return <span className="popup-skel" style={{ width }} />
}

// tag: short orange text on the right of the label, e.g. the wind direction
function Stat({ label, value, sub, tag, accent }) {
  return (
    <div className="popup-stat">
      <div className="popup-stat-head">
        <div className="popup-label">{label}</div>
        {tag && <span className="popup-stat-tag">{tag}</span>}
      </div>
      <div className={`popup-stat-value popup-mono${accent ? ' accent' : ''}`}>{value}</div>
      {sub && <div className="popup-stat-sub">{sub}</div>}
    </div>
  )
}

// The card that opens next to a clicked fire dot.
// map: the Leaflet map. fire: one row from /api/fires. safeTop: px of the map hidden by the stat cards.
export default function FirePopup({ map, fire, safeTop, onClose, onOpenAnalysis }) {
  const panel = useRef(null)
  const [place, setPlace] = useState(null)
  const { status, context } = useFireContext(fire)
  const loading = status === 'loading'
  const c = context || {}
  const w = c.weather

  // Position the popup, and again whenever its content or the map changes size
  useLayoutEffect(() => {
    const update = () => setPlace(computePlace(map, fire, panel.current?.scrollHeight, safeTop))
    update()
    const observer = new ResizeObserver(update)
    observer.observe(panel.current)
    map.on('resize', update)
    return () => {
      observer.disconnect()
      map.off('resize', update)
    }
  }, [map, fire, safeTop])

  // If panning or zooming puts the dot under the popup, move the popup (or the map, when docked)
  useEffect(() => {
    const onMoveEnd = () => {
      if (!place) return
      const p = map.latLngToContainerPoint([fire.latitude, fire.longitude])
      const pad = 24
      const covered =
        p.x > place.x - pad && p.x < place.x + place.width + pad && p.y > place.y - pad && p.y < place.y + place.height + pad
      if (!covered) return
      if (place.compact) map.panBy([0, p.y - (place.y - 56)])
      else setPlace(computePlace(map, fire, panel.current?.scrollHeight, safeTop))
    }
    map.on('moveend', onMoveEnd)
    return () => map.off('moveend', onMoveEnd)
  }, [map, fire, place, safeTop])

  // Keyboard users land in the popup; Escape closes it
  useEffect(() => {
    panel.current?.focus({ preventScroll: true })
    const onKey = (e) => e.key === 'Escape' && onClose()
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [fire, onClose])

  const placeName = c.location ? c.location.name : loading ? null : 'Location unavailable'
  const country = c.location?.country && c.location.country !== 'United States' ? ` · ${c.location.country}` : ''

  return (
    <section
      ref={panel}
      className={`fire-popup${place?.compact ? ' compact' : ''}`}
      role="dialog"
      aria-modal="false"
      aria-labelledby="popup-title"
      aria-busy={loading}
      tabIndex={-1}
      style={
        place
          ? { width: place.width, maxHeight: place.maxHeight, transform: `translate3d(${Math.round(place.x)}px, ${Math.round(place.y)}px, 0)` }
          : { visibility: 'hidden' }
      }
    >
      <header className="popup-head">
        <span className="popup-eyebrow">Fire details</span>
        <button type="button" className="popup-close" aria-label="Close fire details" onClick={onClose}>
          <X size={14} />
        </button>
      </header>

      <div className="popup-body">
        <div className="popup-item">
          <h2 className="popup-title" id="popup-title">{placeName ?? <Skeleton width="70%" />}</h2>
          <div className="popup-coords">{formatCoord(fire.latitude, fire.longitude)}{country}</div>
        </div>

        <div className="popup-item">
          {w ? (
            <div className="popup-risk" data-risk={w.risk.key}>
              <span className="popup-risk-dot" aria-hidden="true" />
              <div>
                <div className="popup-label">Fire-weather risk</div>
                <div className="popup-risk-level">{w.risk.label}</div>
              </div>
              <div className="popup-risk-index" title={w.risk.method}>
                <span className="popup-mono">{w.risk.index.toFixed(0)}</span>
                <span className="popup-risk-max">/100</span>
              </div>
              <div className="popup-meter" aria-hidden="true">
                <span style={{ width: `${Math.max(2, w.risk.index)}%` }} />
              </div>
            </div>
          ) : loading ? (
            <Skeleton width="100%" />
          ) : (
            <div className="popup-muted">Weather unavailable</div>
          )}
        </div>

        <div className="popup-item">
          <div className="popup-section-title">Satellite</div>
          <div className="popup-stats">
            <Stat label="FRP" value={`${fire.frp.toFixed(1)} MW`} />
            <Stat label="Confidence" value={fire.confidence_label} accent />
            <Stat
              label="Seen"
              value={`${fire.acq_time_utc} UTC`}
              sub={`${fire.acq_date.slice(5)} · ${fire.daynight === 'N' ? 'night' : 'day'}`}
            />
          </div>
        </div>

        {w ? (
          <div className="popup-item">
            <div className="popup-section-title">Weather</div>
            <div className="popup-stats">
              <Stat label="Temp" value={`${Math.round(w.current.temperature_f)}°F`} />
              <Stat label="Humidity" value={`${Math.round(w.current.humidity)}%`} />
              <Stat label="Wind" value={`${Math.round(w.current.wind_mph)} mph`} tag={compass(w.current.wind_direction)} />
            </div>
          </div>
        ) : (
          loading && (
            <div className="popup-item">
              <Skeleton width="100%" />
            </div>
          )
        )}

        <div className="popup-item">
          <button
            type="button"
            className="popup-analysis"
            onClick={() => onOpenAnalysis({ lat: fire.latitude, lon: fire.longitude, label: c.location?.name ?? formatCoord(fire.latitude, fire.longitude) })}
          >
            Regional analysis <span aria-hidden="true">→</span>
          </button>
        </div>
      </div>
    </section>
  )
}
