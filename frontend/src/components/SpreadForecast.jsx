import { useEffect } from 'react'
import { useMap } from 'react-leaflet'
import { X } from 'lucide-react'
import L from '../lib/leaflet.js'
import { SPREAD_EXAMPLES } from '../lib/spreadExamples.js'

// The next-day spread forecast from /api/spread-forecast (backend/spread_forecast.py):
// SpreadOverlay draws it on the map, SpreadLegend explains it.

const CELL = 10 // canvas pixels per forecast pixel, so outlines and markers stay crisp
const POSSIBLE_FROM = 0.15 // below this probability nothing is drawn

// "Likely" = at or above the model's threshold (chosen on 2023 fires to best match what burned)
const COLORS = {
  likely: [229, 72, 58],
  possible: [242, 160, 29],
}
const ACTUAL = '#1f6feb' // what actually burned on the forecast day (past examples)

// show: which layers to draw, { forecast, today, actual } (the legend's checkboxes)
function drawForecast(forecast, outline, show) {
  const { size, probability, likely, fire_today: today, actual, threshold } = forecast
  const canvas = document.createElement('canvas')
  canvas.width = size * CELL
  canvas.height = size * CELL
  const ctx = canvas.getContext('2d')
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      if (!show.forecast) break
      const p = probability[row][col]
      if (likely[row][col]) {
        ctx.fillStyle = `rgba(${COLORS.likely.join(',')}, 0.78)`
      } else if (p >= POSSIBLE_FROM) {
        // Fade in from faint to half-opaque as the probability approaches the threshold
        const alpha = 0.18 + 0.32 * ((p - POSSIBLE_FROM) / (threshold - POSSIBLE_FROM))
        ctx.fillStyle = `rgba(${COLORS.possible.join(',')}, ${alpha.toFixed(3)})`
      } else {
        continue
      }
      ctx.fillRect(col * CELL, row * CELL, CELL, CELL)
    }
  }
  // Today's fire: an outline inside each burning pixel
  ctx.strokeStyle = outline
  ctx.lineWidth = 2
  for (let row = 0; row < size && show.today; row++) {
    for (let col = 0; col < size; col++) {
      if (today[row][col]) ctx.strokeRect(col * CELL + 1.5, row * CELL + 1.5, CELL - 3, CELL - 3)
    }
  }
  // Past examples: a blue square in each pixel that actually burned on the forecast day. Red with a
  // blue square = forecast and burned; blue square alone = burned but not forecast.
  if (actual && show.actual) {
    ctx.fillStyle = ACTUAL
    for (let row = 0; row < size; row++) {
      for (let col = 0; col < size; col++) {
        if (actual[row][col]) ctx.fillRect(col * CELL + 3, row * CELL + 3, CELL - 6, CELL - 6)
      }
    }
  }
  return canvas.toDataURL('image/png')
}

// forecast rows run north to south and columns west to east, so the image maps straight onto its bounds
export function SpreadOverlay({ forecast, theme, show }) {
  const map = useMap()
  useEffect(() => {
    const { south, west, north, east } = forecast.bounds
    const bounds = L.latLngBounds([south, west], [north, east])
    const outline = theme === 'dark' ? '#ffffff' : '#1b1d22'
    // Its own pane above the fire dots (overlayPane, z 400) and below popups and markers, so the
    // dots don't hide the forecast
    if (!map.getPane('spreadPane')) map.createPane('spreadPane').style.zIndex = 450
    const image = L.imageOverlay(drawForecast(forecast, outline, show), bounds, {
      pane: 'spreadPane',
      interactive: false,
      className: 'spread-overlay',
    }).addTo(map)
    const frame = L.rectangle(bounds, {
      pane: 'spreadPane',
      interactive: false,
      fill: false,
      weight: 1.5,
      dashArray: '5 4',
      className: 'spread-frame',
    }).addTo(map)
    return () => {
      map.removeLayer(image)
      map.removeLayer(frame)
    }
  }, [map, forecast, theme, show])

  // Bring the forecast area into view when a new forecast arrives
  useEffect(() => {
    const { south, west, north, east } = forecast.bounds
    map.flyToBounds([[south, west], [north, east]], { padding: [60, 60], maxZoom: 10, duration: 0.8 })
  }, [map, forecast])

  return null
}

const formatDay = (iso) => {
  const date = new Date(`${iso}T00:00:00`)
  const options = { month: 'short', day: 'numeric' }
  // Show the year for past examples
  if (date.getFullYear() !== new Date().getFullYear()) options.year = 'numeric'
  return date.toLocaleDateString('en-US', options)
}
const percent = (value) => `${Math.round(value * 100)}%`

// A legend row whose checkbox shows or hides that layer on the map
function LayerKey({ swatch, checked, onChange, children }) {
  return (
    <li>
      <label className="spread-key">
        <input type="checkbox" checked={checked} onChange={onChange} />
        <span className={`spread-swatch ${swatch}`} aria-hidden="true" />
        {children}
      </label>
    </li>
  )
}

// state: { status: 'loading' | 'ready' | 'error', data, error, example }
// show / onToggle: which layers are drawn, toggled by the checkboxes
export function SpreadLegend({ state, onClose, show, onToggle }) {
  const forecast = state.data
  const km2 = forecast ? forecast.pixel_km.north_south * forecast.pixel_km.east_west : 0
  const scores = forecast?.model.test_scores
  const check = forecast?.check
  const area = (pixels) => `${Math.round(pixels * km2)} km²`

  return (
    <section className="map-control spread-legend" aria-live="polite" aria-label="Next-day spread forecast">
      <header className="spread-head">
        <span className="popup-eyebrow">Next-day spread forecast</span>
        <button type="button" className="popup-close" aria-label="Close the spread forecast" onClick={onClose}>
          <X size={14} />
        </button>
      </header>

      {state.example && (
        <p className="spread-example">
          Past fire: <strong>{state.example.label}</strong>. From the 2024–25 test seasons, which the model never saw.
        </p>
      )}

      {state.status === 'loading' && <p className="spread-muted">Building the forecast from satellite fire, weather and terrain data… (about 10 seconds)</p>}

      {state.status === 'error' && <p className="spread-error">Couldn't make a forecast: {state.error}</p>}

      {forecast && (
        <>
          <p className="spread-dates">
            For <strong>{formatDay(forecast.forecast_day)}</strong>, from fires seen {formatDay(forecast.inputs.fire_days['t-2'])} – {formatDay(forecast.fire_day)}
          </p>
          <ul className="spread-keys">
            <LayerKey swatch="today" checked={show.today} onChange={() => onToggle('today')}>
              Burning on {formatDay(forecast.fire_day)} <span className="spread-num">{area(forecast.fire_today_pixels)}</span>
            </LayerKey>
            <LayerKey swatch="likely" checked={show.forecast} onChange={() => onToggle('forecast')}>
              Forecast: likely burning <span className="spread-num">{area(forecast.forecast_pixels)}</span>
            </LayerKey>
            <li className="spread-key-sub">
              <span className="spread-swatch possible" aria-hidden="true" />
              Possible
            </li>
            {forecast.actual && (
              <LayerKey swatch="actual" checked={show.actual} onChange={() => onToggle('actual')}>
                Actually burned on {formatDay(forecast.forecast_day)} <span className="spread-num">{area(check.burned_pixels)}</span>
              </LayerKey>
            )}
          </ul>
          <p className="spread-muted">
            New spread forecast: {Math.round(forecast.new_fire_pixels * km2)} km² beyond today's fire.
          </p>
          {check && (
            <p className="spread-check">
              <strong>How this forecast did:</strong> the fire spread to {area(check.new_fire_pixels)} of new ground, and
              the model had forecast {area(check.new_fire_caught_pixels)} of it. Overlap score (IoU) with what burned:{' '}
              {check.model_iou?.toFixed(2) ?? '–'}, against {check.persistence_iou?.toFixed(2) ?? '–'} for assuming the
              fire stays where it was.
            </p>
          )}
          <p className="spread-accuracy">
            How good is it? Tested on 2024–25 fires it never saw, its overlap score with what actually burned the next
            day (IoU, 0–1) was {scores.all_fires.model.iou.toFixed(2)}, against {scores.all_fires.persistence.iou.toFixed(2)} for
            assuming tomorrow looks like today; on larger, growing fires {scores.growing_fires.model.iou.toFixed(2)} against{' '}
            {scores.growing_fires.persistence.iou.toFixed(2)}. About {percent(scores.all_fires.model.precision)} of the area it
            marks likely did burn, and it caught {percent(scores.growing_fires.model.new_fire_recall)} of new spread on
            growing fires. Treat it as guidance, not a certainty.
          </p>
          {forecast.notes.length > 0 && (
            <ul className="spread-notes">
              {forecast.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  )
}

// A short list of past fires (held-out test seasons) to show the forecast against what burned
export function SpreadExamples({ active, onPick }) {
  return (
    <section className="map-control spread-examples" aria-label="Past fire examples">
      <div className="popup-label">Past fire examples</div>
      {SPREAD_EXAMPLES.map((example) => (
        <button
          key={example.id}
          type="button"
          className={`spread-example-button${active === example.id ? ' active' : ''}`}
          aria-pressed={active === example.id}
          onClick={() => onPick(example)}
        >
          <span>{example.label}</span>
          <span className="spread-num">{formatDay(example.day)}</span>
        </button>
      ))}
    </section>
  )
}
