import { useEffect } from 'react'
import { useMap } from 'react-leaflet'
import { X } from 'lucide-react'
import L from '../lib/leaflet.js'

// The next-day spread forecast from /api/spread-forecast (backend/spread_forecast.py):
// SpreadOverlay draws it on the map, SpreadLegend explains it.

const CELL = 8 // canvas pixels per forecast pixel, so today's-fire outlines stay crisp
const POSSIBLE_FROM = 0.15 // below this probability nothing is drawn

// "Likely" = at or above the model's threshold (chosen on 2023 fires to best match what burned)
const COLORS = {
  likely: [229, 72, 58],
  possible: [242, 160, 29],
}

function drawForecast(forecast, outline) {
  const { size, probability, likely, fire_today: today, threshold } = forecast
  const canvas = document.createElement('canvas')
  canvas.width = size * CELL
  canvas.height = size * CELL
  const ctx = canvas.getContext('2d')
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
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
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      if (today[row][col]) ctx.strokeRect(col * CELL + 1.5, row * CELL + 1.5, CELL - 3, CELL - 3)
    }
  }
  return canvas.toDataURL('image/png')
}

// forecast rows run north to south and columns west to east, so the image maps straight onto its bounds
export function SpreadOverlay({ forecast, theme }) {
  const map = useMap()
  useEffect(() => {
    const { south, west, north, east } = forecast.bounds
    const bounds = L.latLngBounds([south, west], [north, east])
    const outline = theme === 'dark' ? '#ffffff' : '#1b1d22'
    // Its own pane above the fire dots (overlayPane, z 400) and below popups and markers, so the
    // dots don't hide the forecast
    if (!map.getPane('spreadPane')) map.createPane('spreadPane').style.zIndex = 450
    const image = L.imageOverlay(drawForecast(forecast, outline), bounds, {
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
  }, [map, forecast, theme])

  // Bring the forecast area into view when a new forecast arrives
  useEffect(() => {
    const { south, west, north, east } = forecast.bounds
    map.flyToBounds([[south, west], [north, east]], { padding: [60, 60], maxZoom: 10, duration: 0.8 })
  }, [map, forecast])

  return null
}

const formatDay = (iso) => new Date(`${iso}T00:00:00`).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
const percent = (value) => `${Math.round(value * 100)}%`

// state: { status: 'loading' | 'ready' | 'error', data, error }
export function SpreadLegend({ state, onClose }) {
  const forecast = state.data
  const km2 = forecast ? forecast.pixel_km.north_south * forecast.pixel_km.east_west : 0
  const scores = forecast?.model.test_scores

  return (
    <section className="map-control spread-legend" aria-live="polite" aria-label="Next-day spread forecast">
      <header className="spread-head">
        <span className="popup-eyebrow">Next-day spread forecast</span>
        <button type="button" className="popup-close" aria-label="Close the spread forecast" onClick={onClose}>
          <X size={14} />
        </button>
      </header>

      {state.status === 'loading' && <p className="spread-muted">Building the forecast from satellite fire, weather and terrain data… (about 10 seconds)</p>}

      {state.status === 'error' && <p className="spread-error">Couldn't make a forecast: {state.error}</p>}

      {forecast && (
        <>
          <p className="spread-dates">
            For <strong>{formatDay(forecast.forecast_day)}</strong>, from fires seen {formatDay(forecast.inputs.fire_days['t-2'])} – {formatDay(forecast.fire_day)}
          </p>
          <ul className="spread-keys">
            <li>
              <span className="spread-swatch likely" aria-hidden="true" />
              Likely burning <span className="spread-num">{Math.round(forecast.forecast_pixels * km2)} km²</span>
            </li>
            <li>
              <span className="spread-swatch possible" aria-hidden="true" />
              Possible
            </li>
            <li>
              <span className="spread-swatch today" aria-hidden="true" />
              Burning on {formatDay(forecast.fire_day)} <span className="spread-num">{Math.round(forecast.fire_today_pixels * km2)} km²</span>
            </li>
          </ul>
          <p className="spread-muted">
            New spread forecast: {Math.round(forecast.new_fire_pixels * km2)} km² beyond today's fire.
          </p>
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
