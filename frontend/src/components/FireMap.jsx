import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AttributionControl, CircleMarker, MapContainer, TileLayer, useMap, useMapEvents } from 'react-leaflet'
import { Minus, Plus, Volume2, VolumeX } from 'lucide-react'
import L from '../lib/leaflet.js'
import { densityWeights } from '../lib/heatWeights.js'
import { loadSoundSetting, playSound, saveSoundSetting } from '../lib/sound.js'
import FirePopup from './FirePopup.jsx'

const CENTER = [39, -98]
const ZOOM_START = 4

// Light: Esri street map. Dark: Stadia "Alidade Smooth Dark".
const TILES = {
  light: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
  dark: 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png',
}

// Credit line each provider's terms require, shown in the map's corner
const TILE_ATTRIBUTION = {
  light: 'Tiles &copy; <a href="https://www.esri.com/">Esri</a> &mdash; Source: Esri, HERE, Garmin, USGS, NGA, EPA, USDA, NPS',
  dark: '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}

// Keep blur below the radius; a wider blur flattens every peak into pale blue
const HEAT = {
  radius: 14,
  blur: 8,
  minOpacity: 0.3,
  gradient: { 0.3: 'blue', 0.5: 'cyan', 0.7: 'lime', 0.85: 'yellow', 1.0: 'red' },
}

// The red fill is the visible dot (DOT_RADIUS). The stroke is invisible but still catches
// clicks, so each dot's click area reaches DOT_RADIUS + weight / 2 = 9 px.
const DOT_RADIUS = 4
const DOT_STYLE = { color: 'transparent', weight: 10, fillColor: 'red', fillOpacity: 0.9 }

// Make a dot reachable with Tab and readable by screen readers
function labelDot(layer, fire, dotToFire) {
  const path = layer.getElement()
  if (!path) return
  dotToFire.set(path, fire)
  path.setAttribute('tabindex', '0')
  path.setAttribute('role', 'button')
  path.setAttribute('aria-label', `Fire detected ${fire.acq_date} ${fire.acq_time_utc} UTC, ${fire.frp} megawatts. Open details.`)
}

// One small red dot per detection. useMemo keeps ~1,800 markers from re-rendering
// when the popup opens or the theme changes.
function FireDots({ fires, onSelect, dotToFire }) {
  return useMemo(
    () =>
      fires.map((fire) => (
        <CircleMarker
          key={fire.id}
          center={[fire.latitude, fire.longitude]}
          radius={DOT_RADIUS}
          pathOptions={DOT_STYLE}
          bubblingMouseEvents={false} // a dot click shouldn't also reach the map (which closes the popup)
          eventHandlers={{
            click: (e) => onSelect(fire, e.target.getElement(), false),
            // Leaflet makes a fresh <path> each time the layer is shown again
            add: (e) => labelDot(e.target, fire, dotToFire),
          }}
        />
      )),
    [fires, onSelect, dotToFire]
  )
}

function HeatLayer({ fires }) {
  const map = useMap()
  useEffect(() => {
    const weights = densityWeights(fires, ZOOM_START, HEAT.radius)
    const layer = L.heatLayer(fires.map((f, i) => [f.latitude, f.longitude, weights[i]]), HEAT).addTo(map)
    return () => map.removeLayer(layer)
  }, [map, fires])
  return null
}

// Clicking empty map closes the popup
function MapClick({ onClick }) {
  useMapEvents({ click: onClick })
  return null
}

// Message over the map while detections load, if they fail, or if there are none.
// Without it all three look the same as a quiet map: an empty basemap.
function MapStatus({ fires, error }) {
  if (error) return <div className="map-control map-status map-status-error" role="alert">Couldn't load fire detections: {error}</div>
  if (!fires) return <div className="map-control map-status" role="status">Loading fire detections…</div>
  if (fires.length === 0) return <div className="map-control map-status" role="status">No fire detections in this time range.</div>
  return null
}

// fires: every detection from /api/fires. error: set if loading them failed.
// safeTop: px of the map covered by the stat cards.
// visible: false while another page is open (the map stays loaded, just hidden).
export default function FireMap({ fires, error, theme, safeTop, visible, onOpenAnalysis }) {
  const [map, setMap] = useState(null)
  const [layer, setLayer] = useState('dots') // 'dots' | 'heat'
  const [selected, setSelected] = useState(null) // { fire, trigger }
  const [zoom, setZoom] = useState(ZOOM_START)
  const [soundOn, setSoundOn] = useState(loadSoundSetting)
  // Refs let select/close read the latest values without being re-created
  // (re-creating them would re-render all ~1,800 dots)
  const selectedRef = useRef(null)
  const soundRef = useRef(soundOn)
  const dotToFire = useMemo(() => new WeakMap(), []) // <path> -> fire, for keyboard use

  // trigger: the dot's element when opened from the keyboard, so focus can go back to it
  const select = useCallback((fire, element, fromKeyboard) => {
    const current = selectedRef.current
    if (current?.fire === fire) return
    if (soundRef.current) playSound(current ? 'switch' : 'open')
    const next = { fire, trigger: current ? current.trigger : fromKeyboard ? element : null }
    selectedRef.current = next
    setSelected(next)
  }, [])

  const close = useCallback(() => {
    const current = selectedRef.current
    if (!current) return
    if (soundRef.current) playSound('close')
    selectedRef.current = null
    setSelected(null)
    if (current.trigger && document.contains(current.trigger)) current.trigger.focus({ preventScroll: true })
  }, [])

  // Keyboard: Enter / Space on a focused fire dot opens it
  useEffect(() => {
    if (!map) return
    const onKey = (e) => {
      const fire = dotToFire.get(e.target)
      if (fire && (e.key === 'Enter' || e.key === ' ')) {
        e.preventDefault()
        select(fire, e.target, true)
      }
    }
    const container = map.getContainer()
    container.addEventListener('keydown', onKey)
    const onZoom = () => setZoom(map.getZoom())
    map.on('zoomend', onZoom)
    return () => {
      container.removeEventListener('keydown', onKey)
      map.off('zoomend', onZoom)
    }
  }, [map, dotToFire, select])

  // Leaflet can't measure a hidden map, so re-measure when the map page is shown again
  useEffect(() => {
    if (visible && map) map.invalidateSize()
  }, [visible, map])

  const showLayer = (next) => {
    setLayer(next)
    if (next === 'heat') close() // the heatmap hides the dots
  }

  const toggleSound = () => {
    const next = !soundOn
    setSoundOn(next)
    soundRef.current = next
    saveSoundSetting(next)
    if (next) playSound('switch')
  }

  return (
    <div className="fire-map">
      <MapContainer
        ref={setMap}
        center={CENTER}
        zoom={ZOOM_START}
        zoomControl={false}
        attributionControl={false} // replaced by the bottom-left one below; bottom right holds our buttons
        className="fire-map-canvas"
      >
        <AttributionControl position="bottomleft" />
        <TileLayer key={theme} url={TILES[theme]} attribution={TILE_ATTRIBUTION[theme]} />
        {fires && layer === 'dots' && <FireDots fires={fires} onSelect={select} dotToFire={dotToFire} />}
        {fires && layer === 'heat' && <HeatLayer fires={fires} />}
        {selected && (
          <CircleMarker
            center={[selected.fire.latitude, selected.fire.longitude]}
            radius={DOT_RADIUS + 5}
            pathOptions={{ className: 'fire-ring', weight: 2, fill: false }}
            interactive={false}
          />
        )}
        <MapClick onClick={close} />
      </MapContainer>

      <MapStatus fires={fires} error={error} />

      {/* Dots / heatmap switch (top right) */}
      <div className="map-control layer-toggle" role="radiogroup" aria-label="Map layer">
        <label>
          <input type="radio" name="map-layer" checked={layer === 'dots'} onChange={() => showLayer('dots')} />
          Fire dots
        </label>
        <label>
          <input type="radio" name="map-layer" checked={layer === 'heat'} onChange={() => showLayer('heat')} />
          Fire density heatmap
        </label>
      </div>

      {/* Sound and zoom buttons (bottom right) */}
      <div className="map-controls-bottom">
        <div className="map-control">
          <button
            type="button"
            className="map-button"
            aria-pressed={soundOn}
            aria-label={soundOn ? 'Sound effects on' : 'Sound effects off'}
            title={soundOn ? 'Sound effects on' : 'Sound effects off'}
            onClick={toggleSound}
          >
            {soundOn ? <Volume2 size={16} /> : <VolumeX size={16} />}
          </button>
        </div>
        <div className="map-control">
          <button type="button" className="map-button" aria-label="Zoom in" disabled={!map || zoom >= map.getMaxZoom()} onClick={() => map.zoomIn()}>
            <Plus size={16} />
          </button>
          <button type="button" className="map-button" aria-label="Zoom out" disabled={!map || zoom <= map.getMinZoom()} onClick={() => map.zoomOut()}>
            <Minus size={16} />
          </button>
        </div>
      </div>

      {map && selected && (
        <FirePopup map={map} fire={selected.fire} safeTop={safeTop} onClose={close} onOpenAnalysis={onOpenAnalysis} />
      )}
    </div>
  )
}
