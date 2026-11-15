// One React Leaflet map under both layouts. The overview clips it to its slot, so the map view only widens the clip.
import { Fragment, memo, useMemo } from 'react'
import L from 'leaflet'
import { CircleMarker, ImageOverlay, MapContainer, Marker, Pane, Rectangle, SVGOverlay, TileLayer } from 'react-leaflet'
import { RISK } from '../lib/format.js'
import { MODEL } from '../lib/model.js'
import { US, heatURL } from '../lib/live.js'
import { POINTS, WIND, rasterCells, sketchBounds, sketchToLatLng } from '../lib/sample.js'
import { drawPast } from '../lib/spreadExamples.js'
import { TILES } from './bits.jsx'

const FireDots = memo(function FireDots({ fires }) {             // today's FIRMS detections; the map draws them on one canvas
  return fires.map((f, i) => <CircleMarker key={i} center={[f.latitude, f.longitude]} radius={3.5} pathOptions={{ color: '#FFFFFF', weight: 0.8, fillColor: '#E8452C', fillOpacity: 0.9 }} interactive={false} />)
})

function NumberedMarkers({ D, hz, sel, focus, onSelect }) {       // 1, 2, 3… with the place name beside each
  return D.items.map((it, i) => {
    const on = i === sel
    const icon = L.divIcon({
      className: 'lm-mk' + (on ? ' on' : '') + (on && focus ? ' lm-focus' : ''),
      iconSize: [24, 24], iconAnchor: [12, 12],
      html: `<span class="lm-k">${i + 1}</span><span class="lm-lab">${it.name.replace(/</g, '&lt;')}</span>`,
    })
    return <Marker key={`${hz}-${i}`} position={it.latlng || sketchToLatLng(hz, it.pos)} icon={icon} title={`${it.name}, ${RISK[it.risk]} ${D.live ? 'fire-weather' : 'predicted'} risk`} zIndexOffset={on ? 500 : 0} eventHandlers={{ click: () => onSelect(i) }} />
  })
}

function SampleRaster({ hz, layers }) {                          // the sample "model output" grid, wind arrows, and observation points
  const cells = useMemo(() => rasterCells(hz), [hz])
  return <>
    <SVGOverlay bounds={sketchBounds(hz)} attributes={{ viewBox: '0 0 400 260' }} interactive={false}>
      {layers.predicted && cells.map(({ x, y, lv }) => <rect key={`${x}-${y}`} className={`m-cell l${lv}`} x={x} y={y} width="12.4" height="12.4" />)}
      {layers.context && hz === 'fire' && WIND.map(([x, y]) => <path key={`${x}-${y}`} className="m-wind" d={`M${x - 7} ${y - 5}L${x + 7} ${y + 5}M${x + 7} ${y + 5}l-5.5 0.4M${x + 7} ${y + 5}l-2.2 -5`} strokeWidth="1.2" />)}
    </SVGOverlay>
    {layers.observed && POINTS[hz].map((p, i) => hz === 'fire'
      ? <CircleMarker key={`p${i}`} center={sketchToLatLng(hz, p)} radius={4} pathOptions={{ color: '#25272C', weight: 1, fillColor: '#F5823A', fillOpacity: 1 }} interactive={false} />
      : <Marker key={`g${i}`} position={sketchToLatLng(hz, p)} icon={L.divIcon({ className: 'lm-gauge', iconSize: [11, 11] })} interactive={false} keyboard={false} />)}
  </>
}

function ModelTiles({ mSel, mLayers, onSelectTile }) {           // the detector's output, drawn where the tiles really are
  return MODEL.detector.rows.map((r, i) => {
    const n = i + 1, on = i === mSel
    return (
      <Fragment key={r.patch}>
        {mLayers.sat && <ImageOverlay url={`/model_results/det_${n}_base256.png`} bounds={r.bounds} className="m-px" />}
        {mLayers.label && <ImageOverlay url={`/model_results/det_${n}_label_mask.png`} bounds={r.bounds} className="m-px" />}
        {mLayers.model && <ImageOverlay url={`/model_results/det_${n}_model_mask.png`} bounds={r.bounds} className="m-px" opacity={mLayers.label ? 0.85 : 1} />}
        <Rectangle bounds={r.bounds} pathOptions={{ color: on ? '#25272C' : '#FFFFFF', weight: on ? 3 : 1.5, fill: true, fillOpacity: 0, dashArray: on ? null : '4 4' }} eventHandlers={{ click: () => onSelectTile(i) }} />
        <Marker position={[r.bounds[1][0], r.bounds[0][1]]} icon={L.divIcon({ className: '', iconSize: [22, 22], iconAnchor: [-4, -4], html: `<span class="m-tile-k${on ? ' on' : ''}">${n}</span>` })} title={`Tile ${n}: caught ${r.found} of ${r.label} fire pixels`} eventHandlers={{ click: () => onSelectTile(i) }} />
      </Fragment>
    )
  })
}

const LIVE_LAYERS = { today: true, forecast: true, actual: false }
function SelectedForecast({ fc, theme }) {                        // the selected live fire's forecast, drawn pixel by pixel
  const url = useMemo(() => drawPast(fc, LIVE_LAYERS, theme), [fc, theme])
  const b = [[fc.bounds.south, fc.bounds.west], [fc.bounds.north, fc.bounds.east]]
  return <>
    <ImageOverlay url={url} bounds={b} className="m-px" interactive={false} />
    <Rectangle bounds={b} pathOptions={{ color: theme === 'dark' ? '#FFFFFF' : '#25272C', weight: 1, fill: false, dashArray: '4 4', opacity: 0.7 }} interactive={false} />
  </>
}

function SpreadPast({ fc, layers, theme }) {                     // the spread model's forecast on a past fire, next to what burned
  const url = useMemo(() => (fc && fc.status === 'ok' ? drawPast(fc, layers, theme) : null), [fc, layers, theme])
  if (!url) return null
  const b = [[fc.bounds.south, fc.bounds.west], [fc.bounds.north, fc.bounds.east]]
  return <>
    <ImageOverlay url={url} bounds={b} className="m-px" interactive={false} />
    <Rectangle bounds={b} pathOptions={{ color: theme === 'dark' ? '#FFFFFF' : '#25272C', weight: 1.5, fill: false, dashArray: '5 4' }} interactive={false} />
  </>
}

const RING = L.divIcon({ className: 'dot-ring', iconSize: [20, 20], iconAnchor: [10, 10] })
const RING_PENDING = L.divIcon({ className: 'dot-ring pending', iconSize: [20, 20], iconAnchor: [10, 10] })

export default function MapStage({ s, act, setMap }) {   // attribution is our own .attr line, so Leaflet's is off
  const { D, hz, mode, layers } = s
  const modelMode = hz === 'fire' && mode === 'model', spreadMode = hz === 'fire' && mode === 'spread'
  return (
    <MapContainer className="lmap" ref={setMap} bounds={US} zoomControl={false} attributionControl={false} scrollWheelZoom={false} zoomSnap={0.25} zoomDelta={0.5} preferCanvas role="region" aria-label="Map of fires and fire-weather risk">
      <TileLayer key={s.theme} url={TILES[s.theme].url} maxZoom={16} />
      {spreadMode ? <SpreadPast fc={s.spFc} layers={s.spLayers} theme={s.theme} />
        : modelMode ? <ModelTiles mSel={s.mSel} mLayers={s.mLayers} onSelectTile={act.selectTile} /> : <>
        {D.live && layers.predicted && D.modelReady && D.items.map((it, i) => {
          if (!it.fc || it.fc.status !== 'ok') return null
          const b = it.fc.bounds, on = i === s.sel
          // The selected fire: crisp pixels (likely, possible, its fire outlined), like the past-fire view; others: the heat map
          // The selected forecast sits above the numbered markers (pane z 650 > markers 600), so a marker can't hide it
          return on ? <Pane key={`fc${i}`} name="fc-selected" style={{ zIndex: 650, pointerEvents: 'none' }}><SelectedForecast fc={it.fc} theme={s.theme} /></Pane>
            : <ImageOverlay key={`fc${i}`} url={heatURL(it.fc)} bounds={[[b.south, b.west], [b.north, b.east]]} opacity={0.9} interactive={false} />
        })}
        {D.live && layers.observed && <FireDots fires={s.fires} />}
        {/* The dot under the pointer (it can be clicked) and the one you just clicked while it loads */}
        {D.live && s.hoverDot && <Marker position={s.hoverDot} icon={RING} interactive={false} keyboard={false} />}
        {D.live && s.pendingDot && <Marker position={s.pendingDot} icon={RING_PENDING} interactive={false} keyboard={false} zIndexOffset={1000} />}
        {!D.live && <SampleRaster hz={hz} layers={layers} />}
        <NumberedMarkers D={D} hz={hz} sel={s.sel} focus={s.focus} onSelect={act.select} />
      </>}
    </MapContainer>
  )
}
