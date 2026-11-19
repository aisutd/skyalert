import { useCallback, useEffect, useRef, useState } from 'react'
import AnalyticsPage from './pages/AnalyticsPage.jsx'
import FireMap from './components/FireMap.jsx'
import Sidebar from './components/Sidebar.jsx'
import StatCards from './components/StatCards.jsx'
import { getFires, getSummary } from './lib/api.js'

// Remember the chosen theme between visits. Wrapped in try/catch because some
// browsers (private mode, blocked storage) throw when localStorage is used.
function loadTheme() {
  try {
    return localStorage.getItem('skyalert-theme') === 'dark' ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

export default function App() {
<<<<<<< Updated upstream
  // Which sidebar page is open: "map" or "analytics"
  const [activePage, setActivePage] = useState('map')
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState(null)
  // Every detection, loaded once and shared by the map and the Analytics page
  const [fires, setFires] = useState(null)
  const [firesError, setFiresError] = useState(null)
  const [theme, setTheme] = useState(loadTheme) // light by default
  // Set when "Regional analysis" is clicked in the map's fire popup
  const [analysisFocus, setAnalysisFocus] = useState(null)
  // How far down the map the stat cards reach, so the popup opens below them
  const [safeTop, setSafeTop] = useState(0)
  const mapArea = useRef(null)
  const statCards = useRef(null)
=======
  const [theme, setTheme] = useState(() => store.get('theme') || 'light')   // light unless you've picked dark before; the system theme is ignored
  const [hz, setHzState] = useState('fire')
  const [sel, setSel] = useState(0)
  const [focus, setFocus] = useState(false)                       // map view zoomed to one picked fire instead of the top 5
  const [mode, setModeState] = useState('live')                   // Fire only: 'live' (today), 'model' (the detector on a past fire) or 'spread' (the spread model on a past fire)
  const [mSel, setMSel] = useState(0)
  const [layers, setLayers] = useState({ observed: true, predicted: true, context: true })
  const [mLayers, setMLayers] = useState({ sat: false, label: true, model: true })
  const [spSel, setSpSel] = useState(0)                           // which past fire the spread model is shown on (lib/spreadExamples.js)
  const [spFc, setSpFc] = useState(null)                          // its forecast from /api/spread-forecast?day=, null while loading
  const [hoverDot, setHoverDot] = useState(null)                  // [lat, lon] of the fire dot under the pointer (clickable)
  const [pendingDot, setPendingDot] = useState(null)              // [lat, lon] of the dot you clicked, while it loads
  const [live, setLive] = useState(null)                          // today's fire data, built from /api/fires + /api/fire-context
  const [fires, setFires] = useState([])
  const [modelStatus, setModelStatus] = useState({ ready: false })
  const [phase, setPhase] = useState('loading')                   // loading → fading (loader fades out) → ready
  const [banner, setBanner] = useState(null)
  const [busyText, setBusyText] = useState(null)
  const [announce, setAnnounce] = useState('')
  const [dsOpen, setDsOpen] = useState(false)
  const [map, setMapState] = useState(null)
>>>>>>> Stashed changes

  // Load the stats and the cleaned fire detections from the FastAPI backend once on page load
  // (lib/api.js -> backend/dashboard.py -> main.fetch_fires -> firms_client + data_cleaning)
  useEffect(() => {
    getSummary().then(setSummary).catch((err) => setError(err.message))
    getFires().then(setFires).catch((err) => setFiresError(err.message))
  }, [])

  // Re-measure the stat cards whenever they change size (data loads, window resizes)
  useEffect(() => {
    if (!statCards.current) return
    const measure = () => {
      const cards = statCards.current.getBoundingClientRect()
      const area = mapArea.current.getBoundingClientRect()
      setSafeTop(Math.max(0, cards.bottom - area.top))
    }
<<<<<<< Updated upstream
    const observer = new ResizeObserver(measure)
    observer.observe(statCards.current)
    return () => observer.disconnect()
  }, [])

  // Whenever the theme changes: recolour the page (CSS reads data-theme) and save the choice
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem('skyalert-theme', theme)
    } catch {
      // storage blocked; the theme just won't be remembered next time
    }
  }, [theme])

  // "Regional analysis" in the popup switches to Analytics, focused on that fire
  const openAnalysis = useCallback((focus) => {
    setAnalysisFocus(focus)
    setActivePage('analytics')
  }, [])

  return (
    <div className="app">
      <Sidebar
        activePage={activePage}
        onSelect={(page) => {
          // Picking a page from the sidebar shows everything, not one fire's region
          setAnalysisFocus(null)
          setActivePage(page)
        }}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
      />

      {/* The map stays loaded while other pages are open (just hidden), so
          coming back to it is instant instead of rebuilding the whole map */}
      <main className="map-area" ref={mapArea} hidden={activePage !== 'map'}>
        <FireMap fires={fires} error={firesError} theme={theme} safeTop={safeTop} visible={activePage === 'map'} onOpenAnalysis={openAnalysis} />
        <StatCards summary={summary} error={error} ref={statCards} />
      </main>

      {activePage === 'analytics' && (
        <AnalyticsPage fires={fires} error={firesError} focus={analysisFocus} onClearFocus={() => setAnalysisFocus(null)} />
      )}
    </div>
  )
=======
    const onClick = (e) => {                                      // a fire dot (nearest within 14 px) or any other spot joins the list, as in the prototype
      if (busy.current || !dotsOn()) return
      const near = nearestDot(e.containerPoint)
      if (view.current === 'ov' && !near) return                  // in the overview's small map, only fire dots react
      pick.current = near ? clusterAround(S.current.fires, near) : { latlng: [e.latlng.lat, e.latlng.lng], clickedSpot: true }
      setHoverDot(null)
      setFocus(false)
      if (view.current === 'ov') go('mv')                         // a dot clicked in the overview opens the map view on it
      loadPick()
    }
    let raf = 0, last = null
    const onMouseMove = (e) => {                                  // pointer + ring when a dot is within reach
      last = e.containerPoint
      if (raf) return
      raf = requestAnimationFrame(() => {
        raf = 0
        const near = !busy.current && dotsOn() && !map.dragging?.moving?.() ? nearestDot(last) : null
        map.getContainer().classList.toggle('dot-hover', !!near)
        setHoverDot((h) => (near ? (h && h[0] === near.latitude && h[1] === near.longitude ? h : [near.latitude, near.longitude]) : null))
      })
    }
    const onMouseOut = () => { map.getContainer().classList.remove('dot-hover'); setHoverDot(null) }
    const forget = () => { dotPts.current = null }                // dots moved on screen: re-project on the next hover
    map.on('moveend', onMoveEnd)
    map.on('click', onClick)
    map.on('mousemove', onMouseMove)
    map.on('mouseout', onMouseOut)
    map.on('move zoom resize viewreset', forget)
    return () => {
      map.off('moveend', onMoveEnd); map.off('click', onClick); map.off('mousemove', onMouseMove); map.off('mouseout', onMouseOut)
      map.off('move zoom resize viewreset', forget); clearTimeout(areaT.current); cancelAnimationFrame(raf)
    }
  }, [map])                                                       // eslint-disable-line react-hooks/exhaustive-deps

  // ---------- the spread model on a past fire ----------
  const spSeq = useRef(0)
  function loadSpread(i) {                                        // the past fire's forecast, and what burned the next day
    const seq = ++spSeq.current, ex = SPREAD_EXAMPLES[i]
    setSpFc(null)
    setBusyText('Running the spread model on a past fire…')
    getForecast(ex.lat, ex.lon, ex.day).then((fc) => {
      if (seq !== spSeq.current) return                           // another fire was picked meanwhile
      setBusyText(null)
      flushSync(() => setSpFc(fc))
      if (S.current.mode === 'spread' && view.current === 'mv') fitView(true)
      setAnnounce(fc.status === 'ok' && fc.check
        ? `${ex.label}: the model forecast ${fc.check.new_fire_caught_pixels} of ${fc.check.new_fire_pixels} pixels of new fire.`
        : `${ex.label}: ${fc.note || 'forecast unavailable'}.`)
    })
  }

  // ---------- actions ----------
  const act = {
    setHz(h) {
      if (busy.current || h === S.current.hz) return
      flushSync(() => { setHzState(h); setSel(0); setModeState('live'); setFocus(false) })
      const lay = view.current === 'ov' ? ov.current : mv.current
      lay.classList.remove('is-swapping'); void lay.offsetWidth; lay.classList.add('is-swapping')   // re-render fade after a fire / flood switch
      setTimeout(() => lay.classList.remove('is-swapping'), 500)
      fitView(true)
      setAnnounce(`${h === 'fire' ? 'Fire' : 'Flood'} mode.`)
    },
    toggleTheme() { setTheme((t) => { const n = t === 'dark' ? 'light' : 'dark'; store.set('theme', n); return n }) },
    openData() { setDsOpen(true) },
    focusMap() { go('mv') },
    back() { go('ov') },
    openModel() { if (busy.current) return; flushSync(() => setModeState('spread')); loadSpread(S.current.spSel); go('mv') },   // the detector tab is gone, so this opens the spread model on a past fire
    setMode(m) {
      if (busy.current || m === S.current.mode) return
      flushSync(() => { setModeState(m); setFocus(false) })
      if (m === 'spread') loadSpread(S.current.spSel)
      fitView(true)
      setAnnounce(m === 'model' ? 'Showing the fire detector on a past fire, Sequoia National Forest, August 2020.'
        : m === 'spread' ? `Showing the spread model on a past fire, ${SPREAD_EXAMPLES[S.current.spSel].label}.` : "Showing today's live fires.")
    },
    selectSpread(i) {
      if (busy.current || i === S.current.spSel) return
      flushSync(() => setSpSel(i))
      loadSpread(i)
    },
    zoomToForecast() {                                            // fly in close on the selected fire's forecast squares (burning, likely, possible)
      const it = S.current.D.items[S.current.sel], m = mapRef.current
      if (busy.current || !m || !it || !it.fc || it.fc.status !== 'ok') return
      const { tl, br } = pads()
      ownMove.current = true                                      // the fire list stays as it is for this move
      clearTimeout(ownMoveT.current)
      ownMoveT.current = setTimeout(() => { ownMove.current = false }, 2000)
      m.once('moveend', () => { clearTimeout(ownMoveT.current); ownMoveT.current = setTimeout(() => { ownMove.current = false }, 80) })
      m.flyToBounds(forecastFocus(it.fc).bounds, { paddingTopLeft: [tl[0], tl[1] + 44], paddingBottomRight: br, maxZoom: 14, duration: REDUCE.matches ? 0 : 0.85 })   // + room above for the marker's stem
    },
    selectTile(i) {
      if (busy.current) return
      setMSel(i)
      const r = MODEL.detector.rows[i]
      setAnnounce(`Tile ${i + 1}: caught ${r.found} of ${r.label} fire pixels.`)
    },
    select(i) {
      if (busy.current) return
      const { D, sel, hz, mode, focus } = S.current
      const flyTo = view.current === 'mv' && !(hz === 'fire' && mode === 'model')   // in the map view, picking a fire flies to it
      if (i === sel) { if (flyTo) { flushSync(() => setFocus(!focus)); fitView(true) } return }   // same fire again: toggle between it and the top 5
      const prev = D.items[sel], it = D.items[i]
      flushSync(() => { setSel(i); if (flyTo) setFocus(true) })
      if (flyTo) fitView(true)
      if (view.current === 'ov' && !REDUCE.matches) countUp(ov.current.querySelector('.ov-num .fn'), prev.prob, it.prob, 420)
      if (view.current === 'mv') {
        rollCoords(mv.current.querySelector('.selp-id'), prev.id, it.id, 600)
        const fwi = mv.current.querySelector('.fwi-n')                // fire-weather index counts from the last fire's score
        if (fwi && Number.isFinite(prev.fwi) && !REDUCE.matches) countUp(fwi, prev.fwi, it.fwi, 600)
      }
      setAnnounce(`${it.name} selected: ${it.prob} percent, ${RISK[it.risk]} risk.`)
    },
    setLayer(k, v) { setLayers((l) => ({ ...l, [k]: v })) },
    setMLayer(k, v) { setMLayers((l) => ({ ...l, [k]: v })) },
    zoomIn() { mapRef.current?.zoomIn() },
    zoomOut() { mapRef.current?.zoomOut() },
  }
  const actRef = useRef(act); actRef.current = act

  useEffect(() => {                                               // Escape leaves the map view (the dialog handles its own keys)
    const onKey = (e) => { if (e.key === 'Escape' && view.current === 'mv' && !document.querySelector('dialog[open]')) actRef.current.back() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [])

  const s = { D, hz, sel, focus, mode, mSel, layers, mLayers, theme, fires, modelStatus, spSel, spFc, hoverDot, pendingDot }
  return <>
    {shown && (
      <div className="app" ref={app} data-hz={hz}>
        <MapStage s={s} act={act} setMap={setMap} />
        <Overview ref={ov} s={s} act={act} />
        <MapView ref={mv} s={s} act={act} />
        <div className="flyer" ref={fly} aria-hidden="true" hidden><span className="fn" /><span className="pct" /></div>
        <div className="sr-only" aria-live="polite">{announce}</div>
        {busyText && <div className="busy-dot" role="status">{busyText}</div>}
      </div>
    )}
    <ResultsDialog open={dsOpen} hz={hz} onClose={() => { setDsOpen(false); document.querySelector(`.layer:not([hidden]) [data-act="data"]`)?.focus({ preventScroll: true }) }}
      onShowFire={() => { setDsOpen(false); actRef.current.setHz('fire'); setTimeout(() => setDsOpen(true)) }} />
    {phase !== 'ready' && <Loader out={phase === 'fading'} />}
    {banner && <div className="banner" role="status">Live data isn't available ({banner}), so this is sample data. Start the backend with <code>python -m backend.dashboard</code>, then reload.</div>}
  </>
>>>>>>> Stashed changes
}
