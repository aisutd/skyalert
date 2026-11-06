// SkyAlert: the overview (concept 07) and the map view (concept 05), with one map underneath both.
// State lives here; the overview ⇄ map morph works on the rendered DOM (measure, then animate), like the prototype did.
import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import L from 'leaflet'
import Overview from './components/Overview.jsx'
import MapView from './components/MapView.jsx'
import MapStage from './components/MapStage.jsx'
import ResultsDialog from './components/ResultsDialog.jsx'
import Loader from './components/Loader.jsx'
import { RISK, REDUCE, countUp, rollCoords, store } from './lib/format.js'
import { SAMPLE, sketchBounds, sketchToLatLng } from './lib/sample.js'
import { MODEL, modelBounds } from './lib/model.js'
import { CA, buildLive } from './lib/live.js'
import { getFires, getModelStatus } from './lib/api.js'

const TOP_N = 5                                                   // the map view zooms to the 5 highest-index fires; the list still shows all of them
const MOVE = 740                                                  // ms the shared pieces take to fly between layouts
const FULL = 'inset(0px 0px 0px 0px round 0px)'
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

export default function App() {
  const [theme, setTheme] = useState(() => store.get('theme') || 'light')   // light unless you've picked dark before; the system theme is ignored
  const [hz, setHzState] = useState('fire')
  const [sel, setSel] = useState(0)
  const [focus, setFocus] = useState(false)                       // map view zoomed to one picked fire instead of the top 5
  const [mode, setModeState] = useState('live')                   // Fire only: 'live' (today) or 'model' (the detector on a past fire)
  const [mSel, setMSel] = useState(0)
  const [layers, setLayers] = useState({ observed: true, predicted: true, context: true })
  const [mLayers, setMLayers] = useState({ sat: false, label: true, model: true })
  const [live, setLive] = useState(null)                          // today's fire data, built from /api/fires + /api/fire-context
  const [fires, setFires] = useState([])
  const [modelStatus, setModelStatus] = useState({ ready: false })
  const [phase, setPhase] = useState('loading')                   // loading → fading (loader fades out) → ready
  const [banner, setBanner] = useState(null)
  const [busyText, setBusyText] = useState(null)
  const [announce, setAnnounce] = useState('')
  const [dsOpen, setDsOpen] = useState(false)
  const [map, setMapState] = useState(null)

  const app = useRef(null), ov = useRef(null), mv = useRef(null), fly = useRef(null), mapRef = useRef(null)
  const view = useRef('ov'), busy = useRef(false), liveSeq = useRef(0), flyT = useRef(0)
  const D = hz === 'fire' ? live || SAMPLE.fire : SAMPLE.flood
  const S = useRef({}); S.current = { hz, sel, focus, mode, D, fires, modelStatus }   // latest state for the imperative code below
  const setMap = (m) => { mapRef.current = m; setMapState(m) }

  useLayoutEffect(() => { document.documentElement.dataset.theme = theme }, [theme])

  // ---------- where the map should look ----------
  const rel = (el) => { const a = app.current.getBoundingClientRect(), r = el.getBoundingClientRect(); return { x: r.left - a.left, y: r.top - a.top, w: r.width, h: r.height } }
  const clipOf = (r) => `inset(${r.y}px ${app.current.clientWidth - r.x - r.w}px ${app.current.clientHeight - r.y - r.h}px ${r.x}px round 14px)`
  const slot = () => ov.current.querySelector('.ov-slot')
  function bounds() {
    const { hz, mode, D, sel, focus } = S.current
    if (hz === 'fire' && mode === 'model') return L.latLngBounds(modelBounds())
    const it = D.items[sel]
    if (view.current === 'mv' && focus) return L.latLng(it.latlng || sketchToLatLng(hz, it.pos)).toBounds(30000)   // a fire picked from the list: ~30 km around it
    if (D.live) return view.current === 'mv' && D.items.length ? L.latLngBounds(D.items.slice(0, TOP_N).map((x) => x.latlng)).pad(0.15) : L.latLngBounds(CA)
    return L.latLngBounds(sketchBounds(hz))
  }
  function pads() {                                               // the part of the map the current layout leaves visible
    const W = app.current.clientWidth, H = app.current.clientHeight
    if (view.current === 'ov') { const r = rel(slot()); return { tl: [r.x + 12, r.y + 12], br: [W - r.x - r.w + 12, H - r.y - r.h + 12] } }
    const q = (c) => mv.current.querySelector(c), a = rel(q('.fp1')), b = rel(q('.fp2')), c = rel(q('.fp3'))
    if (c.w > 0) { const left = a.x + a.w + 20, right = W - b.x + 20; return { tl: [W - left - right > 260 ? left : 24, 24], br: [right, H - c.y + 20] } }   // desktop: between the three panels
    return { tl: [12, a.y + a.h + 10], br: [12, H - b.y + 10] }   // phone: between the top and bottom panels
  }
  function fitView(animate) {
    const m = mapRef.current
    if (!m) return
    const { tl, br } = pads(), { hz, mode, D, focus } = S.current
    m.invalidateSize(false)
    const o = { paddingTopLeft: tl, paddingBottomRight: br, ...(view.current === 'mv' && !(hz === 'fire' && mode === 'model') && (D.live || focus) ? { maxZoom: 9 } : {}) }   // cap zoom so close-together fires don't zoom to street level
    if (animate && !REDUCE.matches) {
      const el = m.getContainer(), land = () => { clearTimeout(flyT.current); el.classList.remove('is-flying') }
      el.classList.add('is-flying')                               // the dot canvas is CSS-scaled mid-flight, so it hides until the flight ends
      clearTimeout(flyT.current); flyT.current = setTimeout(land, 1500)   // safety net if moveend never comes
      m.once('moveend', land)
      m.flyToBounds(bounds(), { ...o, duration: 0.85, easeLinearity: 0.4 })
    } else m.fitBounds(bounds(), { ...o, animate: false })
  }
  function layoutMap() {
    const m = mapRef.current
    if (!m || !ov.current) return
    m.getContainer().style.clipPath = view.current === 'ov' ? clipOf(rel(slot())) : 'none'
    fitView(false)
  }

  // ---------- the overview ⇄ map view morph ----------
  async function go(to) {
    if (busy.current || to === view.current) return
    if (to === 'ov' && S.current.mode === 'model') flushSync(() => setModeState('live'))   // the overview always describes today's live map
    busy.current = true
    const toMap = to === 'mv', from = toMap ? ov.current : mv.current, next = toMap ? mv.current : ov.current, mapEl = mapRef.current?.getContainer()
    const k = REDUCE.matches ? 0.001 : 1, A = []
    const anim = (el, kf, o) => { const a = el.animate(kf, { duration: o.d * k, delay: (o.delay || 0) * k, easing: o.e || 'cubic-bezier(.65, 0, .35, 1)', fill: 'both' }); A.push(a); return a }
    const visible = (r) => r && r.w > 0 && r.h > 0
    const textEl = (el) => { const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, { acceptNode: (t) => t.textContent.trim() ? 1 : 3 }); return w.nextNode()?.parentElement || el }   // the first element that actually holds text
    const fontPx = (el) => parseFloat(getComputedStyle(textEl(el)).fontSize)
    const pieces = (layer) => [...layer.querySelectorAll('[data-k], .pc')].filter((el) => !el.parentElement.closest('[data-k], .pc'))   // leaf pieces only: a child moves with its parent
    const numStyle = (el) => { const cs = getComputedStyle(el), p = getComputedStyle(el.querySelector('.pct')); return { r: rel(el), v: +el.querySelector('.fn').textContent, fs: cs.fontSize, fw: cs.fontWeight, ls: cs.letterSpacing, c: cs.color, pfs: p.fontSize, pml: p.marginLeft, pc: p.color, pfw: p.fontWeight, pls: p.letterSpacing } }
    from.inert = true

    // 1. measure the layout we are leaving
    const oldP = pieces(from), oldR = new Map(oldP.map((el) => [el, rel(el)]))
    const oldNum = from.querySelector('.num'), oN = numStyle(oldNum)
    const slotR = toMap ? rel(slot()) : null

    // 2. show the new layout underneath the old one and measure it
    next.hidden = false
    view.current = to
    flushSync(() => setFocus(false))                              // opening the map view shows the top 5 again
    const newP = pieces(next), newR = new Map(newP.map((el) => [el, rel(el)]))
    const newNum = next.querySelector('.num'), nN = numStyle(newNum)
    const oldByK = new Map(oldP.filter((el) => el.dataset.k && visible(oldR.get(el))).map((el) => [el.dataset.k, el]))
    const panels = [...mv.current.querySelectorAll('.fp')].map((fp) => ({ bg: fp.querySelector('.pbg'), f: rel(fp), a: rel(fp.querySelector(fp.dataset.anchor)) }))
    const slotTo = toMap ? null : rel(slot())
    mapRef.current?.scrollWheelZoom[toMap ? 'enable' : 'disable']()
    fitView(true)

    const lead = toMap ? 0 : 150                                  // going back, the panels fold up first
    const sharedOld = new Set(), sharedNew = new Set()

    // 3. pieces in both layouts fly from the old spot to the new one, cross-fading on the way
    for (const el of newP) {
      const o = el.dataset.k && oldByK.get(el.dataset.k), n = newR.get(el)
      if (!o || !visible(n)) continue
      const r = oldR.get(o), s = fontPx(o) / fontPx(el)   // scale by text size, not box height: a piece that wraps to 2 rows in one layout would otherwise blow up 2x
      sharedOld.add(o); sharedNew.add(el)
      anim(el, [{ transform: `translate(${r.x - n.x}px, ${r.y - n.y}px) scale(${s})`, opacity: 0 }, { opacity: 0, offset: 0.3 }, { opacity: 1, offset: 0.62 }, { transform: 'none', opacity: 1 }], { d: MOVE, delay: lead })
      anim(o, [{ transform: 'none', opacity: 1 }, { opacity: 1, offset: 0.12 }, { opacity: 0, offset: 0.42 }, { transform: `translate(${n.x - r.x}px, ${n.y - r.y}px) scale(${1 / s})`, opacity: 0 }], { d: MOVE, delay: lead })
    }

    // 4. pieces only in the old layout fade out; pieces only in the new one fade in after the move
    oldP.filter((el) => !sharedOld.has(el)).forEach((el) => anim(el, [{ opacity: 1 }, { opacity: 0 }], { d: 200, e: 'ease' }))
    newP.filter((el) => !sharedNew.has(el) && visible(newR.get(el)))
      .sort((a, b) => newR.get(a).y - newR.get(b).y)
      .forEach((el, i) => anim(el, [{ opacity: 0, transform: 'translateY(8px)' }, { opacity: 1, transform: 'none' }], { d: 360, delay: (toMap ? 640 : 560) + Math.min(i * 22, 260), e: 'cubic-bezier(.2, .7, .2, 1)' }))

    // 5. the big number shrinks (or grows) into place and counts to its new value
    oldNum.style.visibility = newNum.style.visibility = 'hidden'
    const F = fly.current, fn = F.querySelector('.fn'), fp = F.querySelector('.pct')
    fp.textContent = next.querySelector('.num .pct').textContent
    F.hidden = false
    anim(F, [
      { transform: `translate(${oN.r.x}px, ${oN.r.y}px)`, fontSize: oN.fs, fontWeight: oN.fw, letterSpacing: oN.ls, color: oN.c },
      { transform: `translate(${nN.r.x}px, ${nN.r.y}px)`, fontSize: nN.fs, fontWeight: nN.fw, letterSpacing: nN.ls, color: nN.c },
    ], { d: MOVE + 60, delay: lead })
    anim(fp, [{ fontSize: oN.pfs, marginLeft: oN.pml, color: oN.pc, fontWeight: oN.pfw, letterSpacing: oN.pls }, { fontSize: nN.pfs, marginLeft: nN.pml, color: nN.pc, fontWeight: nN.pfw, letterSpacing: nN.pls }], { d: MOVE + 60, delay: lead })   // weight + spacing too, so "/100" lands without a jump
    countUp(fn, oN.v, nN.v, (MOVE + 60) * k, lead * k)

    // 6. each panel's box grows out from the data that just landed in it (and folds back into it on the way out)
    panels.forEach(({ bg, f, a }, i) => {
      const p = 10
      if (!visible(f) || !visible(a)) return
      const small = { top: `${a.y - f.y - p}px`, left: `${a.x - f.x - p}px`, right: `${f.x + f.w - a.x - a.w - p}px`, bottom: `${f.y + f.h - a.y - a.h - p}px`, borderRadius: '10px' }
      const full = { top: '0px', left: '0px', right: '0px', bottom: '0px', borderRadius: '16px' }
      if (toMap) anim(bg, [{ ...small, opacity: 0 }, { ...small, opacity: 1, offset: 0.2 }, { ...full, opacity: 1 }], { d: 580, delay: 340 + i * 60 })
      else anim(bg, [{ ...full, opacity: 1 }, { ...small, opacity: 1, offset: 0.7 }, { ...small, opacity: 0 }], { d: 380 })
    })

    // 7. the map opens out of its slot (or closes back into it) and re-centres
    if (mapEl) anim(mapEl, toMap ? [{ clipPath: clipOf(slotR) }, { clipPath: FULL }] : [{ clipPath: FULL }, { clipPath: clipOf(slotTo) }], { d: MOVE + 80, delay: lead })

    await Promise.all(A.map((a) => a.finished)).catch(() => {})
    from.hidden = true
    next.inert = false
    if (mapEl) mapEl.style.clipPath = toMap ? 'none' : clipOf(slotTo)
    oldNum.style.visibility = newNum.style.visibility = ''
    F.hidden = true
    A.forEach((a) => a.cancel())
    busy.current = false
    next.querySelector('[data-k="cta"]')?.focus({ preventScroll: true })
    const d = S.current.D, it = d.items[S.current.sel]
    setAnnounce(toMap ? `Map view. ${d.region}: ${d.prob} percent ${d.regionLabel}, ${RISK[d.risk]} risk.` : `Overview. ${it.name}: ${it.prob} percent.`)
  }

  // ---------- live data ----------
  async function refreshLive(selIdx = 0) {
    const seq = ++liveSeq.current
    setBusyText((t) => (t && t.startsWith('Running') ? t : 'Updating fires…'))
    const d = await buildLive({ fires: S.current.fires, bounds: CA, modelStatus: S.current.modelStatus, withModel: true })
    if (seq !== liveSeq.current) return                           // a newer click already replaced this one
    while (busy.current) await sleep(80)                          // never rebuild the layout mid-transition
    setBusyText(null)
    const ci = d.items.findIndex((x) => x.clicked)
    setLive(d)
    if (S.current.hz !== 'fire') return
    const i = ci >= 0 ? ci : Math.min(selIdx, d.items.length - 1)
    setSel(i)
    setAnnounce(`${d.items.length} spots shown. ${d.items[i].name}: fire-weather index ${d.items[i].prob}, ${RISK[d.items[i].risk]}.`)
  }

  const started = useRef(false)
  useEffect(() => {                                               // first load: today's fires, then weather for the strongest; the model fills in after
    if (started.current) return
    started.current = true
    ;(async () => {
      let ms = { ready: false }, ok = false
      try {
        const f = await getFires()
        ms = await getModelStatus()
        S.current.fires = f; S.current.modelStatus = ms
        const d = await buildLive({ fires: f, bounds: CA, modelStatus: ms, withModel: false })
        setFires(f); setModelStatus(ms); setLive(d)
        ok = true
      } catch (err) {
        setBanner(err.message || String(err))
      }
      setPhase('fading')
      setTimeout(() => setPhase('ready'), 400)
      if (ok && ms.ready) { setBusyText("Running the spread model on today's data…"); refreshLive(0) }
    })()
  }, [])

  // ---------- first layout, and keeping the map fitted ----------
  const shown = phase !== 'loading'
  useLayoutEffect(() => { if (shown && mv.current && view.current === 'ov') { mv.current.hidden = true; mv.current.inert = true } }, [shown])
  useEffect(() => {
    if (!shown || !map) return
    layoutMap()
    document.fonts?.ready.then(() => { if (!busy.current) layoutMap() })
    let raf = 0
    const ro = new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => { if (!busy.current) layoutMap() }) })
    ro.observe(app.current)
    return () => { ro.disconnect(); cancelAnimationFrame(raf) }
  }, [shown, map])                                                // eslint-disable-line react-hooks/exhaustive-deps

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
    openModel() { if (busy.current) return; flushSync(() => setModeState('model')); go('mv') },
    setMode(m) {
      if (busy.current || m === S.current.mode) return
      flushSync(() => { setModeState(m); setFocus(false) })
      fitView(true)
      setAnnounce(m === 'model' ? 'Showing the fire detector on a past fire, Sequoia National Forest, August 2020.' : "Showing today's live fires.")
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

  const s = { D, hz, sel, focus, mode, mSel, layers, mLayers, theme, fires, modelStatus }
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
}
