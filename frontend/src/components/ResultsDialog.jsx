// "Model results" dialog: the fire detector, the spread predictor, and the training data
import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { MODEL, STATS, detectorTotals, scene } from '../lib/model.js'
import { nf, pc0 } from '../lib/format.js'
import { ExImgs } from './MapView.jsx'
import { FireIcon, XMark } from './icons.jsx'

const TABS = [['detector', 'Fire today'], ['spread', "Tomorrow's risk"], ['training', 'Training data']]
const UINT16 = 65535

const Head = ({ eye, title, children }) => <header className="ds-hd"><div><p className="ds-eye">{eye}</p><h2 id="ds-title">{title}</h2><p className="ds-sub">{children}</p></div></header>
const Tile = ({ l, v, s }) => <div className="tile"><span className="tile-l">{l}</span><span className="tile-v">{v}</span><span className="tile-s">{s}</span></div>
const NextStep = () => <p className="ds-note">These are past fires where the correct answer is known. Live CA inputs are already built; connecting the model to them is the next step.</p>

function Detector() {
  const R = MODEL.detector.rows, { L, M, F } = detectorTotals()
  return <>
    <Head eye="Model results · Fire detector" title="Finding fire burning today">Landsat-8 satellite tiles over Sequoia National Forest, CA, on Aug 27, 2020. Each tile covers 7.7 × 7.7 km at 30 m per pixel.</Head>
    <div className="tiles t3">
      <Tile l="Real fire found" v={pc0(F, L)} s={`${nf(F)} of ${nf(L)} fire pixels in these tiles`} />
      <Tile l="Alerts that were real" v={pc0(F, M)} s={`${nf(F)} of ${nf(M)} pixels it flagged`} />
      <Tile l="Overall test score" v="93% · 82%" s="Found · real, on 3,113 unseen tiles" />
    </div>
    <section className="cs" aria-labelledby="ex1">
      <div className="cs-hd"><h3 id="ex1">The model next to the correct answer</h3></div>
      <p className="cs-tk">Left: what the satellite sees, in infrared bands where fire glows. Middle: what the model marked as fire, in red. Right: the correct answer, in orange. They should match closely.</p>
      <div className="exs">{R.map((r, i) => (
        <article className="ex" key={r.patch}>
          <ExImgs k="det" i={i} alts={['Satellite view', 'Model: fire in red', 'Correct answer']} />
          <p className="ex-st"><b>Caught {r.found} of {r.label} fire pixels ({pc0(r.found, r.label)})</b><span>{r.model - r.found} false-alarm pixels · tile {r.patch}</span></p>
        </article>
      ))}</div>
    </section>
    <NextStep />
  </>
}

function Spread() {
  const D = MODEL.spread
  return <>
    <Head eye="Model results · Spread predictor" title="Forecasting tomorrow's fire">64 × 64 km areas in 1 km squares, from past fires in the Next Day Wildfire Spread test data. That data has no locations, so these can’t go on the map.</Head>
    <div className="tiles t3">
      <Tile l="Next-day fire caught" v="52%" s="vs 27% for “tomorrow = today”" />
      <Tile l="Flagged squares that burn" v="1 in 4" s="Leans toward warning too much" />
      <Tile l="Risk cutoff" v={`${Math.round(D.cutoff * 100)}%`} s="Squares above this count as fire" />
    </div>
    <section className="cs" aria-labelledby="ex2">
      <div className="cs-hd"><h3 id="ex2">Today's fire, the forecast, and what burned</h3></div>
      <p className="cs-tk">Left: where fire is burning on day 1. Middle: the model's chance of fire on day 2. Brighter means more likely, and the cyan outline is today's fire, so you can see where it expects growth. Right: what actually burned on day 2.</p>
      <div className="exs">{D.rows.map((r, i) => (
        <article className="ex" key={i}>
          <ExImgs k="spr" i={i} alts={['Day 1: fire today', 'Forecast for day 2', 'Day 2: what burned']} />
          <p className="ex-st"><b>{r.kind}: caught {r.caught} of {r.burned} km² ({pc0(r.caught, r.burned)})</b><span>Fire today {r.today} km² · flagged {r.flagged} km² · burned {r.burned} km²</span></p>
        </article>
      ))}</div>
      <p className="cs-tk">The first three are among the model's best forecasts. The fourth is typical and matches its overall score of about half. In every case it flags a larger area than burns, which is the safer mistake for responders.</p>
    </section>
    <NextStep />
  </>
}

function Training({ hz, onShowFire, setTip }) {
  const [table, setTable] = useState({ fire: false, bands: false })
  const T = STATS[hz]
  if (!T) return <>
    <Head eye="Training data · Flood" title="No flood dataset loaded yet">Run the flood inspection script and its output will show here.</Head>
    <div className="ds-empty"><p>The fire dataset is available. Switch to Fire to see it.</p><button type="button" className="dsb" onClick={onShowFire}><FireIcon /><span>Show fire training data</span></button></div>
  </>
  const px = T.maskShape[0] * T.maskShape[1]
  const peak = Math.max(...T.patches.flatMap((p) => T.masks.map((m) => p[m.k]))), xmax = Math.ceil(peak / 40) * 40
  const ratios = T.patches.flatMap((p) => [p.murphy / p.schroeder, p.kumarRoy / p.schroeder])
  const lo = Math.round((Math.min(...ratios) - 1) * 100), hi = Math.round((Math.max(...ratios) - 1) * 100)
  const sat = T.bands.filter((b) => b.max >= UINT16)
  const pct = (v) => `${((v / px) * 100).toFixed(2)}%`
  const most = (p) => Math.max(p.schroeder, p.murphy, p.kumarRoy)
  const Axis = ({ ticks, max, fmt }) => <div className="pl-axis" aria-hidden="true">{ticks.map((t) => <span key={t} style={{ left: `${(t / max) * 100}%` }}>{fmt(t)}</span>)}</div>
  const tip = (d) => ({ onPointerMove: (e) => setTip({ ...d, x: e.clientX, y: e.clientY }), onPointerLeave: () => setTip(null) })
  const Toggle = ({ k }) => <button type="button" className="tvb" onClick={() => setTable((t) => ({ ...t, [k]: !t[k] }))}>{table[k] ? 'Show chart' : 'Show table'}</button>

  return <>
    <Head eye="Training data · Fire" title={`${T.dataset} sample`}>Read from <code>{T.source}</code>, the output of <code>{T.script}</code>. These describe the dataset the fire model will train on.</Head>
    <p className="ds-note">These describe the data the detector learned from. The model's results are in the Fire today and Tomorrow's risk tabs.</p>
    <div className="tiles">
      <Tile l="Patches inspected" v={<>{T.patches.length}<u>of {T.sampleSize}</u></>} s="The ones with the most fire" />
      <Tile l="Patch size" v={<>{T.imageShape[0]}×{T.imageShape[1]}<u>px</u></>} s={`30 m per pixel, ${T.imageDtype}`} />
      <Tile l="Spectral bands" v={T.imageShape[2]} s="Band 8 (panchromatic) left out" />
      <Tile l="Most fire in one patch" v={pct(peak)} s={`${peak} of ${nf(px)} pixels`} />
    </div>
    <section className="cs" aria-labelledby="cs1">
      <div className="cs-hd"><h3 id="cs1">Fire pixels per patch, by mask algorithm</h3><span className="sp" /><Toggle k="fire" /></div>
      <p className="cs-tk">Murphy and Kumar-Roy mark {lo}–{hi}% more fire pixels than Schroeder in every patch. Even the busiest patch is over 99.7% non-fire, so training has to account for class imbalance.</p>
      {table.fire ? (
        <div className="tbl-wrap"><table className="tbl">
          <thead><tr><th>Patch</th><th className="tl">Scene date</th><th className="tl">Path / row</th>{T.masks.map((m) => <th key={m.k}>{m.name}</th>)}<th>Fire share (max)</th></tr></thead>
          <tbody>{T.patches.map((p) => { const s = scene(p.id); return <tr key={p.id}><td>{s.patch}</td><td className="tl">{s.date}</td><td className="tl">{s.path} / {s.row}</td>{T.masks.map((m) => <td key={m.k}>{p[m.k]}</td>)}<td>{pct(most(p))}</td></tr> })}</tbody>
        </table></div>
      ) : <>
        <div className="key">{T.masks.map((m) => <span key={m.k}><i style={{ background: `var(--${m.s})` }} />{m.name}</span>)}</div>
        <div className="pl" role="img" aria-label="Fire pixels per patch for three mask algorithms. Table view available.">
          {T.patches.map((p) => { const s = scene(p.id); return (
            <div className="pl-row hit" key={p.id} {...tip({ t: `${s.patch} · ${s.date}`, rows: T.masks.map((m) => [m.s, m.name, `${p[m.k]} px (${pct(p[m.k])})`]) })}>
              <div className="pl-lab"><b>{s.patch}</b><small>{s.date} · {s.path}/{s.row}</small></div>
              <div className="pl-plot"><div className="gb">{T.masks.map((m, i) => <div className={`gb-r ${m.s}`} key={m.k}><span className="gb-b" style={{ width: `${(p[m.k] / xmax) * 100}%` }} />{(i === 0 || p[m.k] === most(p)) && <span className="gb-v">{p[m.k]}</span>}</div>)}</div></div>
            </div>
          ) })}
          <Axis ticks={[0, xmax / 4, xmax / 2, (xmax * 3) / 4, xmax]} max={xmax} fmt={(t) => t} />
        </div>
      </>}
    </section>
    <section className="cs" aria-labelledby="cs2">
      <div className="cs-hd"><h3 id="cs2">Pixel values per band</h3><span className="sp" /><Toggle k="bands" /></div>
      <p className="cs-tk">{sat.map((b) => b.name).join(' and ')} hit the 16-bit ceiling of 65,535. Fire is brightest in shortwave infrared, so those pixels saturate. Values are raw, not scaled to reflectance.</p>
      {table.bands ? (
        <div className="tbl-wrap"><table className="tbl">
          <thead><tr><th>Channel</th><th>Landsat band</th><th className="tl">Name</th><th>Min</th><th>Max</th><th>Mean</th></tr></thead>
          <tbody>{T.bands.map((b) => <tr key={b.ch}><td>{b.ch}</td><td>{b.band}</td><td className="tl">{b.name}</td><td>{nf(b.min)}</td><td>{nf(b.max)}{b.max >= UINT16 ? ' (saturated)' : ''}</td><td>{nf(b.mean)}</td></tr>)}</tbody>
        </table></div>
      ) : <>
        <div className="key"><span><i className="k-span" />Min to max</span><span><i className="k-mean" />Mean</span><span><i style={{ background: 'var(--hz-deep)', width: 4, height: 12, borderRadius: 2 }} />Hits the 65,535 ceiling</span></div>
        <div className="pl" role="img" aria-label="Pixel value range per band on the 0 to 65,535 scale. Table view available.">
          {T.bands.map((b) => (
            <div className="pl-row hit" key={b.ch} {...tip({ t: `Band ${b.band} · ${b.name}`, rows: [[null, 'Min', nf(b.min)], [null, 'Mean', nf(b.mean)], [null, 'Max', nf(b.max) + (b.max >= UINT16 ? ', saturated' : '')]] })}>
              <div className="pl-lab">B{b.band} · {b.name}</div>
              <div className="pl-plot"><div className="rg">
                <span className="rg-span" style={{ left: `${(b.min / UINT16) * 100}%`, width: `${((b.max - b.min) / UINT16) * 100}%` }} />
                {b.max >= UINT16 && <span className="rg-sat" style={{ left: '100%' }} />}
                <span className="rg-mean" style={{ left: `${(b.mean / UINT16) * 100}%` }} />
              </div></div>
            </div>
          ))}
          <Axis ticks={[0, 16384, 32768, 49152, UINT16]} max={UINT16} fmt={(t) => (t === UINT16 ? '65,535' : t ? `${t / 1024}k` : '0')} />
        </div>
      </>}
    </section>
  </>
}

function Tip({ tip }) {                                           // hover card for the training charts; flips above the pointer near the bottom edge
  const ref = useRef(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el || !tip) return
    const w = el.offsetWidth, h = el.offsetHeight
    el.style.left = `${Math.min(tip.x + 14, innerWidth - w - 8)}px`
    el.style.top = `${tip.y + 16 + h > innerHeight ? tip.y - h - 12 : tip.y + 16}px`
  })
  if (!tip) return null
  return <div className="tip" role="tooltip" ref={ref}><b>{tip.t}</b>{tip.rows.map(([s, l, v]) => <div key={l}>{s && <i style={{ background: `var(--${s})` }} />}{l}: {v}</div>)}</div>
}

export default function ResultsDialog({ open, hz, onClose, onShowFire }) {
  const dlg = useRef(null)
  const [tab, setTab] = useState('detector')
  const [tip, setTip] = useState(null)

  useEffect(() => {                                               // the native <dialog>: modal, Escape closes it, focus is trapped
    const d = dlg.current
    if (open && !d.open) { d.showModal(); d.scrollTop = 0; d.querySelector('[data-act="close"]')?.focus() }
    else if (!open && d.open) d.close()
  }, [open])

  const pick = (k) => { setTab(k); dlg.current.scrollTop = 0; requestAnimationFrame(() => dlg.current.querySelector(`[data-tab="${k}"]`)?.focus()) }
  const onKey = (e) => {                                          // arrow keys move between tabs
    if (!e.target.closest?.('[role="tab"]') || !['ArrowLeft', 'ArrowRight'].includes(e.key)) return
    e.preventDefault()
    const i = TABS.findIndex(([k]) => k === tab), n = TABS.length
    pick(TABS[(i + (e.key === 'ArrowRight' ? 1 : n - 1)) % n][0])
  }

  return (
    <dialog className="ds" ref={dlg} aria-labelledby="ds-title" data-hz={tab === 'training' ? hz : 'fire'} onClose={() => { setTip(null); onClose() }}
      onClick={(e) => { if (e.target === dlg.current) onClose() }} onKeyDown={onKey} onScroll={() => setTip(null)}>
      {open && <div className="ds-in">
        <div className="ds-bar">
          <div className="tabs" role="tablist" aria-label="Model results">
            {TABS.map(([k, l]) => <button key={k} type="button" role="tab" className="tab" data-tab={k} aria-selected={k === tab} tabIndex={k === tab ? 0 : -1} onClick={() => pick(k)}>{l}</button>)}
          </div>
          <span className="sp" />
          <button type="button" className="thb" data-act="close" aria-label="Close" title="Close" onClick={onClose}><XMark /></button>
        </div>
        <div className="ds-body" role="tabpanel">
          {tab === 'detector' ? <Detector /> : tab === 'spread' ? <Spread /> : <Training hz={hz} onShowFire={onShowFire} setTip={setTip} />}
        </div>
      </div>}
      <Tip tip={tip} />
    </dialog>
  )
}
