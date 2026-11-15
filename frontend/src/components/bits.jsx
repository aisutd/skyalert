// Small pieces shared by the overview and the map view
import { useLayoutEffect, useRef } from 'react'
import { RISK, f1, stopAnim } from '../lib/format.js'
import { FireIcon, FloodIcon, Logo, Moon, Sun } from './icons.jsx'

export const Brand = () => <span className="br"><Logo />SkyAlert</span>

export function HazardToggle({ hz, onSet }) {                     // the indicator slides because the same element stays on screen
  const onKey = (e) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return
    e.preventDefault()
    const to = hz === 'fire' ? 'flood' : 'fire'
    onSet(to)
    e.currentTarget.querySelector(`[data-set-hz="${to}"]`)?.focus()
  }
  return (
    <div className="hzt" role="radiogroup" aria-label="Hazard" data-hz={hz} onKeyDown={onKey}>
      <span className="hzt-ind" aria-hidden="true" />
      {['fire', 'flood'].map((k) => (
        <button key={k} type="button" role="radio" data-set-hz={k} aria-checked={k === hz} tabIndex={k === hz ? 0 : -1} onClick={() => onSet(k)}>
          {k === 'fire' ? <FireIcon /> : <FloodIcon />}<span>{k === 'fire' ? 'Fire' : 'Flood'}</span>
        </button>
      ))}
    </div>
  )
}

export function ThemeButton({ theme, onToggle }) {
  const to = theme === 'dark' ? 'light' : 'dark'
  return <button type="button" className="thb" data-act="theme" aria-label={`Switch to ${to} theme`} title={`Switch to ${to} theme`} onClick={onToggle}>{theme === 'dark' ? <Sun /> : <Moon />}</button>
}

export function DataBadge({ hz, mode, live }) {
  if (hz === 'fire' && mode === 'model') return <span className="smp">Past fire, 2020</span>
  return live ? <span className="smp live">Live data</span> : <span className="smp">Sample data</span>
}

export const Ticks = ({ l }) => <span className="tk" aria-hidden="true">{[1, 2, 3, 4].map((i) => <i key={i} className={i <= l ? 'on' : undefined} />)}</span>

export function Risk({ l, cls = '' }) {                           // 'bare': bars only, word kept for screen readers
  return <span className={`rk ${cls}`} data-l={l}><Ticks l={l} /><span className={cls === 'bare' ? 'sr-only' : undefined}>{RISK[l]}</span></span>
}

export function Kind({ D, it }) {
  return D.live ? <span className="kd">{it && it.picked ? 'No fire' : 'Live fire'}</span> : <span className="kd"><i aria-hidden="true" />Predicted</span>
}

export function Legend({ D }) {
  return <span className="lgd"><span><i />{D && D.live ? 'Past 24 h' : 'Observed'}</span><span className="lg-p"><i />{D && D.live ? 'Forecast' : 'Predicted'}</span></span>
}

export function ConfRing({ conf, s, sw }) {
  const r = s / 2 - sw, C = 2 * Math.PI * r
  return (
    <svg className="cfr" viewBox={`0 0 ${s} ${s}`} width={s} height={s} role="img" aria-label={`Model confidence ${conf} percent`}>
      <circle className="cfr-bg" cx={s / 2} cy={s / 2} r={r} strokeWidth={sw} />
      <circle className="cfr-fg" cx={s / 2} cy={s / 2} r={r} strokeWidth={sw} strokeDasharray={`${f1((C * conf) / 100)} ${f1(C)}`} transform={`rotate(-90 ${s / 2} ${s / 2})`} />
      <text x={s / 2} y={s / 2} dy=".35em" textAnchor="middle" fontSize={f1(s * 0.24)}>{conf}%</text>
    </svg>
  )
}

export function Layers({ D, layers, onLayer }) {
  const keys = D.live ? ['observed', 'predicted'] : ['observed', 'predicted', 'context']
  return (
    <fieldset className="lys">
      <legend className="sr-only">Map layers</legend>
      {keys.map((k) => {
        const off = D.live && k === 'predicted' && !D.modelReady   // no model output yet, so its layer can't be turned on
        return (
          <label className="ly" key={k}>
            <input type="checkbox" data-layer={k} checked={layers[k] && !off} disabled={off} onChange={(e) => onLayer(k, e.target.checked)} />
            <span className={`ly-g ly-${k}`} aria-hidden="true" /><span>{D.layers[k]}</span>
          </label>
        )
      })}
    </fieldset>
  )
}

export const TILES = {
  light: { url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', attr: 'Tiles © Esri — Source: Esri, HERE, Garmin, USGS, NGA, EPA, USDA, NPS' },
  dark: { url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', attr: 'Tiles © Esri — Esri, HERE, Garmin, © OpenStreetMap contributors' },
}
export function Attr({ cls, hz, mode, live, theme }) {
  const src = hz === 'fire' && mode === 'model' ? 'Model output: SkyAlert fire detector on Landsat-8, Aug 27, 2020'
    : live ? 'Fires: NASA FIRMS, VIIRS NOAA-20 · Weather: Open-Meteo · Places: © OpenStreetMap' : 'Sample overlay, not real data'
  return <p className={`attr ${cls}`} data-k="attr">{src} · {TILES[theme].attr}</p>
}

export function AnimText({ value, className }) {                  // text that animations write to directly; React only sets it when the value changes
  const ref = useRef(null)
  useLayoutEffect(() => { const el = ref.current; if (el.dataset.v !== String(value)) { stopAnim(el); el.textContent = value; el.dataset.v = value } }, [value])
  return <span className={className} ref={ref} />
}

export const nowPct = (Sr) => f1(((Sr.obs.length - 1) / (Sr.obs.length + Sr.pred.length - 1)) * 100)
export function Chart({ series: Sr, layers: Ly, w, h, pad, bars, xlabels }) {
  const [pt, pr, pb, pl] = pad
  const nObs = Sr.obs.length, n = nObs + Sr.pred.length, now = nObs - 1
  if (!n) return null
  const iw = w - pl - pr, ih = h - pt - pb, X = (i) => f1(pl + (iw * i) / (n - 1)), Y = (v) => f1(pt + ih * (1 - Math.min(v, Sr.max) / Sr.max))
  const showObs = Sr.live || Ly.observed, showPred = Sr.live || Ly.predicted, showTh = Sr.live || Ly.context   // live weather lines don't depend on the map layers
  const bw = (iw / (n - 1)) * 0.56, bmax = Sr.bars ? Math.max(...Sr.bars) : 1
  const up = Sr.spread ? Sr.pred.map((v, i) => `${X(nObs + i)} ${Y(Math.min(Sr.max, v + Sr.spread[i]))}`) : []
  const dn = Sr.spread ? Sr.pred.map((v, i) => `${X(nObs + i)} ${Y(Math.max(0, v - Sr.spread[i]))}`).reverse() : []
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`${Sr.name}: the last 24 hours and the next 24 hours${Sr.live ? '' : ', sample data'}`}>
      {bars && Sr.bars && Ly.observed && Sr.bars.map((b, i) => { const bh = (ih * 0.4 * b) / bmax; return <rect key={i} className="c-bar" x={f1(X(i) - bw / 2)} y={f1(pt + ih - bh)} width={f1(bw)} height={f1(bh)} /> })}
      {showTh && <line className="c-th" x1={pl} x2={w - pr} y1={Y(Sr.threshold)} y2={Y(Sr.threshold)} />}
      {showPred && Sr.pred.length > 0 && Sr.spread && <path className="c-band" d={`M${X(now)} ${Y(Sr.obs[now])} L${up.join(' L')} L${dn.join(' L')} Z`} />}
      {showPred && Sr.pred.length > 0 && <path className="c-pred" d={`M${X(now)} ${Y(Sr.obs[now])} L${Sr.pred.map((v, i) => `${X(nObs + i)} ${Y(v)}`).join(' L')}`} />}
      {showObs && nObs > 0 && <path className="c-obs" d={`M${Sr.obs.map((v, i) => `${X(i)} ${Y(v)}`).join(' L')}`} />}
      <line className="c-now" x1={X(now)} x2={X(now)} y1={pt - 4} y2={pt + ih} />
      {xlabels && <>
        <text className="c-ax" x={X(0)} y={h - 5}>−24 h</text>
        <text className="c-nowt" x={X(now)} y={h - 5} textAnchor="middle">Now</text>
        <text className="c-ax" x={X(n - 1)} y={h - 5} textAnchor="end">+24 h</text>
      </>}
    </svg>
  )
}

