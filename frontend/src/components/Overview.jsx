// The overview (concept 07). data-k marks a piece that also exists in the map view and flies there; .pc marks a piece that only fades.
import { forwardRef, useLayoutEffect, useRef } from 'react'
import { RISK, nf } from '../lib/format.js'
import { measFor, seriesFor } from '../lib/live.js'
import { AnimText, Attr, Brand, Chart, DataBadge, HazardToggle, Kind, Layers, Legend, Risk, ThemeButton, nowPct } from './bits.jsx'
import { Bars, Expand } from './icons.jsx'

const Overview = forwardRef(function Overview({ s, act }, ref) {
  const { D, sel, hz } = s, it = D.items[sel]
  const meas = measFor(D, sel), series = seriesFor(D, sel)
  const cells = useRef(null)

  useLayoutEffect(() => {                                         // only fires that fit on line 1 show; cut-off ones can't be tabbed to
    const ul = cells.current
    const fit = () => { const lis = [...ul.children], top = lis[0]?.offsetTop; lis.forEach((li) => { li.inert = li.offsetTop > top }) }
    fit()
    const ro = new ResizeObserver(fit); ro.observe(ul)
    return () => ro.disconnect()
  })

  return (
    <main className="layer ov" ref={ref} aria-label="Overview">
      <header className="ov-hd">
        <div data-k="brand"><Brand /></div>
        <div data-k="tog"><HazardToggle hz={hz} onSet={act.setHz} /></div>
        <span className="sp" />
        <button type="button" className="dsb pc" data-act="data" aria-haspopup="dialog" onClick={act.openData}><Bars /><span>Model results</span></button>
        <div data-k="smp"><DataBadge hz={hz} mode={s.mode} live={D.live} /></div>
        <div data-k="thm"><ThemeButton theme={s.theme} onToggle={act.toggleTheme} /></div>
      </header>

      <section className="ov-hero" aria-label="Selected cell">
        <div className="ov-eye"><b data-k="selname">{it.name}</b><span className="pc"><Kind D={D} it={it} /></span><span className="pc">{D.window}</span></div>
        <div className="num ov-num"><AnimText className="fn" value={it.prob} /><span className="pct">{D.unit || '%'}</span></div>
        <p className="ov-cap" data-k="plabel">{D.probLabel}</p>
        <div className="ov-row">
          <div data-k="risk"><Risk l={it.risk} cls="xl" /></div>
          <p className="ov-conf" data-k="conf">{D.live ? <><b className="ov-cnt">{nf(D.count)}</b> fire detections</> : <>Model confidence {it.conf}%<br />{D.model}</>}</p>
        </div>
        <button type="button" className="cta" data-k="cta" data-act="focus" onClick={act.focusMap}>Focus on the map <Expand /></button>
        {hz === 'fire'
          ? <button type="button" className="cta ghost pc" data-act="model" onClick={act.openModel}>See the model on a past fire</button>
          : <span className="cta ghost cta-ph" aria-hidden="true">See the model on a past fire</span>}{/* keeps the space so the centred hero doesn't jump */}
      </section>

      <section className="ov-right" aria-label="Cells">
        <div className="ov-slot pc"><button type="button" className="ov-go" data-act="focus" aria-label="Focus on the map" title="Focus on the map" onClick={act.focusMap}><Expand /></button></div>
        <Attr cls="ov-attr" hz={hz} mode={s.mode} live={D.live} theme={s.theme} />
        <ul className="ov-cells" ref={cells}>
          {D.items.slice(0, 3).map((x, i) => (
            <li key={i}><button type="button" className="ov-cell" data-k={`cell-${i}`} data-sel={i} aria-pressed={i === sel} onClick={() => act.select(i)}><b>{i + 1}</b>{x.name} · {RISK[x.risk]}</button></li>
          ))}
        </ul>
        <div className="ov-layers" data-k="layers"><Layers D={D} layers={s.layers} onLayer={act.setLayer} /></div>
      </section>

      <section className="ov-foot" aria-label="Conditions">
        <div className="ov-meas">
          {meas.map((m, i) => <div className="mrow" data-k={`m-${i}`} key={m.l}><span className="ml">{m.l}<small>{m.s}</small></span><span className="mv">{m.v}<u>{m.u}</u></span></div>)}
        </div>
        <div className="ov-sh pc"><span>{series.name}</span><Legend D={D} /></div>
        <div className="ov-spark" data-k="chart"><Chart series={series} layers={s.layers} w={1150} h={50} pad={[6, 4, 4, 4]} /></div>
        <div className="ov-spark-l pc" aria-hidden="true"><span>−24 h</span><span style={{ left: `${nowPct(series)}%` }}>Now</span><span>+24 h</span></div>
      </section>
    </main>
  )
})

export default Overview
