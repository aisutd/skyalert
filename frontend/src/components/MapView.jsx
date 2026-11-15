// The map view (concept 05): full-bleed map with three floating panels. Fire has a second mode, the model on a past fire.
import { forwardRef } from 'react'
import { MODEL, detectorTotals } from '../lib/model.js'
import { measFor, seriesFor, selHeadline } from '../lib/live.js'
import { SPREAD_EXAMPLES, longDay, pixelKm2 } from '../lib/spreadExamples.js'
import { nf, pc0 } from '../lib/format.js'
import { AnimText, Attr, Brand, Chart, DataBadge, HazardToggle, Kind, Layers, Legend, Risk, ThemeButton } from './bits.jsx'
import { Back, Bars, Collapse } from './icons.jsx'

function TopRows({ s, act }) {                                    // brand row + toggle row, the same in both modes
  return <>
    <div className="fp1-top">
      <div data-k="brand"><Brand /></div><span className="sp" />
      <div data-k="smp"><DataBadge hz={s.hz} mode={s.mode} live={s.D.live} /></div>
      <button type="button" className="thb dsi pc" data-act="data" aria-haspopup="dialog" aria-label="Model results" title="Model results" onClick={act.openData}><Bars /></button>
      <div data-k="thm"><ThemeButton theme={s.theme} onToggle={act.toggleTheme} /></div>
    </div>
    <div className="fp1-r2">
      <div data-k="tog"><HazardToggle hz={s.hz} onSet={act.setHz} /></div>
      <button type="button" className="cta back" data-k="cta" data-act="back" onClick={act.back}><Back />Overview</button>
    </div>
    {s.hz === 'fire' && (
      <div className="msw pc" role="group" aria-label="What the map shows">
        {[['live', 'Today, live'], ['model', 'Detector, past fire'], ['spread', 'Spread, past fire']].map(([k, l]) => <button key={k} type="button" data-mode={k} aria-pressed={s.mode === k} onClick={() => act.setMode(k)}>{l}</button>)}
      </div>
    )}
  </>
}

function ZoomStack({ act }) {
  return (
    <div className="zm pc">
      <button type="button" data-act="zin" aria-label="Zoom in" onClick={act.zoomIn}>+</button>
      <button type="button" data-act="zout" aria-label="Zoom out" onClick={act.zoomOut}>−</button>
      <button type="button" className="zm-back" data-act="back" aria-label="Back to overview" title="Back to overview" onClick={act.back}><Collapse /></button>
    </div>
  )
}

function ExImgs({ k, i, alts, className = 'ex-imgs' }) {
  return <div className={className}>{alts.map((a, j) => <figure key={a}><img src={`/model_results/${k}_${i + 1}_${j + 1}.png`} alt={a} loading="lazy" width="300" height="300" /><figcaption>{a}</figcaption></figure>)}</div>
}
export { ExImgs }

function ModelPanels({ s, act }) {
  const R = MODEL.detector.rows, t = R[s.mSel], { L, M, F } = detectorTotals()
  const lyr = (k, label) => <label className="ly"><input type="checkbox" data-mlayer={k} checked={s.mLayers[k]} onChange={(e) => act.setMLayer(k, e.target.checked)} /><span className={`ly-g ly-${k}`} aria-hidden="true" /><span>{label}</span></label>
  return <>
    <section className="fp fp1" data-anchor=".num" aria-label="Summary">
      <div className="pbg" aria-hidden="true" />
      <TopRows s={s} act={act} />
      <h2 className="pc">Our fire detector on a wildfire in Sequoia National Forest, Aug 27, 2020</h2>
      <hr className="pc" />
      <div className="m-stats">
        <div className="pr"><div className="num mv-num"><AnimText className="fn" value={Math.round((F / L) * 100)} /><span className="pct">%</span></div><span className="pr-l" data-k="plabel">of real fire found</span></div>
        <div className="pr pc"><div className="mv-num2">{Math.round((F / M) * 100)}%</div><span className="pr-l">of its alerts were real fire</span></div>
        <div className="pr pc"><div className="mv-num2">93% · 82%</div><span className="pr-l">overall, on 3,113 unseen tiles</span></div>
      </div>
      <hr className="pc" />
      <fieldset className="lys pc"><legend className="sr-only">Map layers</legend>{lyr('sat', 'Satellite image')}{lyr('model', 'Model: fire found')}{lyr('label', 'Correct answer')}</fieldset>
    </section>
    <section className="fp fp2" data-anchor=".selp-name" aria-label="Selected tile">
      <div className="pbg" aria-hidden="true" />
      <div className="selp-top pc"><span className="kd">Fire detector · tile {s.mSel + 1}</span><span className="selp-id">{t.patch}</span></div>
      <h3 className="selp-name" data-k="selname">Caught {t.found} of {t.label} fire pixels</h3>
      <div className="m-meter pc" role="img" aria-label={`${pc0(t.found, t.label)} of real fire found`}><i style={{ width: `${(t.found / t.label) * 100}%` }} /></div>
      <ExImgs k="det" i={s.mSel} alts={['Satellite view', 'Model', 'Correct answer']} className="ex-imgs pc" />
      <dl className="selp-dl pc">
        <div><dt>Real fire found</dt><dd>{pc0(t.found, t.label)}</dd></div><div><dt>Alerts that were real</dt><dd>{pc0(t.found, t.model)}</dd></div>
        <div><dt>False alarms</dt><dd>{t.model - t.found} px</dd></div><div><dt>Missed</dt><dd>{t.label - t.found} px</dd></div>
      </dl>
      <hr className="pc" />
      <h4 className="fp-h pc">All 4 tiles, 7.7 × 7.7 km each</h4>
      <ul className="its">
        {R.map((x, i) => (
          <li key={x.patch}><button type="button" className={`it${i === s.mSel ? ' on' : ''}`} data-k={`cell-${i}`} data-msel={i} aria-pressed={i === s.mSel} onClick={() => act.selectTile(i)}>
            <span className="it-n">{i + 1}</span><span className="it-name">Tile {x.patch}<small>{x.center[0].toFixed(3)}, {x.center[1].toFixed(3)}</small></span><span /><span className="it-p">{pc0(x.found, x.label)}<small>found</small></span>
          </button></li>
        ))}
      </ul>
    </section>
    <section className="fp fp3 m-fp3" data-anchor=".mss" aria-label="Spread predictor">
      <div className="pbg" aria-hidden="true" />
      <div className="mss m-sp">
        <div className="m-sp-h"><b>Spread predictor: the next day's fire</b><span>On {MODEL.spread.test.period} fires it never saw, its forecast overlapped what burned {MODEL.spread.test.all.iou.toFixed(2)} (IoU), vs {MODEL.spread.test.all.stays.toFixed(2)} for “the fire stays where it is”; {MODEL.spread.test.growing.iou.toFixed(2)} vs {MODEL.spread.test.growing.stays.toFixed(2)} on larger, growing fires.</span></div>
        <button type="button" className="cta m-sp-go" onClick={() => act.setMode('spread')}>See it on a past fire, next to what burned</button>
      </div>
    </section>
    <ZoomStack act={act} />
    <Attr cls="mv-attr" hz={s.hz} mode={s.mode} live={s.D.live} theme={s.theme} />
  </>
}

function LivePanels({ s, act }) {
  const { D, sel } = s, it = D.items[sel]
  const meas = measFor(D, sel), series = seriesFor(D, sel)
  return <>
    <section className="fp fp1" data-anchor=".num" aria-label="Summary">
      <div className="pbg" aria-hidden="true" />
      <TopRows s={s} act={act} />
      {D.live && D.topName ? <h2 data-k="selname">{D.topName}</h2> : <h2 className="pc">{D.headline}</h2>}
      <hr className="pc" />
      <div className="fp1-row">
        <div data-k="risk"><Risk l={D.risk} cls="lg" /></div>
        <div className="pr"><div className="num mv-num"><AnimText className="fn" value={D.prob} /><span className="pct">{D.unit || '%'}</span></div><span className="pr-l" data-k="plabel">{D.regionLabel}</span></div>
        <div className="cf" data-k="conf">{D.live ? <><span className="cf-n">{nf(D.count)}</span><span className="cf-l">fire detections</span></> : <><span className="cf-n">{D.conf}<span className="cf-u">%</span></span><span className="cf-l">model confidence</span></>}</div>
      </div>
      <hr className="pc" />
      <div data-k="layers"><Layers D={D} layers={s.layers} onLayer={act.setLayer} /></div>
    </section>
    <section className="fp fp2" data-anchor=".selp-name" aria-label="Selected cell">
      <div className="pbg" aria-hidden="true" />
      <div className="selp-top pc"><Kind D={D} it={it} /><AnimText className="selp-id" value={it.id} /></div>
      {D.live ? <h3 className="selp-name selp-head pc">{selHeadline(D, it)}</h3> : <h3 className="selp-name" data-k="selname">{it.name}</h3>}
      <div className="pc"><Risk l={it.risk} /></div>
      <dl className="selp-dl pc">
        {D.live ? <>
          <div><dt>Fire-weather index</dt><dd><AnimText className="fwi-n" value={it.fwi} /> / 100</dd></div>
          <div><dt>Model: fire {D.forecastDay ? `on ${D.forecastDay}` : 'next day'}</dt><dd>{it.fc && it.fc.status === 'ok' ? `${it.prob}% within 2 km` : s.modelStatus.ready ? 'Inputs unavailable' : 'Waiting for weights'}</dd></div>
          <div><dt>{D.metricName}</dt><dd>{it.metric}</dd></div><div><dt>Evidence</dt><dd>{it.evidence}</dd></div>
        </> : <>
          <div><dt>Probability</dt><dd>{it.prob}%</dd></div><div><dt>Confidence</dt><dd>{it.conf}%</dd></div>
          <div><dt>{D.metricName}</dt><dd>{it.metric}</dd></div><div><dt>Evidence</dt><dd>{it.evidence}</dd></div>
        </>}
      </dl>
      <p className="selp-win pc">{it.window}</p>
      {D.live && it.reasons && it.reasons.length > 0 && <ul className="rs pc" aria-label="Why this risk level">{it.reasons.map((r, i) => <li key={i} className={`rs-${r.effect}`}>{r.text}</li>)}</ul>}
      <hr className="pc" />
      <h4 className="fp-h pc">{D.listTitle || `All cells, ${D.window.toLowerCase()}`}</h4>
      <ul className="its">
        {D.items.map((x, i) => (
          <li key={i}><button type="button" className={`it${i === sel ? ' on' : ''}`} data-k={`cell-${i}`} data-sel={i} aria-pressed={i === sel} onClick={() => act.select(i)}>
            <span className="it-n">{i + 1}</span><span className="it-name">{x.name}<small>{x.id}</small></span><Risk l={x.risk} cls="bare" /><span className="it-p">{x.prob}{D.live && !D.modelReady ? '' : '%'}</span>
          </button></li>
        ))}
      </ul>
    </section>
    <section className="fp fp3" data-anchor=".mss" aria-label="Conditions">
      <div className="pbg" aria-hidden="true" />
      <div className="mss">{meas.map((m, i) => <div className="ms" data-k={`m-${i}`} key={m.l}><span className="ms-l">{m.l}</span><span className="ms-v">{m.v}<span className="ms-u">{m.u}</span></span></div>)}</div>
      <div className="fp3-c"><div className="fp3-h pc"><span>{series.name}</span><Legend D={D} /></div><div data-k="chart"><Chart series={series} layers={s.layers} w={460} h={120} pad={[10, 6, 22, 6]} bars xlabels /></div></div>
    </section>
    <ZoomStack act={act} />
    <Attr cls="mv-attr" hz={s.hz} mode={s.mode} live={D.live} theme={s.theme} />
  </>
}

function SpreadPanels({ s, act }) {                              // the spread model on a past fire: forecast vs what burned
  const ex = SPREAD_EXAMPLES[s.spSel], fc = s.spFc, ok = fc && fc.status === 'ok' && fc.check
  const k = ok ? fc.check : null, km2 = ok ? pixelKm2(fc) : 0, area = (px) => `${Math.round(px * km2)} km²`
  const T = MODEL.spread.test
  const caughtPct = ok && k.new_fire_pixels ? Math.round((k.new_fire_caught_pixels / k.new_fire_pixels) * 100) : 0
  const lyr = (key, label) => <label className="ly"><input type="checkbox" data-splayer={key} checked={s.spLayers[key]} onChange={(e) => act.setSpLayer(key, e.target.checked)} /><span className={`ly-g ly-sp-${key}`} aria-hidden="true" /><span>{label}</span></label>
  return <>
    <section className="fp fp1" data-anchor=".num" aria-label="Summary">
      <div className="pbg" aria-hidden="true" />
      <TopRows s={s} act={act} />
      <h2 className="pc">Our spread model on a past fire: {ex.label}, {longDay(ex.day)}</h2>
      <hr className="pc" />
      <div className="m-stats">
        <div className="pr"><div className="num mv-num"><AnimText className="fn" value={ok ? caughtPct : 0} /><span className="pct">%</span></div><span className="pr-l" data-k="plabel">of the new ground it forecast</span></div>
        <div className="pr pc"><div className="mv-num2">{ok ? `${k.model_iou.toFixed(2)} vs ${k.persistence_iou.toFixed(2)}` : '…'}</div><span className="pr-l">overlap with what burned, vs “stays where it is”</span></div>
        <div className="pr pc"><div className="mv-num2">{T.all.iou.toFixed(2)} vs {T.all.stays.toFixed(2)}</div><span className="pr-l">overall, on {T.period} fires</span></div>
      </div>
      <hr className="pc" />
      <fieldset className="lys pc"><legend className="sr-only">Map layers</legend>
        {lyr('today', ok ? `Burning on ${longDay(fc.fire_day)}` : 'Burning on the fire day')}{lyr('forecast', 'Model: forecast for the next day')}{lyr('actual', ok ? `Actually burned on ${longDay(fc.forecast_day)}` : 'Actually burned the next day')}
      </fieldset>
    </section>
    <section className="fp fp2" data-anchor=".selp-name" aria-label="Selected past fire">
      <div className="pbg" aria-hidden="true" />
      <div className="selp-top pc"><span className="kd">Spread model · past fire {s.spSel + 1}</span><span className="selp-id">{ex.day}</span></div>
      <h3 className="selp-name" data-k="selname">{ok ? `Forecast ${area(k.new_fire_caught_pixels)} of ${area(k.new_fire_pixels)} of new ground` : fc && fc.status !== 'ok' ? 'Forecast unavailable' : 'Running the model…'}</h3>
      {ok && <div className="m-meter pc" role="img" aria-label={`${caughtPct}% of the new ground forecast`}><i style={{ width: `${caughtPct}%` }} /></div>}
      {ok ? (
        <dl className="selp-dl pc">
          <div><dt>Burning on the fire day</dt><dd>{area(fc.fire_today_pixels)}</dd></div><div><dt>Model: likely next day</dt><dd>{area(fc.forecast_pixels)}</dd></div>
          <div><dt>Actually burned</dt><dd>{area(k.burned_pixels)}</dd></div><div><dt>Overlap (IoU)</dt><dd>{k.model_iou.toFixed(2)} vs {k.persistence_iou.toFixed(2)}</dd></div>
        </dl>
      ) : fc && fc.status !== 'ok' ? <p className="selp-win pc">{fc.note}</p> : null}
      <p className="selp-win pc">Held-out fires where the model caught the most spread: its best cases. On larger, growing fires overall it scores {T.growing.iou.toFixed(2)} vs {T.growing.stays.toFixed(2)}.</p>
      <hr className="pc" />
      <h4 className="fp-h pc">{SPREAD_EXAMPLES.length} past fires, {T.period}</h4>
      <ul className="its">
        {SPREAD_EXAMPLES.map((x, i) => (
          <li key={x.id}><button type="button" className={`it${i === s.spSel ? ' on' : ''}`} data-k={`cell-${i}`} data-spsel={i} aria-pressed={i === s.spSel} onClick={() => act.selectSpread(i)}>
            <span className="it-n">{i + 1}</span><span className="it-name">{x.label}<small>{longDay(x.day)}</small></span><span /><span />
          </button></li>
        ))}
      </ul>
    </section>
    <section className="fp fp3 m-fp3" data-anchor=".mss" aria-label="How to read the map">
      <div className="pbg" aria-hidden="true" />
      <div className="mss m-sp">
        <div className="m-sp-h"><b>How to read the map</b><span>Red squares are where the model expected fire the next day, orange where it was possible. Dark outlines show the fire on the day it started from; blue squares show what really burned. Red with blue is a hit; blue alone is a miss.</span></div>
      </div>
    </section>
    <ZoomStack act={act} />
    <Attr cls="mv-attr" hz={s.hz} mode={s.mode} live={s.D.live} theme={s.theme} />
  </>
}

const MapView = forwardRef(function MapView({ s, act }, ref) {
  const panels = s.hz === 'fire' && s.mode === 'model' ? <ModelPanels s={s} act={act} />
    : s.hz === 'fire' && s.mode === 'spread' ? <SpreadPanels s={s} act={act} /> : <LivePanels s={s} act={act} />
  return <main className="layer mv-layer" ref={ref} aria-label="Map view">{panels}</main>
})

export default MapView
