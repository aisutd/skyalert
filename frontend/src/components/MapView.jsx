// The map view (concept 05): full-bleed map with three floating panels. Fire has a second mode, the model on a past fire.
import { forwardRef } from 'react'
import { MODEL, detectorTotals } from '../lib/model.js'
import { measFor, seriesFor } from '../lib/live.js'
import { nf, pc0 } from '../lib/format.js'
import { AnimText, Attr, Brand, Chart, ConfRing, DataBadge, HazardToggle, Kind, Layers, Legend, Risk, ThemeButton } from './bits.jsx'
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
        {[['live', 'Today, live'], ['model', 'Model on a past fire']].map(([k, l]) => <button key={k} type="button" data-mode={k} aria-pressed={s.mode === k} onClick={() => act.setMode(k)}>{l}</button>)}
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
        <div className="m-sp-h"><b>Spread predictor: tomorrow's fire, on past fires</b><span>Catches 52% of next-day fire, vs 27% for “tomorrow = today”. Not on the map: its test data has no locations.</span></div>
        <div className="m-sp-row">
          {MODEL.spread.rows.map((r, i) => (
            <figure className="m-ex" key={i}>
              <div className="m-ex-imgs"><img src={`/model_results/spr_${i + 1}_2.png`} alt="Forecast for day 2" loading="lazy" width="150" height="150" /><img src={`/model_results/spr_${i + 1}_3.png`} alt="What burned on day 2" loading="lazy" width="150" height="150" /></div>
              <p><b>{r.kind.replace(' forecast', '')}: caught {r.caught} of {r.burned} km²</b><small>Forecast · what burned</small></p>
            </figure>
          ))}
        </div>
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
      <h2 className="pc">{D.headline}</h2>
      <hr className="pc" />
      <div className="fp1-row">
        <div data-k="risk"><Risk l={D.risk} cls="lg" /></div>
        <div className="pr"><div className="num mv-num"><AnimText className="fn" value={D.prob} /><span className="pct">{D.unit || '%'}</span></div><span className="pr-l" data-k="plabel">{D.regionLabel}</span></div>
        <div className="cf" data-k="conf">{D.live ? <><span className="cf-n">{nf(D.count)}</span><span className="cf-l">fire detections</span></> : <><ConfRing conf={D.conf} s={52} sw={5} /><span>model<br />confidence</span></>}</div>
      </div>
      <hr className="pc" />
      <div data-k="layers"><Layers D={D} layers={s.layers} onLayer={act.setLayer} /></div>
    </section>
    <section className="fp fp2" data-anchor=".selp-name" aria-label="Selected cell">
      <div className="pbg" aria-hidden="true" />
      <div className="selp-top pc"><Kind D={D} it={it} /><AnimText className="selp-id" value={it.id} /></div>
      <h3 className="selp-name" data-k="selname">{it.name}</h3>
      <div className="pc"><Risk l={it.risk} /></div>
      <dl className="selp-dl pc">
        {D.live ? <>
          <div><dt>Fire-weather index</dt><dd><AnimText className="fwi-n" value={it.fwi} /> / 100</dd></div>
          <div><dt>Model: fire tomorrow</dt><dd>{it.fc && it.fc.status === 'ok' ? `${it.prob}% within 2 km` : s.modelStatus.ready ? 'Inputs unavailable' : 'Waiting for weights'}</dd></div>
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

const MapView = forwardRef(function MapView({ s, act }, ref) {
  return <main className="layer mv-layer" ref={ref} aria-label="Map view">{s.hz === 'fire' && s.mode === 'model' ? <ModelPanels s={s} act={act} /> : <LivePanels s={s} act={act} />}</main>
})

export default MapView
