// Playback for "Spread, past fire" on the map: the fire grows from the start day into the next-day forecast,
// then starts over until paused. The model gives ONE forecast for the whole next day, so the hours in between
// are filled in by lighting squares outward from the start-day fire, nearest first. That part is illustration.
// The confidence dial is real: every position is the model's own output at that level of certainty.
//
//   prepare(fc)      -> the forecast as flat arrays, plus the order squares light up in (cached per forecast)
//   play, onFrame    -> one shared clock (0 = start day, 1 = next day) that the map and the panel both follow
//   drawFrame        -> one frame of the fire on a canvas

const REDUCED = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches
const prepared = new WeakMap()

export function prepare(fc) {
  if (prepared.has(fc)) return prepared.get(fc)
  const n = fc.size, len = n * n
  const p = new Float32Array(len), today = new Uint8Array(len), actual = new Uint8Array(len), likely = new Uint8Array(len)
  for (let r = 0; r < n; r++) for (let k = 0; k < n; k++) {
    const i = r * n + k
    p[i] = fc.probability[r][k]; today[i] = fc.fire_today[r][k] ? 1 : 0; actual[i] = fc.actual && fc.actual[r][k] ? 1 : 0; likely[i] = fc.likely[r][k] ? 1 : 0
  }
  const dist = new Int16Array(len).fill(-1), queue = []           // distance, in squares, from the start-day fire
  for (let i = 0; i < len; i++) if (today[i]) { dist[i] = 0; queue.push(i) }
  if (!queue.length) { let m = 0; for (let i = 1; i < len; i++) if (p[i] > p[m]) m = i; dist[m] = 0; queue.push(m) }
  for (let h = 0; h < queue.length; h++) {
    const i = queue[h], r = Math.floor(i / n), k = i % n
    for (let dr = -1; dr <= 1; dr++) for (let dk = -1; dk <= 1; dk++) {
      const R = r + dr, K = k + dk
      if (R < 0 || K < 0 || R >= n || K >= n) continue
      const j = R * n + K
      if (dist[j] < 0) { dist[j] = dist[i] + 1; queue.push(j) }
    }
  }
  let far = 1
  for (let i = 0; i < len; i++) if (actual[i] || p[i] >= 0.05) far = Math.max(far, dist[i])
  const key = new Float32Array(len)                               // lighting order: nearer first, and surer squares a little earlier
  for (let i = 0; i < len; i++) key[i] = (dist[i] / far) * 0.8 + (1 - p[i]) * 0.2
  const count = (a) => a.reduce((s, v) => s + v, 0)
  // The model's own cut. It decides "likely" on its raw score (fc.threshold), but `probability` is that score turned
  // into a real chance, so on the dial its own setting is the lowest chance among the squares it called likely.
  let own = 1
  for (let i = 0; i < len; i++) if (likely[i] && p[i] < own) own = p[i]
  if (own === 1) own = fc.threshold
  const P = { n, len, p, today, actual, likely, key, own, paces: new Map(), nToday: count(today), nActual: count(actual),
    km2: fc.pixel_km.north_south * fc.pixel_km.east_west }
  P.burnPace = pace(P, (i) => actual[i], (i) => dist[i])          // the same for what really burned
  prepared.set(fc, P)
  return P
}

function pace(P, want, order) {                                   // when each wanted square lights up: 0 for the start-day fire, then evenly up to 1
  const T = new Float32Array(P.len), ids = []
  for (let i = 0; i < P.len; i++) { T[i] = P.today[i] ? 0 : 1; if (!P.today[i] && want(i)) ids.push(i) }
  ids.sort((a, b) => order(a) - order(b)).forEach((i, j) => { T[i] = 0.04 + 0.96 * (j + 1) / ids.length })
  return T
}

// The dial is in whole percent of chance. At the model's own setting the called squares are exactly its `likely`
// mask, so the scores there match the ones the backend reports.
export const ownPct = (P) => Math.round(P.own * 100)
export const thrOf = (P, pct) => (pct === ownPct(P) ? P.own : pct / 100)
const called = (P, thr) => (thr === P.own ? (i) => P.likely[i] === 1 : (i) => P.p[i] >= thr)

export function paceFor(P, thr) {
  if (!P.paces.has(thr)) P.paces.set(thr, pace(P, called(P, thr), (i) => P.key[i]))
  return P.paces.get(thr)
}

export function score(P, thr) {                                   // the forecast at this confidence against what burned
  let tp = 0, fp = 0, fn = 0
  const is = called(P, thr)
  for (let i = 0; i < P.len; i++) { const a = is(i), y = P.actual[i]; if (a && y) tp++; else if (a) fp++; else if (y) fn++ }
  return { right: tp / Math.max(1, tp + fp), caught: tp / Math.max(1, tp + fn), overlap: tp / Math.max(1, tp + fp + fn), called: tp + fp }
}

// Squares shown as burning at time t. It runs from the start-day area (t = 0) to the forecast area (t = 1),
// so both ends agree with the numbers in the side panel.
export function areaAt(P, T, thr, t) {
  let lit = 0, all = 0, total = 0
  const is = called(P, thr)
  for (let i = 0; i < P.len; i++) {
    if (!is(i)) continue
    total++
    if (P.today[i]) continue
    all++; if (T[i] <= t) lit++
  }
  return P.nToday + (total - P.nToday) * (all ? lit / all : t)
}
export function burnedAt(P, t) {                                  // the same for what really burned
  let lit = 0, all = 0
  for (let i = 0; i < P.len; i++) { if (P.today[i] || !P.actual[i]) continue; all++; if (P.burnPace[i] <= t) lit++ }
  return P.nToday + (P.nActual - P.nToday) * (all ? lit / all : t)
}

// ---------- the shared clock ----------
const RUN = 8000, HOLD = 2600                                     // ms to play through, then to rest on the result before starting over
export const play = { t: 0, playing: !REDUCED, holding: false, held: 0, pct: 32, actual: true }   // actual: also mark what really burned
const subs = new Set()
let raf = 0, last = 0
function loop(now) {
  const dt = Math.min(100, now - last); last = now
  if (play.playing) {
    if (play.t >= 1) { play.holding = true; play.held += dt; if (play.held > HOLD) rewind() }
    else play.t = Math.min(1, play.t + dt / RUN)
  }
  subs.forEach((fn) => fn(now))
  raf = subs.size ? requestAnimationFrame(loop) : 0
}
export function onFrame(fn) {                                     // fn(now) every frame while subscribed; returns the unsubscribe
  subs.add(fn)
  if (!raf) { last = performance.now(); raf = requestAnimationFrame(loop) }
  return () => { subs.delete(fn) }
}
export function rewind() { play.t = 0; play.held = 0; play.holding = false }
export function seek(t) { play.playing = false; play.t = t; play.held = 0; play.holding = false }
export function toggle() { play.playing = !play.playing; if (play.playing && play.t >= 1) rewind() }
export function restart() { rewind(); play.playing = !REDUCED }

// ---------- drawing ----------
const CELL = 16                                                   // canvas pixels per forecast square
function squares(g, P, test, color, blur, alpha) {                // one glowing shape per colour keeps this fast
  g.beginPath()
  for (let i = 0; i < P.len; i++) if (test(i)) g.rect((i % P.n) * CELL + 1, Math.floor(i / P.n) * CELL + 1, CELL - 2, CELL - 2)
  g.shadowColor = color; g.shadowBlur = blur * CELL; g.globalAlpha = alpha; g.fillStyle = color; g.fill()
  g.shadowBlur = 0; g.globalAlpha = 1
}
// One frame: the start-day fire dark, squares that caught a while ago red, the newest edge bright.
// night (the dark theme) glows more. showActual adds a small blue square on each square that really burned,
// appearing at the same pace, so the forecast underneath stays visible.
export function drawFrame(canvas, P, { t, thr, night, now, showActual }) {
  if (canvas.width !== P.n * CELL) canvas.width = canvas.height = P.n * CELL
  const g = canvas.getContext('2d'), T = paceFor(P, thr)
  g.clearRect(0, 0, canvas.width, canvas.height)
  const flick = REDUCED ? 1 : 0.82 + 0.18 * Math.sin(now / 170)
  const is = called(P, thr), on = (i) => (P.today[i] || is(i)) && T[i] <= t, age = (i) => (T[i] === 0 ? 1 : t - T[i])
  squares(g, P, (i) => on(i) && age(i) >= 0.3, night ? '#A5311A' : '#9A2D18', night ? 0.5 : 0, 0.95)
  squares(g, P, (i) => on(i) && age(i) >= 0.08 && age(i) < 0.3, night ? '#FF5A45' : '#E5483A', night ? 1.1 : 0.5, 0.94 * flick)
  squares(g, P, (i) => on(i) && age(i) < 0.08, night ? '#FFC766' : '#FF9A1F', night ? 1.6 : 1.2, flick)
  if (showActual) {
    const m = 5                                                   // inset: the mark is the middle third, so the forecast colour shows around it
    g.fillStyle = night ? '#6EC1FF' : '#2A78D6'; g.beginPath()
    for (let i = 0; i < P.len; i++) if (P.actual[i] && P.burnPace[i] <= t) g.rect((i % P.n) * CELL + m, Math.floor(i / P.n) * CELL + m, CELL - 2 * m, CELL - 2 * m)
    g.fill()
  }
}
