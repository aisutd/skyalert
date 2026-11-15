// Small shared helpers: number formatting, risk names, saved settings, and the number animations

export const RISK = ['', 'Low', 'Moderate', 'High', 'Critical']   // index = risk level 1-4

export const f1 = (v) => Math.round(v * 10) / 10
export const nf = (v) => v.toLocaleString('en-US')
export const pc0 = (a, b) => `${Math.round((a / b) * 100)}%`
export const compass = (deg) => ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][Math.round((((deg % 360) + 360) % 360) / 45) % 8]

export const store = {                                            // localStorage can throw (private mode, blocked storage)
  get(k) { try { return localStorage.getItem('skyalert-flow-' + k) } catch { return null } },
  set(k, v) { try { localStorage.setItem('skyalert-flow-' + k, v) } catch { /* not remembered */ } },
}

export const REDUCE = typeof matchMedia === 'function' ? matchMedia('(prefers-reduced-motion: reduce)') : { matches: false }
export const easeIO = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)

const running = new WeakMap()                                     // element -> cancel function of its current animation
export function stopAnim(el) { running.get(el)?.(); running.delete(el) }

export function countUp(el, from, to, dur, delay = 0) {          // the number changes value while it travels
  if (!el) return
  stopAnim(el)
  const t0 = performance.now() + delay
  let raf = 0
  const step = (now) => {
    const t = Math.min(1, Math.max(0, (now - t0) / dur))
    el.textContent = Math.round(from + (to - from) * easeIO(t))
    if (t < 1) raf = requestAnimationFrame(step)
    else running.delete(el)
  }
  el.textContent = from
  raf = requestAnimationFrame(step)
  running.set(el, () => { cancelAnimationFrame(raf); el.textContent = to })
}

const COORD = /^\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*$/
export function rollCoords(el, from, to, dur) {                   // "lat, lon" rolls from the last fire's spot to the new one
  const a = COORD.exec(from), b = COORD.exec(to)
  if (!el || !a || !b || REDUCE.matches) return
  stopAnim(el)
  const dec = (b[1].split('.')[1] || '').length, t0 = performance.now()
  let raf = 0
  const step = (now) => {
    const t = Math.min(1, (now - t0) / dur), e = easeIO(t)
    el.textContent = t < 1 ? `${(+a[1] + (b[1] - a[1]) * e).toFixed(dec)}, ${(+a[2] + (b[2] - a[2]) * e).toFixed(dec)}` : to
    if (t < 1) raf = requestAnimationFrame(step)
    else running.delete(el)
  }
  el.textContent = from                                           // no flash of the new value before the roll starts
  raf = requestAnimationFrame(step)
  running.set(el, () => { cancelAnimationFrame(raf); el.textContent = to })
}
