import { useId } from 'react'
import { FLAME, WAVE, WAVE_ICON } from '../lib/logoPaths.js'

export function Logo({ className = 'br-m' }) {                    // solid theme colours, no gradients
  return (
    <svg className={className} viewBox="0 0 378 519" aria-hidden="true">
      <path className="lg-flame" fillRule="evenodd" d={FLAME} />
      <path className="lg-wave" fillRule="evenodd" d={WAVE} />
    </svg>
  )
}

export function FireIcon() {                                      // the flame symbol, which fills in when Fire is on
  return (
    <svg className="ico ico-fire" viewBox="0 0 32 32" aria-hidden="true">
      <path className="fl-outer" d="M16 3C17.6 8.2 24 10.8 24 18.2A8 8 0 0 1 8 18.2C8 14.8 9.6 12.4 11.4 10.8C11.6 13.8 13 15.6 14.8 15.8C14 11.2 14.4 7.2 16 3Z" />
      <path className="fl-inner" d="M16.6 15C17.4 17.6 20.4 18.8 20.4 22A4.4 4.4 0 0 1 11.6 22C11.6 20.3 12.4 19.1 13.3 18.3C13.4 19.7 14.1 20.6 15 20.7C14.6 18.5 15 16.8 16.6 15Z" />
    </svg>
  )
}

export function FloodIcon() {                                     // clipped to its own shape, so the outline sits inside the edge
  const id = 'icc' + useId().replace(/[^a-zA-Z0-9_-]/g, '')
  return (
    <svg className="ico ico-flood" viewBox="-4 -4 108 108" aria-hidden="true">
      <clipPath id={id}><path clipRule="evenodd" d={WAVE_ICON} /></clipPath>
      <path className="ic-wave" fillRule="evenodd" clipPath={`url(#${id})`} d={WAVE_ICON} />
    </svg>
  )
}

const line = { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true }
export const Sun = () => <svg {...line}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
export const Moon = () => <svg {...line}><path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a6.8 6.8 0 0 0 10.5 10.5z" /></svg>
export const Collapse = () => <svg {...line}><path d="M4 14h6v6M20 10h-6V4M14 10l7-7M3 21l7-7" /></svg>   // arrows pointing in: leave the map view
export const Expand = () => <svg {...line}><path d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7" /></svg>
export const Back = () => <svg {...line}><path d="M19 12H5M11 6l-6 6 6 6" /></svg>
export const Bars = () => <svg {...line}><path d="M4 20h16M7 16v-5M12 16V6M17 16v-8" /></svg>
export const XMark = () => <svg {...line}><path d="M6 6l12 12M18 6L6 18" /></svg>
