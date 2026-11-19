// Loading screen: the logo forms from its bottom-left corner across to the flame tip, then the message and bouncing dots
import { Logo } from './icons.jsx'

export default function Loader({ out }) {
  return (
    <div className={`loader${out ? ' out' : ''}`} role="status">
      <div>
        <span className="ld-logo" aria-hidden="true"><Logo className="ld-svg" /></span>
        <p className="ld-msg">Loading today's fires from NASA FIRMS<span className="ld-dots" aria-hidden="true"><i /><i /><i /></span></p>
      </div>
    </div>
  )
}
