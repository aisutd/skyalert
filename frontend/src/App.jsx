import { useEffect, useRef, useState } from 'react'
import AnalyticsPage from './pages/AnalyticsPage.jsx'
import Sidebar from './components/Sidebar.jsx'
import StatCards from './components/StatCards.jsx'

// Remember the chosen theme between visits. Wrapped in try/catch because some
// browsers (private mode, blocked storage) throw when localStorage is used.
function loadTheme() {
  try {
    return localStorage.getItem('skyalert-theme') === 'dark' ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

export default function App() {
  // Which sidebar page is open: "map" or "analytics"
  const [activePage, setActivePage] = useState('map')
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState(null)
  const [theme, setTheme] = useState(loadTheme) // light by default
  const mapFrame = useRef(null)

  // The map's starting theme goes in its URL. This is only worked out once, so
  // flipping the toggle doesn't reload the whole map.
  const [mapSrc] = useState(() => `/api/map?theme=${theme}`)

  // Load the stats (detection count, region, dates) from dashboard.py once on page load
  useEffect(() => {
    fetch('/api/summary')
      .then((res) => {
        if (!res.ok) throw new Error(`Server returned ${res.status}`)
        return res.json()
      })
      .then(setSummary)
      .catch((err) => setError(err.message))
  }, [])

  // Tell the map page (inside the iframe) which theme to use
  const sendThemeToMap = () => {
    mapFrame.current?.contentWindow?.postMessage({ type: 'skyalert-theme', theme }, window.location.origin)
  }

  // Whenever the theme changes: recolour this page (CSS reads data-theme),
  // save the choice, and recolour the map controls too
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem('skyalert-theme', theme)
    } catch {
      // storage blocked; the theme just won't be remembered next time
    }
    sendThemeToMap()
  }, [theme])

  return (
    <div className="app">
      <Sidebar
        activePage={activePage}
        onSelect={setActivePage}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
      />

      {/* The map stays loaded while other pages are open (just hidden), so
          coming back to it is instant instead of rebuilding the whole map */}
      <main className="map-area" hidden={activePage !== 'map'}>
        {/* The existing Folium map, served by dashboard.py */}
        <iframe ref={mapFrame} className="map-frame" src={mapSrc} title="Active fire map" onLoad={sendThemeToMap} />
        <StatCards summary={summary} error={error} />
      </main>

      {activePage === 'analytics' && <AnalyticsPage />}
    </div>
  )
}
