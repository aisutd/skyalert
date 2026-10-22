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
  // Set when "Open regional analysis" is clicked in the map's fire panel
  const [analysisFocus, setAnalysisFocus] = useState(null)
  const mapFrame = useRef(null)
  const statCards = useRef(null)

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

  // Tell the map how far down the stat cards reach, so its fire panel opens below them
  const sendSafeAreaToMap = () => {
    const cards = statCards.current?.getBoundingClientRect()
    const frame = mapFrame.current?.getBoundingClientRect()
    if (!cards || !frame) return
    mapFrame.current.contentWindow?.postMessage(
      { type: 'skyalert-safe-area', top: Math.max(0, cards.bottom - frame.top) },
      window.location.origin
    )
  }

  // Re-measure whenever the cards change size (data loads, window resizes)
  useEffect(() => {
    if (!statCards.current) return
    const observer = new ResizeObserver(sendSafeAreaToMap)
    observer.observe(statCards.current)
    return () => observer.disconnect()
  }, [])

  // Messages from the map page: "Open regional analysis" switches to Analytics,
  // focused on the fire that was clicked
  useEffect(() => {
    const onMessage = (event) => {
      if (event.origin !== window.location.origin || event.source !== mapFrame.current?.contentWindow) return
      if (event.data?.type === 'skyalert-open-analysis') {
        const { lat, lon, label } = event.data
        setAnalysisFocus({ lat, lon, label })
        setActivePage('analytics')
      }
    }
    window.addEventListener('message', onMessage)
    return () => window.removeEventListener('message', onMessage)
  }, [])

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
        onSelect={(page) => {
          // Picking a page from the sidebar shows everything, not one fire's region
          setAnalysisFocus(null)
          setActivePage(page)
        }}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
      />

      {/* The map stays loaded while other pages are open (just hidden), so
          coming back to it is instant instead of rebuilding the whole map */}
      <main className="map-area" hidden={activePage !== 'map'}>
        {/* The existing Folium map, served by dashboard.py */}
        <iframe
          ref={mapFrame}
          className="map-frame"
          src={mapSrc}
          title="Active fire map"
          onLoad={() => {
            sendThemeToMap()
            sendSafeAreaToMap()
          }}
        />
        <StatCards summary={summary} error={error} ref={statCards} />
      </main>

      {activePage === 'analytics' && (
        <AnalyticsPage focus={analysisFocus} onClearFocus={() => setAnalysisFocus(null)} />
      )}
    </div>
  )
}
