import { useCallback, useEffect, useRef, useState } from 'react'
import AnalyticsPage from './pages/AnalyticsPage.jsx'
import FireMap from './components/FireMap.jsx'
import Sidebar from './components/Sidebar.jsx'
import StatCards from './components/StatCards.jsx'
import { getFires, getSummary } from './lib/api.js'

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
  // Every detection, loaded once and shared by the map and the Analytics page
  const [fires, setFires] = useState(null)
  const [firesError, setFiresError] = useState(null)
  const [theme, setTheme] = useState(loadTheme) // light by default
  // Set when "Regional analysis" is clicked in the map's fire popup
  const [analysisFocus, setAnalysisFocus] = useState(null)
  // How far down the map the stat cards reach, so the popup opens below them
  const [safeTop, setSafeTop] = useState(0)
  const mapArea = useRef(null)
  const statCards = useRef(null)

  // Load the stats and the cleaned fire detections from the FastAPI backend once on page load
  // (lib/api.js -> backend/dashboard.py -> main.fetch_fires -> firms_client + data_cleaning)
  useEffect(() => {
    getSummary().then(setSummary).catch((err) => setError(err.message))
    getFires().then(setFires).catch((err) => setFiresError(err.message))
  }, [])

  // Re-measure the stat cards whenever they change size (data loads, window resizes)
  useEffect(() => {
    if (!statCards.current) return
    const measure = () => {
      const cards = statCards.current.getBoundingClientRect()
      const area = mapArea.current.getBoundingClientRect()
      setSafeTop(Math.max(0, cards.bottom - area.top))
    }
    const observer = new ResizeObserver(measure)
    observer.observe(statCards.current)
    return () => observer.disconnect()
  }, [])

  // Whenever the theme changes: recolour the page (CSS reads data-theme) and save the choice
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem('skyalert-theme', theme)
    } catch {
      // storage blocked; the theme just won't be remembered next time
    }
  }, [theme])

  // "Regional analysis" in the popup switches to Analytics, focused on that fire
  const openAnalysis = useCallback((focus) => {
    setAnalysisFocus(focus)
    setActivePage('analytics')
  }, [])

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
      <main className="map-area" ref={mapArea} hidden={activePage !== 'map'}>
        <FireMap fires={fires} error={firesError} theme={theme} safeTop={safeTop} visible={activePage === 'map'} onOpenAnalysis={openAnalysis} />
        <StatCards summary={summary} error={error} ref={statCards} />
      </main>

      {activePage === 'analytics' && (
        <AnalyticsPage fires={fires} error={firesError} focus={analysisFocus} onClearFocus={() => setAnalysisFocus(null)} />
      )}
    </div>
  )
}
