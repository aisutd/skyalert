import {
  Bell,
  ChartColumn,
  FileDown,
  Flame,
  History,
  Info,
  Map as MapIcon, // renamed so it doesn't clash with JavaScript's built-in Map
  Moon,
  Satellite,
  ScanSearch,
  Settings,
} from 'lucide-react'

// Sidebar options grouped into sections. "map" and "analytics" work; the rest are
// placeholders (marked "Soon") that we'll build step by step.
const SECTIONS = [
  {
    title: 'Monitor',
    items: [
      { id: 'map', label: 'Live Map', icon: MapIcon, ready: true },
      { id: 'analytics', label: 'Analytics', icon: ChartColumn, ready: true },
      { id: 'alerts', label: 'Alerts', icon: Bell },
      { id: 'history', label: 'History', icon: History },
    ],
  },
  {
    title: 'AI Models',
    items: [
      { id: 'detection', label: 'Fire Detection', icon: ScanSearch },
      { id: 'damage', label: 'Damage Mapping', icon: Satellite },
    ],
  },
  {
    title: 'Data',
    items: [{ id: 'reports', label: 'Reports & Export', icon: FileDown }],
  },
]

const FOOTER_ITEMS = [
  { id: 'settings', label: 'Settings', icon: Settings },
  { id: 'about', label: 'About', icon: Info },
]

function NavItem({ item, active, onSelect }) {
  const Icon = item.icon
  return (
    <button
      className={`nav-item${active ? ' active' : ''}`}
      onClick={() => item.ready && onSelect(item.id)}
      disabled={!item.ready}
      title={item.ready ? item.label : `${item.label} (coming soon)`}
    >
      <Icon size={18} strokeWidth={1.8} />
      <span className="nav-label">{item.label}</span>
      {!item.ready && <span className="nav-soon">Soon</span>}
    </button>
  )
}

// Dark mode switch at the bottom of the sidebar
function ThemeToggle({ theme, onToggle }) {
  const dark = theme === 'dark'
  return (
    <button className="nav-item theme-toggle" onClick={onToggle} role="switch" aria-checked={dark} title="Dark mode">
      <Moon size={18} strokeWidth={1.8} />
      <span className="nav-label">Dark mode</span>
      <span className={`switch${dark ? ' on' : ''}`}>
        <span className="switch-knob" />
      </span>
    </button>
  )
}

export default function Sidebar({ activePage, onSelect, theme, onToggleTheme }) {
  return (
    <nav className="sidebar">
      <div className="brand">
        <div className="brand-mark">
          <Flame size={18} strokeWidth={2.2} />
        </div>
        <span className="brand-name">SkyAlert</span>
      </div>

      <div className="nav-sections">
        {SECTIONS.map((section) => (
          <div className="nav-section" key={section.title}>
            <div className="nav-section-title">{section.title}</div>
            {section.items.map((item) => (
              <NavItem key={item.id} item={item} active={item.id === activePage} onSelect={onSelect} />
            ))}
          </div>
        ))}
      </div>

      <div className="nav-footer">
        {FOOTER_ITEMS.map((item) => (
          <NavItem key={item.id} item={item} active={item.id === activePage} onSelect={onSelect} />
        ))}
        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </div>
    </nav>
  )
}
