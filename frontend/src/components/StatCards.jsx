import { CalendarRange, Flame, MapPin } from 'lucide-react'

// Turns "2026-10-09" and "2026-10-10" into "Oct 9 – Oct 10, 2026"
function formatDateRange(start, end) {
  if (!start || !end) return 'No detections'
  // Adding a time makes the browser read the date in local time instead of UTC,
  // otherwise it can show the previous day
  const toDate = (s) => new Date(`${s}T00:00:00`)
  const short = { month: 'short', day: 'numeric' }
  const withYear = { ...short, year: 'numeric' }
  if (start === end) return toDate(start).toLocaleDateString('en-US', withYear)
  return `${toDate(start).toLocaleDateString('en-US', short)} – ${toDate(end).toLocaleDateString('en-US', withYear)}`
}

function StatCard({ icon: Icon, label, value, detail }) {
  return (
    <div className="stat-card">
      <div className="stat-icon">
        <Icon size={18} strokeWidth={2} />
      </div>
      <div>
        <div className="stat-label">{label}</div>
        <div className="stat-value">{value}</div>
        {detail && <div className="stat-detail">{detail}</div>}
      </div>
    </div>
  )
}

export default function StatCards({ summary, error }) {
  if (error) {
    return <div className="stat-cards"><div className="stat-card stat-error">Couldn't load stats: {error}</div></div>
  }

  // While loading, show the cards with "…" so the layout doesn't jump
  const loading = !summary
  return (
    <div className="stat-cards">
      <StatCard
        icon={Flame}
        label="Detections"
        value={loading ? '…' : summary.detections.toLocaleString()}
        detail={loading ? null : `Last ${summary.day_range} days · VIIRS`}
      />
      <StatCard
        icon={MapPin}
        label="Selected region"
        value={loading ? '…' : summary.region.name}
      />
      <StatCard
        icon={CalendarRange}
        label="Acquisition dates"
        value={loading ? '…' : formatDateRange(summary.date_range.start, summary.date_range.end)}
      />
    </div>
  )
}
