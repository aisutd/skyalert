// Leaflet, plus the heatmap plugin.
// leaflet.heat is an old plugin that adds itself to a global "L", so Leaflet has to be
// made global before the plugin loads. Imports run in order, so './leaflet-global.js' goes first.
import L from './leaflet-global.js'
import 'leaflet.heat'
import 'leaflet/dist/leaflet.css'

export default L
