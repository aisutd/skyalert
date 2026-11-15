// Model results (teammate's figures, cut into panels in public/model_results/) and training-data stats (skyalert/output.txt)

export const MODEL = {
  detector: {
    rows: [
      // bounds: where each 256 × 256 tile sits, found by matching it against Landsat-8 scene 042/035 from that day
      { patch: 'p00877', label: 616, model: 573, found: 552, bounds: [[36.23056, -118.4351], [36.30078, -118.3484]] },
      { patch: 'p00846', label: 474, model: 456, found: 430, bounds: [[36.16029, -118.51927], [36.23056, -118.43257]] },
      { patch: 'p00847', label: 444, model: 433, found: 420, bounds: [[36.09108, -118.51793], [36.16135, -118.43132]] },
      { patch: 'p00878', label: 332, model: 329, found: 316, bounds: [[36.16135, -118.43383], [36.23156, -118.34721]] },
    ],
  },
  spread: {
    cutoff: 0.62,
    rows: [
      { kind: 'Good forecast', today: 16, flagged: 52, burned: 31, caught: 31 },
      { kind: 'Good forecast', today: 20, flagged: 86, burned: 61, caught: 53 },
      { kind: 'Good forecast', today: 43, flagged: 124, burned: 75, caught: 63 },
      { kind: 'Typical forecast', today: 37, flagged: 107, burned: 58, caught: 28 },
    ],
  },
}
MODEL.detector.rows.forEach((r) => { r.center = [(r.bounds[0][0] + r.bounds[1][0]) / 2, (r.bounds[0][1] + r.bounds[1][1]) / 2] })
export const detectorTotals = () => {
  const R = MODEL.detector.rows, sum = (k) => R.reduce((a, r) => a + r[k], 0)
  return { L: sum('label'), M: sum('model'), F: sum('found') }
}
export const modelBounds = () => {                                // one box around all four detector tiles
  const R = MODEL.detector.rows
  return [[Math.min(...R.map((r) => r.bounds[0][0])), Math.min(...R.map((r) => r.bounds[0][1]))], [Math.max(...R.map((r) => r.bounds[1][0])), Math.max(...R.map((r) => r.bounds[1][1]))]]
}

// Training data, copied from skyalert/output.txt (notebooks/inspect_landsat.py)
export const STATS = {
  fire: {
    dataset: 'Landsat-8 Active Fire', source: 'skyalert/output.txt', script: 'notebooks/inspect_landsat.py',
    sampleSize: 100, imageShape: [256, 256, 10], imageDtype: 'uint16', maskShape: [256, 256], maskDtype: 'uint8',
    masks: [{ k: 'schroeder', name: 'Schroeder', s: 's1' }, { k: 'murphy', name: 'Murphy', s: 's2' }, { k: 'kumarRoy', name: 'Kumar-Roy', s: 's3' }],
    bands: [
      { ch: 0, band: 1, name: 'Coastal aerosol', min: 9045, max: 23070, mean: 10387.6 },
      { ch: 1, band: 2, name: 'Blue', min: 8192, max: 23316, mean: 9552.4 },
      { ch: 2, band: 3, name: 'Green', min: 7217, max: 23123, mean: 8647.9 },
      { ch: 3, band: 4, name: 'Red', min: 6259, max: 24575, mean: 8073.4 },
      { ch: 4, band: 5, name: 'Near-infrared', min: 5376, max: 28056, mean: 12272.1 },
      { ch: 5, band: 6, name: 'Shortwave infrared 1', min: 5114, max: 65535, mean: 10797.5 },
      { ch: 6, band: 7, name: 'Shortwave infrared 2', min: 5143, max: 65535, mean: 8397.9 },
      { ch: 7, band: 9, name: 'Cirrus', min: 4969, max: 6410, mean: 5054.5 },
      { ch: 8, band: 10, name: 'Thermal infrared 1', min: 18344, max: 61111, mean: 27938.6 },
      { ch: 9, band: 11, name: 'Thermal infrared 2', min: 16961, max: 45506, mean: 25430.7 },
    ],
    patches: [
      { id: 'LC08_L1TP_001067_20200812_20200812_01_RT_p00441', schroeder: 118, murphy: 142, kumarRoy: 137 },
      { id: 'LC08_L1TP_124017_20200826_20200826_01_RT_p00686', schroeder: 115, murphy: 157, kumarRoy: 156 },
      { id: 'LC08_L1TP_173068_20200817_20200817_01_RT_p00672', schroeder: 41, murphy: 52, kumarRoy: 53 },
      { id: 'LC08_L1TP_103068_20200823_20200823_01_RT_p00610', schroeder: 37, murphy: 52, kumarRoy: 55 },
      { id: 'LC08_L1TP_170069_20200812_20200812_01_RT_p00330', schroeder: 34, murphy: 44, kumarRoy: 47 },
    ],
  },
  flood: null,                                                    // no flood dataset output yet
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
export function scene(id) {                                       // LC08_L1TP_PPPRRR_YYYYMMDD_..._pNNNNN
  const p = id.split('_'), d = p[3]
  return { patch: p[p.length - 1], path: p[2].slice(0, 3), row: p[2].slice(3), date: `${MONTHS[+d.slice(4, 6) - 1]} ${+d.slice(6)}, ${d.slice(0, 4)}` }
}
