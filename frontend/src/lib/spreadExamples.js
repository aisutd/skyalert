// Past fires for showing the next-day spread forecast against what actually burned.
//
// All are from the 2024-2025 fire seasons, the model's held-out test years: it never trained on
// them and they weren't used to choose it. They were picked (docs/spread_retrain_plan.md, "Demo
// examples") as the test fires where the model caught the most real spread, so they show it at its
// best. Its typical accuracy is in the legend's "How good is it?" line.
// lat/lon are the exact tile centres from data/retrain/tiles.csv, so the forecast matches the test.
export const SPREAD_EXAMPLES = [
  { id: 'or-2024-09-04', label: 'Grant County, Oregon', day: '2024-09-04', lat: 44.2374468471588, lon: -119.48264576420598 },
  { id: 'id-2024-09-08', label: 'Adams County, Idaho', day: '2024-09-08', lat: 44.472282069888365, lon: -116.2498836943176 },
  { id: 'az-2025-07-28', label: 'Coconino County, Arizona', day: '2025-07-28', lat: 36.48920152841195, lon: -111.95189270738209 },
  { id: 'ca-2025-08-11', label: 'San Luis Obispo County, California', day: '2025-08-11', lat: 35.2564831528412, lon: -120.41904468125314 },
]
