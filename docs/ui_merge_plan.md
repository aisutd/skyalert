# Merging Sneha's UI (test2) with the spread model work: plan

Branch `merge-ui-ml`, from `feat-spread-forecast` plus `origin/test2`. Decisions (user, 2026-11-14):

| Question | Decision |
|---|---|
| Frontend | Sneha's design and framework (test2 `frontend/`), everything else from `feat-spread-forecast` |
| The spread-forecast features | Rebuilt in her style, not dropped |
| Region | Whole contiguous US; the fire list follows the area on the map |
| Values typed into her code | Real data where possible; anything still fixed is checked against its source and labelled |
| Model output shown as "% chance" | Calibrated, so the % is honest |

## Steps

1. **Merge.** Take test2's frontend and prototype. Keep the v2 model backend, giving her UI the
   API shape it expects:
   - `status`, `chance_near`, `predicted_km2`, `fire_today_km2`
   - `/api/model-status`
   - the prototype mount

   Her old-model `backend/spread_forecast.py` is replaced. These are the pieces from the reverted
   attempt (`backup-merge-ui-ml`) that fit the decisions: merge resolution, calibration, past-fire
   mode, honest wording. They are reused and re-verified.
2. **Calibration.** `training/calibrate_v2.py` maps the model's scores to how often fire really
   followed (fitted on 2023, checked on 2024–25); stored in `models/spread_v2/calibration.json`.
3. **Whole US, following the map.**
   - The first view is the contiguous US.
   - When you pan or zoom in the map view, the list re-groups the fires in the visible area (the
     part not covered by panels) and forecasts the strongest 8. Earlier forecasts stay cached.
   - Only user moves trigger this: the app's own fly-to moves don't, so there's no feedback loop.
   - Labels say "in view" instead of "CA".
4. **Click any fire.**
   - Port the prototype's "a clicked dot joins the list": the nearest dot within 10 px, plus its
     detections within 15 km.
   - Clicking empty map gives that spot's weather, as in her prototype.
   - The clicked fire is selected and forecast.
5. **Forecast detail for the selected fire, in her panels.**
   - The selected fire's panel shows the forecast date, the days of fire it starts from, the likely
     area and the new spread in km², the weather dates, any notes, and the model's tested accuracy.
   - The map adds crisp outlines for the selected fire: today's fire, and likely or possible pixels.
6. **Past fires ("Spread, past fire" mode):** four held-out 2024–25 fires, forecast vs what burned,
   with layer checkboxes.
7. **Values typed into her code:**
   - Spread model numbers: the v2 test results.
   - Detector example and training stats: checked against `data/demos/california/` and
     `output.txt`, with sources cited.
   - Flood: stays sample data, and is labelled as such (there's no flood model on this side).
8. **Verify.**
   - pytest.
   - Frontend build.
   - Drive the app in Chrome: live list on first load, pan to another state and see the list
     follow, click a dot, the past-fire mode, the Results dialog, dark mode.
9. Commit in steps. Push only when the user says so.

## Mac safety

Same as before: servers and scripts under `nice`, nothing heavy running in parallel.

## Done (2026-11-14)

All steps are done and verified in Chrome:
- the US-wide list on first load
- a zoom re-lists the fires in view, and makes no further requests once settled (no loop)
- a clicked unlisted fire joins the list with its forecast panel
- the past-fire mode
- dark mode
- the Results dialog

Found while checking: Sneha's `.lm-mk { position: relative; }` stacked Leaflet's marker icons, so
numbered marker n was drawn 12 px × (n − 1) south of its fire (marker 9 was 96 px off). It's now
`position: absolute`; every marker lands within 1 px of its fire.

## Clicking fire dots (2026-11-14, after user feedback: "hard to click, have to zoom in a lot")

Measured before the fix: clicks did register, but there was no sign a dot was clickable (the cursor stayed
a grab hand, no hover), and nothing visible happened for about 2.7 s, until the place, weather and forecast
had all loaded. Dots in the overview's small map couldn't be clicked at all. Now:
- Pointer cursor and a ring on hover; the nearest dot within 14 px counts. Screen positions are cached per
  map view, so hovering stays cheap with thousands of dots.
- Click → a pulsing ring on the dot at once (0.15 s) → the fire joins the list, selected, with place and
  weather (0.7 s; existing forecasts reused) → its forecast fills in (about 2.6 s).
- A dot clicked in the overview opens the map view on that fire.
- Fires just over the border (FIRMS's box reaches into Canada and Mexico) have no gridMET weather; the
  forecast now says what share of its area used average weather when that's 25% or more.

## Seeing the selected fire's forecast (2026-11-15, after user feedback: "zoom to forecast is still not clear")

The first fix zoomed to the whole 64 km forecast area, where a small fire's one or two 1 km squares are about
10 px across and sit under the 24 px numbered marker. Now:
- **Focus on this fire.** `forecastFocus` (lib/spreadExamples.js) finds the clicked fire's own squares (burning,
  likely or possible): the group nearest the centre, where squares up to 2 apart count as one fire. Other fires in
  the 64 km area, such as farm burns a few km away, stay on the map but don't widen the view. It returns that
  group plus a 2-square margin, at least 9 × 9 squares.
- **Zoom the map in on it** flies to that focus area (max zoom 14), so each 1 km square is about 60–70 px.
- **Once squares are at least 22 px on screen:**
  - a 1 km grid and outline cover the focus area
  - the selected marker moves onto a short stem just north of the squares, labelled "· forecast for {day}", so
    it no longer covers them
  - zooming back out puts it back on the fire
- **Panel close-up:** it shows the same focus area, about 220 px wide and drawn at 2× resolution. The legend
  amounts count what's in the close-up. The whole-area totals are on their own line, which names the dashed box.
- **Sharp squares.** The "burning" frames and grid lines are now drawn on whole canvas pixels. Before, strokes
  on half pixels blurred into grey bands when enlarged. This applies to the past-fire view too.

Checked in Chrome, light and dark, on Yalobusha County, MS (adjacent squares) and Cavalier County, ND
(scattered farm burns), plus the past-fire view. `forecastFocus` is checked on synthetic grids: a centre fire
with distant burns, a 1-square vs 3-square gap, a large fire, an empty grid, an edge clamp, and the bounds
maths.
