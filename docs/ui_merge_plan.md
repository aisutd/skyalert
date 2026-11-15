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
