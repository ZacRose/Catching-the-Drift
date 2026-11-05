# Benguela Drift Explorer

An interactive page for the product. Pick where floating material starts and how long it drifts, and
see where it is likely to be: in the water, out of the box or (optionally) aground on a coast. Each
answer comes with a 95% interval.

## Opening it

Double-click `app/index.html`. It runs in any modern browser (Chrome, Edge, Firefox, Safari) on
Windows, macOS or Linux, with no Python, server or internet connection. Everything it needs is in
this folder, so the folder can be zipped and sent on its own.

Links can be shared: the address bar keeps the view, e.g. `index.html#start=cell:103&days=360&ground=1`.

## Using it

1. **Pick a start.** Click anywhere in the study area on the map, or type a position (e.g. `28S`,
   `14E`). Until you do, the material is released along the whole Benguela coast (the Week 6 coastal
   release).
2. **Pick how long**, from 15 days to 3 years in 15-day steps. The drift then plays once, and the
   results slide in beside the map when it ends. After that, changing the start or the timeframe
   plays it again.

Turn on **Loop** to keep replaying it, or drag the timeline to any day. The ⓘ button at the top
explains the map, the groundings, the data flags, where the numbers come from and how far to trust them.
The **Seasons** tab shows how the drift changes through the year (see below).

On the map:
- Colour bands show the chance of being in each 2° square, drawn as smooth contours through the
  squares' centres. **Grid cells** shows the squares themselves.
- A solid line and a dashed line enclose the areas with a 50% and a 90% chance, and a diamond marks
  the centre of the material, with its path.
- **Groundings** switches to the model in which material can also run aground. The coastline turns
  pink where it has come ashore, brighter where more of it has.
- **Data coverage** colours each square by its data flag: well sampled (100+ drifters), limited
  (10–99) or very thin (under 10). The start card shows the same flag, with its key.

The results list distance travelled, left the study area, ran aground (with Groundings on), in the
Benguela zone and near the Benguela coast. Each has its 95% range and a small chart from release to
the chosen day, and pointing at one highlights its area on the map.

## How it is built

- `src/app/export_app_data.py` packs two models into `app/data/app_data.js`, each with drifter
  bootstrap intervals (1,000 replicates) for every start square and 15-day step. Run:
  `python3 -m src.app.export_app_data` (about 7 minutes).
  - **water** is the product itself, the 2° shrinkage P (alpha 56.2, 15-day steps). Its intervals
    are `bootstrap.py`'s own replicates, and the script stops if the whole-coast release doesn't
    reproduce `outputs/results/coastal_release_tau15d.csv`.
  - **shore** adds running aground. Starts in the last 15 days of a drifter that ran aground
    (GDP typedeath 1) become moves into an "aground" state for the stretch of coast where it came
    ashore. Each square's product row is scaled by its chance of not running aground within 15 days.
    The module docstring has the details and the rule for where a drifter came ashore.
- `src/app/check_intervals.py` confirms the intervals. It recomputes a sample of them the slow,
  obvious way (row weights, forward steps, `np.quantile`) on the same replicates and compares them
  with the app's. It also compares the bagged mean (the average over replicates) with the full-data
  estimate the app shows. Output: `outputs/results/app_interval_check.csv`. Run it after the exporter
  (about 2 minutes).
- `app/js/model.js` steps either model forward in the browser (`x @ P` per step, as
  `Product.forecast`). `tests/test_app.py` runs it in headless Chrome through `tests/app_check.html`
  and checks it against Python; that test is skipped if Chrome isn't installed.
- `app/js/app.js` is the page itself: map, contours, animation, controls and results. There are no
  libraries and no build step.

Rerun the exporter whenever the product changes (e.g. a new alpha or bootstrap).

## The Seasons view

The **Seasons** tab shows how westward drift from the Benguela zone changes through the year, with its 95%
range, and how that differs for drifters with and without their drogue. It's a pattern, not a forecast: the
data stop in October 2022, and the best method is simply the average of past years' same season. It
comes from `python3 -m src.forecasting.seasonal_robustness`, which saves
`data/processed/seasonal_forecast_app.json`; rerun the exporter to update the page. The format is in the docstring
of `src/app/export_app_data.py`. A season switch on the map was considered and left out (decision 16 in
`project_plan.txt`).
