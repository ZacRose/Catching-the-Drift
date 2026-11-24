# South Atlantic Gyre Drift Explorer

An interactive page for the product. Pick where floating material starts and how long it drifts, and
see where it is likely to be: in the water, out of the box or (optionally) aground on a coast. Each
answer comes with a 95% interval.

## Opening it

Double-click `app/index.html`. It runs in any modern browser (Chrome, Edge, Firefox, Safari) on
Windows, macOS or Linux, with no Python, server or internet connection. Everything it needs is in
this folder, so the folder can be zipped and sent on its own.

Links can be shared: the address bar keeps the view, e.g. `index.html#start=cell:103&days=360&ground=1&drifters=undrogued&season=DJF`.

## Using it

1. **Pick a start.** Click anywhere in the study area on the map, or type a position (e.g. `28S`,
   `14E`), or choose the Orange Basin. Until you do, the material is released along the whole Benguela
   coast (the Week 6 coastal release).
2. **Pick how long**, from 15 days to 3 years in 15-day steps. The drift then plays once, and the
   results slide in beside the map when it ends. After that, changing the start or the timeframe
   plays it again.
3. **Choose the conditions**, if you want to. These change the answer in place, without replaying.
   - **Drifters:** *All* is the main estimate, as in the report. *Undrogued only* uses moves made after
     a drifter lost its drogue, which the wind pushes directly: arguably nearer to oil or debris, though
     nothing here weathers or evaporates. The data flags then count undrogued drifters.
   - **Released:** *All year*, or the first day of a season. The drift then changes as the seasons pass,
     every 90 days, shown under the timeline and beside the day. A year released in summer runs summer,
     autumn, winter, spring.

   With anything but all drifters, all year, a hollow diamond and fainter dots keep that answer on the map to compare with.

Turn on **Loop** to keep replaying it, or drag the timeline to any day. The ⓘ button at the top
explains the map, the seasons, the groundings, the data flags, where the numbers come from and how far to trust them.

On the map:
- Colour bands show the chance of being in each 2° square, drawn as smooth contours through the
  squares' centres. **Grid cells** shows the squares themselves.
- A solid line and a dashed line enclose the areas with a 50% and a 90% chance, and a diamond marks
  the centre of the material, with its path.
- **Groundings** switches to the model in which material can also run aground. The coastline turns
  pink where it has come ashore, brighter where more of it has.
- **Data coverage** colours each square by its data flag: well sampled (100+ drifters), limited
  (10–99) or very thin (under 10). The start card shows the same flag, with its key.

Under the headline, **Mostly in** names the subregions holding most of the material. The results list
distance travelled, left the study area, ran aground (with Groundings on), in the
Benguela zone and near the Benguela coast. Each has its 95% range and a small chart from release to
the chosen day, and pointing at one highlights its area on the map. From a start off the coast, the coast's
row also gives the chance of having reached it by then: material passes through, so being there today understates it.
**How far to trust this** then gives only the checks that apply to that start, timeframe and set of conditions.

## How it is built

- `src/app/export_app_data.py` packs every combination into `app/data/app_data.js`, each with drifter
  bootstrap intervals for every start square. Run: `python3 -m src.app.export_app_data` (about 25 minutes).
  - **All drifters** is the product itself, the 2° shrinkage P (alpha 56.2, 15-day steps). Its intervals
    are `bootstrap.py`'s own replicates, and the script stops if the whole-coast release doesn't
    reproduce `outputs/results/coastal_release_tau15d.csv`.
  - **Undrogued only** is the same fit on moves that start without a drogue
    (`src/inference/drogue_comparison.py`). The script stops if it doesn't reproduce that script's coast release.
  - **Groundings** adds running aground to either. Starts in the last 15 days of a drifter that ran aground
    (GDP typedeath 1) become moves into an "aground" state for the stretch of coast where it came
    ashore. Each square's row is scaled by its chance of not running aground within 15 days.
    The module docstring has the details and the rule for where a drifter came ashore.
  - **Seasons** (`src/forecasting/seasonal_matrices.py`). A season's row is the all-year row pulled towards that
    season's moves. The page rebuilds the four matrices from sparse counts, so they add little to the file.
    Their intervals use 200 replicates (1,000 for all year) and are stored at 23 of the 72 steps, every 15 days to 3
    months, then every 30 to a year and every 90 after that. The page reads the steps between off a straight line on a log scale.
- `src/app/check_intervals.py` confirms the intervals. It recomputes a sample of them the slow,
  obvious way (row weights, forward steps, `np.quantile`) on the same replicates and compares them
  with the app's, for both sets and the seasons, including the steps that are read off a line. It also compares the
  bagged mean (the average over replicates) with the full-data estimate the app shows. Output:
  `outputs/results/app_interval_check.csv`. Run it after the exporter (about 40 minutes).
- `app/js/model.js` steps any model forward in the browser (`x @ P` per step, as
  `Product.forecast`), and builds a season's matrix when first used. `tests/test_app.py` runs it in headless
  Chrome through `tests/app_check.html` and checks it against Python; that test is skipped if Chrome isn't installed.
- `app/js/app.js` is the page itself: map, contours, animation, controls and results. There are no
  libraries and no build step.

Rerun the exporter whenever the product changes (e.g. a new alpha or bootstrap).

## What the seasons show

The ⓘ panel quotes the plain seasonal averages: how westward drift from the Benguela zone changes through the year,
and how that differs for drifters with and without their drogue. It's a pattern, not a forecast: the data stop in
October 2022, and the best method is simply the average of past years' same season. It comes from
`python3 -m src.forecasting.seasonal_robustness`, which saves `data/processed/seasonal_pattern_app.json`; rerun the
exporter to update the page. The map's seasons show smaller gaps than these averages, because a whole matrix needs far
more data than one average (decisions 16 and 20 in `project_plan.txt`).
