# Surface transport from the Benguela across the South Atlantic

If something floating enters the water off south-west Africa, where is it likely to be in 15 days, three months
or a year, and how far can that answer be trusted? This project estimates it from 28 years of satellite-tracked
drifting buoys, as a transition matrix anyone can load and step forward, with its uncertainty attached.

**DATA3001, Term 3 2026.** Isaac Rose, Zhiru Lin (computational); Vidisha Trivedi, Liam Lightfoot (quantitative);
Keyao Huang (business). Client: Shane Elipot, University of Miami.

**To see the results now:** open [`app/index.html`](app/index.html) in any browser. It needs no install, server
or internet connection. Pick a start and a timeframe; the [app's README](app/README.md) has the details.

## What we found

All numbers are from the 2° matrix with 95% intervals from resampling drifters, unless stated.

**Where material goes (RQ1).** Released along the Benguela coast, it leaves quickly and crosses the gyre:

| After | Still in the coastal cells | Has left the box | Centre of what remains |
|---|---|---|---|
| 15 days | 31% (22–37) | 0.001% | 27.0°S, 13.0°E |
| 3 months | 0.63% (0.25–1.5) | 0.25% (0.11–0.58) | 23.5°S, 4.8°E |
| 1 year | 0.006% (0.004–0.009) | 0.89% (0.61–1.5) | 22.9°S, 16.8°W |

- **Connections between areas.** After a year, 77% (72–80) of material from the coast is in the central South
  Atlantic and 15% (12–19) in the tropical South Atlantic. It takes about 330 days for 1% of it to have crossed
  30°W, into the western third of the box.
- **From the Orange Basin** (the Venus and Mopane discovery blocks), 84% (80–87) is in the central South Atlantic
  after a year, and there is a 2.3% (0.8–3.9) chance of having reached the Benguela coastal cells by then. Drifters
  that run aground stop reporting, so that is a lower bound.
- **Where it gathers.** Material that stays settles around 30.2°S, 16.4°W, with half of it in 11% of the box's
  area, whatever its release point. The coastal start is forgotten after about 2.7 years.
- **Where it leaves.** Slowly: 0.25% per 15 days (0.22–0.29) in the long run, through the southern (0.14%) and eastern
  (0.10%) edges. Almost none leaves northwards, and none through the western edge.

**How reliable it is (RQ2).** Shrinking thin 2° cells towards their 4° cell beats a fair baseline by 0.90 per move in
held-out log-likelihood (0.84–0.98). Its gain over raw counts comes entirely from never giving a real move zero
probability (raw counts do for 0.76% of test moves); where raw counts can score, they are very slightly better.
Checked on drifters the model never saw:

- **Chances of being in a cell hold up at every horizon**, across five orders of magnitude: after a year, cells given
  about 1.4% really held drifters 1.3% of the time.
- **The "90% area" holds** 92% of real moves at 15 days, 89% at 3 months and 85% at a year: a little optimistic at long times.
- **The chance of having left the box runs high** at long times: 6.6% predicted against 5.2% after 3 months, and 15.4%
  against 10.4% after a year, across all starts.
- **The past predicts later years**: a matrix fitted before 2015 scores moves from 2015 on within 1% of one fitted to the
  same amount of data from all years.

**Wind matters most (RQ3).** Drogue status is the biggest single choice. Drifters that have lost their drogue are pushed
by the wind: from the coast, material carried by them is about 10° further west after a year (27.0°W against 16.8°W) and
7–8 times as likely to have left the box (6.8% against 0.89%). The main estimate pools both, so it sits between "water"
and "floating material". Say which one you need.

**Seasons (RQ4).** Westward drift from the Benguela zone is seasonal: about 120 km per 15 days in summer and spring, about
70 km in winter, and summer beat winter in all 8 years with enough drifters to compare. It survives every other check we
made, but it is mainly in the wind-exposed drifters (+83 km between summer and winter, against +24 km, -4 to +47, for
drifters that follow the water). Knowing the season cuts the typical error in a season's drift from 33 km to 26 km (21.5%
better, interval -0.4% to 39.8%), but one year can still differ a lot from the average, so we offer no validated seasonal
forecast or risk window. The gain from a seasonal matrix is real but about 0.3% per move, so the map pools all seasons.

## How far to trust the intervals

The 95% intervals show which drifters we happened to have. They don't include how much the ocean changes from year to year.
Resampling whole years instead widens the percentages little (1.0–1.2 times) but the centre's longitude after a year by 2.1
times, and the long-run centre's latitude by 2.3 times. In a simulation where the truth is known, 95% intervals held it about 92% of the time from
well-sampled cells, 89% from cells with 10–99 drifters and 83% from cells with under 10. **For the whole-coast release it is only about
75%, and as low as 22%.** Its cells are thin, and shrinkage pulls them towards their 4° cell, which the bootstrap can't see: the 15-day
coastal share comes out about 4 points low (within 1 point with no shrinkage). So read coastal numbers with their range across the
prior as well as their interval: that share is 24–39% from a tenth to ten times the chosen alpha. The north-east of the box
(Angola) is thin at every resolution. P is a planning tool for weeks to a year ahead,
not a forecast for a particular date or an oil-spill model.

## Using the matrix

```python
from src.linalg.product import load_product

prod = load_product()                                # the 2° product; cell_size=4.0 for the 4° baseline
dist = prod.forecast(lat=-29.0, lon=13.0, days=90)   # a distribution over cells and "left the box", in 15-day steps
prod.summary(dist)                                   # % left the box, centre, the most likely cells
prod.se, prod.ci_low, prod.ci_high                   # uncertainty for every entry of P
prod.start_drifters                                  # how much data is behind each row: check it
```

The matrices are plain `.npz` files in `data/processed/`, so `numpy.load` is all another language needs. States are grid
cells, numbered from the south-west corner, then "left the box" as the last state.

## Reproducing the analysis

```bash
conda env create -f environment.yml && conda activate DATA3001
python3 -m pytest tests                  # about a minute; one test also needs Google Chrome, and is skipped without it
python3 -m src.run_all --from-tables     # everything from the committed tables, no internet needed
python3 -m src.run_all                   # everything from the raw data (reads the cloud copy once, about 10 minutes, 3 GB of memory)
```

Seeds are fixed, so a rerun gives the same numbers. `python3 -m src.run_all --list` shows the order; the whole run takes about 35
minutes on a 10-core laptop once the cloud data is cached. Every script also runs alone, and its docstring says what it needs and makes. The raw data are never changed: the GDP is read from
the cloud with [clouddrift](https://github.com/Cloud-Drift/clouddrift) and region R is cached in `data/interim/` (not in git).

To use another region, change the box in `config/region.yaml` (with its coastal zone and subregions) and rerun.
`src/inference/region_swap.py` tests the quantitative half of this on a second box.

## Where things are

| Question | Script | Results (`outputs/results/`, `outputs/figures/`) |
|---|---|---|
| Is there enough data? | `src/inference/adequacy.py`, `transition_adequacy.py` | `adequacy_*`, `transition_adequacy_*` |
| Is the data sound? | `src/inference/data_quality.py`, `data_forensics.py` | `data_quality.png`, `data_forensics.csv` |
| The transition table | `src/gridding/transitions.py`, `lags.py` | `data/processed/transitions_tau*d.npz`, `lag_*` |
| RQ1: where material goes | `src/linalg/coastal_release.py`, `connectivity.py` | `coastal_release_*`, `connectivity*`, `orange_basin_release.png` |
| RQ1: where it gathers | `src/linalg/long_run.py` | `long_run_*` |
| RQ2: which estimator | `src/inference/evaluate.py`, `score_intervals.py`, `prior_sensitivity.py` | `evaluation_*`, `score_intervals*`, `prior_sensitivity*` |
| Reliability | `src/inference/bootstrap.py`, `calibration.py`, `year_intervals.py`, `interval_coverage.py`, `markov_check.py` | `calibration*`, `year_intervals_*`, `interval_coverage_*`, `markov_check*` |
| RQ3 and other choices | `src/inference/robustness.py`, `region_swap.py` | `robustness*`, `region_swap*` |
| RQ4: seasons | `src/forecasting/seasonal_forecast.py`, `seasonal_robustness.py` | `seasonal_*` |
| The app | `src/app/`, `app/` | `app/index.html`, `app_interval_check.csv` |

[`project_plan.txt`](project_plan.txt) is the running record: what was done each week, the decision log (section 7) and the
forensic review of the whole project (section 8). [`logs/ai_use_log.md`](logs/ai_use_log.md) records how AI was used and what
was checked. [`docs/proposal/`](docs/proposal/) has the submitted proposal.

## Limitations

- **Square cells spread material artificially**, a little more with each step (McAdam & van Sebille 2018).
- **Chaining assumes a 15-day step forgets its past.** This costs little at 2° (chaining predicts 30 and 90-day moves better
  than fitting them directly), but a real drifter lingers longer than the chain says.
- **Leaving is permanent in the matrix.** About a fifth of the drifters that leave are back in the box on any later day.
- **Grounding isn't in the main matrix**, because grounded drifters stop reporting, so coastal numbers are lower bounds. The
  app's Groundings switch adds it as a second model.
- **Drifters seldom come close to shore.** A coastal cell is about 200 km across and the 9 sampled ones have 2–71 drifters each.
- **Data end in October 2022**, and all years and seasons are pooled.

## Data and credit

Hourly Global Drifter Program data, version 2.01 (Elipot et al. 2016, 2022;
[doi:10.25921/x46c-3620](https://doi.org/10.25921/x46c-3620)), accessed through clouddrift. Region R: 42°S–4°S, 60°W–20°E,
19,269,758 hourly readings from 1,975 drifters, September 1994 to October 2022, of which 1,892 drifters give the 776,777 usable
15-day moves. Coastline: Natural Earth 1:110m (public domain).
