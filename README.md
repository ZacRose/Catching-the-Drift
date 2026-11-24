# Surface transport from the Benguela across the South Atlantic

If something floating enters the water off south-west Africa, where is it likely to be in 15 days, three months
or a year, and how far can that answer be trusted? We estimate this from 28 years of satellite-tracked drifting
buoys, as a transition matrix that can be loaded and stepped forward, with its uncertainty attached.

**DATA3001, Term 3 2026.** Isaac Rose, Zhiru Lin (computational), Vidisha Trivedi, Liam Lightfoot (quantitative),
Keyao Huang (business). Client: Shane Elipot, University of Miami.

## Overview

### Quick start

Open [`app/index.html`](app/index.html), the South Atlantic Gyre Drift Explorer, in any browser to explore the
results. It needs no installation, server or internet connection. Choose a start, a timeframe and, optionally, the
drifter type and release season. The [app's README](app/README.md) explains how it works.

To rerun the analysis:

```bash
conda env create -f environment.yml && conda activate DATA3001
python3 -m src.run_all --from-tables     # offline, from the committed transition tables
```

Full instructions are in [Reproducibility](#reproducibility).

## Results

All numbers are from the 2° matrix with 95% intervals from resampling drifters, unless stated.

### RQ1: Connectivity

Material released along the Benguela coast moves offshore within weeks and then west across the gyre:

| After | Still in the coastal cells | Has left the box | Centre of what remains |
|---|---|---|---|
| 15 days | 31% (22–37) | 0.001% | 27.0°S, 13.0°E |
| 3 months | 0.63% (0.25–1.5) | 0.25% (0.11–0.58) | 23.5°S, 4.8°E |
| 1 year | 0.006% (0.004–0.009) | 0.89% (0.61–1.5) | 22.9°S, 16.8°W |

- After a year, 77% (72–80) of material from the coast is in the Central South Atlantic and 15% (12–19) in the
  Tropical South Atlantic. It takes about 330 days for 1% of it to cross 30°W, into the western third of the box.
- From the Orange Basin (the Venus and Mopane discovery blocks), 84% (80–87) is in the Central South Atlantic after
  a year, and there is a 2.3% (0.8–3.9) chance it has reached the Benguela coastal cells. Drifters that run aground
  stop reporting, so this is a lower bound.
- Material that stays in R gathers around 30.2°S, 16.4°W, with half of it in 11% of the box's area, whatever its
  release point. A coastal release loses any trace of its starting point after about 2.7 years.
- In the long run, material leaves R slowly, at 0.25% per 15 days (0.22–0.29), mainly through the southern (0.14%)
  and eastern (0.10%) edges. Almost none leaves to the north or west.

### RQ2: Reliability

Shrinking thin 2° cells towards their 4° cell beats a fair baseline by 0.90 per move in held-out log-likelihood
(0.83–0.98). Its gain over raw counts comes entirely from never giving a real move zero probability (raw counts do
for 0.76% of test moves). Where raw counts can score, they are very slightly better. Checked on drifters the model
never saw:

- Predicted probabilities matched what test drifters did at every horizon. After a year, cells given about a 1.4%
  chance held test drifters 1.3% of the time.
- The 90% areas contained 92% of test moves at 15 days, 89% at 3 months and 85% at a year, so they are slightly
  optimistic at long horizons.
- The chance of having left the box is too high at long horizons: 6.6% predicted against 5.2% observed after
  3 months, and 15.4% against 10.4% after a year.
- On drifters first seen from 2015, a matrix fitted only to moves before 2015 scores within 0.0075 per move
  (−0.0155 to +0.0006) of one fitted to the same number of moves from all years, and the interval includes zero.

### RQ3: Drogue status

Drogue status had the largest effect of anything we tested. Drifters that have lost their drogue are pushed by the
wind as well as the current. From the coast, material carried by them is about 10° further west after a year
(27.0°W against 16.8°W, gap 10.2°, 9.1–11.3) and 7–8 times as likely to have left the box (6.8%, 3.9–10.9, against
0.89%). The main estimate pools both kinds of drifter, so it sits between water that follows the current and
floating material exposed to the wind. Users should choose the matrix that matches the material they are modelling.

The undrogued matrix is saved too. On held-out undrogued drifters it predicts no better than the pooled one at
15 days, but better at 3 months and a year, and its 90% area holds what it should at a year (90%, against 85% for the
pooled one). Its coastal intervals are too narrow, more so than the pooled ones. In a known-truth simulation, the
interval on the one-year longitude held the true value 13% of the time (biased 1.3° east), so the westward shift is
clear but its interval is a guide only. Undrogued drifters stand in for oil or debris on the strength of a handful
of studies, and nothing here weathers or evaporates.

### RQ4: Seasonality

Westward drift from the Benguela zone is seasonal: about 120 km per 15 days in summer and spring, against about
70 km in winter, and summer beat winter in all 8 years with enough drifters to compare. It held under every
robustness check we ran (`seasonal_robustness.csv`), but it is mainly in the wind-exposed drifters (+83 km between
summer and winter, against +24 km, −4 to +47, for drifters that follow the water). Knowing the season cuts the
typical error in a season's drift from 33 km to 26 km (21.5% better, interval −0.4% to 39.8%), but one year can
still differ a lot from the average, so we offer no validated seasonal forecast or risk window.

The map can release in any season. When a release runs across seasons, the chain switches to each new season's
matrix. On held-out drifters in other years this beats ignoring the season at every horizon tried, while keeping
the release season's matrix throughout does worse than ignoring the season from about 6 months. The gain is real
but small, about 0.4% per move, and a season's matrix keeps only about a third of the raw gap in westward drift.
The rest does not recur from year to year at this level of detail. As a result, the seasons change the map little,
and the centre after a year moves by 0.4° at most.

## Uncertainty

The 95% intervals reflect sampling variation between drifters only. They do not include how much the ocean changes
from year to year. Resampling whole years instead barely widens the percentages (1.0–1.2 times) but widens the
centre's longitude after a year by 2.1 times, and the long-run centre's latitude by 2.3 times.

In a simulation where the truth is known, 95% intervals held it about 92% of the time from well-sampled cells, 89%
from cells with 10–99 drifters and 83% from cells with under 10. For the release from the whole Benguela coast,
median coverage was only 75%, ranging from 22% to 100% across the 12 quantities. Its cells are thin, and shrinkage
pulls them towards their 4° cell, which the bootstrap cannot detect. The 15-day coastal share comes out about
4 points low (within 1 point with no shrinkage). Coastal numbers should therefore be read with their range across
prior strengths as well as their interval: that share is 24–39% from a tenth to ten times the chosen alpha. The
north-east of the box (Angola) is thin at every resolution.

P is intended for planning over weeks to a year. It does not forecast conditions on a particular date and does not
model how oil weathers.

## Usage

```python
from src.linalg.product import load_product

prod = load_product()                                # the 2° product, cell_size=4.0 for the 4° baseline
dist = prod.forecast(lat=-29.0, lon=13.0, days=90)   # a distribution over cells and "left the box", in 15-day steps
prod.summary(dist)                                   # % left the box, centre, the most likely cells
prod.se, prod.ci_low, prod.ci_high                   # uncertainty for every entry of P
prod.start_drifters                                  # number of drifters behind each row

wind = load_product(drifters="undrogued")            # the same fit on moves that start without a drogue
prod.forecast(lat=-29.0, lon=13.0, days=360, season="DJF")   # released on the first day of a season, changing with the seasons
```

The matrices are plain `.npz` files in `data/processed/`, so `numpy.load` is all another language needs. States are
grid cells, numbered from the south-west corner, then "left the box" as the last state.

## Reproducibility

```bash
conda env create -f environment.yml && conda activate DATA3001
python3 -m pytest tests                  # 139 tests, about a minute (the app test needs Google Chrome)
python3 -m src.run_all --from-tables     # 21 of the 35 steps from the committed tables, no internet needed
python3 -m src.run_all                   # all 35 steps from the raw data (reads the cloud copy once, about 10 minutes, 3 GB of memory)
```

The environment uses Python 3.14, and the tests also pass on 3.11.

Seeds are fixed, so a rerun gives the same numbers. `python3 -m src.run_all --list` shows the steps in the order
they run. Every script also runs on its own, and its docstring says what it reads and writes.

`--from-tables` leaves out the 14 steps that need the hourly data (building the tables, the hourly EDA, and the
checks that use the full drifter records or lags longer than 90 days) and keeps their committed results. The full
run from the raw data takes about 1 hour 50 minutes on a 10-core laptop.

**Last rerun:** 23 November 2026, from a clean clone of commit `5f77bd3` with `python3 -m src.run_all --from-tables`.
All 21 steps ran in 2 hours 18 minutes on a 2-core laptop with 4 GB of memory. Of the 54 result and matrix files,
48 were identical to the committed ones and the other 6 differed only by floating-point rounding (below 1e-9
relative). The version submitted with the report is tagged `final-report`.

The raw data are never changed. The GDP is read from the cloud with
[clouddrift](https://github.com/Cloud-Drift/clouddrift) and region R is cached in `data/interim/` (not in git).

To use another region, change the box in `config/region.yaml` (with its coastal zone and subregions) and rerun.
`src/inference/region_swap.py` tests the quantitative half of this on a second box.

## Repository structure

| Question | Script | Results (`outputs/results/`, `outputs/figures/`) | Report |
|---|---|---|---|
| Study region and data | `src/gridding/trajectory_map.py` | `trajectory_map.png` | Figure 1 |
| Data adequacy | `src/inference/adequacy.py`, `transition_adequacy.py` | `adequacy_*`, `transition_adequacy_*` | Section 2.3, Figure 2 |
| Data quality | `src/inference/data_quality.py`, `data_forensics.py` | `data_quality.png`, `data_forensics.csv` | Section 2.2, Figure 3 |
| Transition table | `src/gridding/transitions.py`, `lags.py` | `data/processed/transitions_tau*d.npz`, `lag_*` | Sections 2.4, 3.1, Figure 4 |
| RQ1: releases and connectivity | `src/linalg/coastal_release.py`, `connectivity.py` | `coastal_release_*`, `connectivity*`, `orange_basin_release.png` | Section 4.3, Figures 8–10, Table 4 |
| RQ1: long-run pattern | `src/linalg/long_run.py` | `long_run_*` | Section 4.4, Figure 11 |
| RQ2: estimators | `src/inference/evaluate.py`, `score_intervals.py`, `prior_sensitivity.py` | `evaluation_*`, `score_intervals*`, `prior_sensitivity*` | Section 4.1, Figure 5, Table 3 |
| RQ2: reliability | `src/inference/bootstrap.py`, `calibration.py`, `year_intervals.py`, `interval_coverage.py`, `markov_check.py` | `calibration*`, `year_intervals_*`, `interval_coverage_*`, `markov_check*` | Section 4.2, Figures 6–7 |
| RQ3: drogue status | `src/inference/drogue_comparison.py` | `drogue_comparison*` | Section 4.5, Figure 12, Table 5 |
| RQ4: seasonality | `src/forecasting/seasonal_forecast.py`, `seasonal_robustness.py`, `seasonal_matrices.py` | `seasonal_*` | Section 4.6, Figures 13–14 |
| Robustness and second region | `src/inference/robustness.py`, `region_swap.py` | `robustness*`, `region_swap*` | Section 4.7, Table 6 |
| App | `src/app/`, `app/` | `app/index.html`, `app_interval_check.csv` | Section 5.1 |

Every figure caption in the report also names the script that made it.

[`project_plan.txt`](project_plan.txt) is the running record of the project: weekly progress, the decision log
(section 7), the forensic review (section 8), and the undrogued comparison and app finish (section 9).
[`logs/ai_use_log.md`](logs/ai_use_log.md) records each AI session, what it produced and how we checked it.
[`docs/proposal/`](docs/proposal/) has the submitted proposal.

## Limitations

- Square cells spread material artificially, a little more with each step (McAdam & van Sebille 2018).
- Chaining assumes a 15-day step forgets its past. This costs little at 2° (chaining predicts 30 and 90-day moves
  better than fitting them directly), but real drifters keep moving in one direction for longer than the chain allows.
- Leaving R is permanent in the matrix, yet about a fifth of the drifters that leave are back inside R at a later
  checkpoint.
- Grounding is not in the main matrix, because grounded drifters stop reporting, so coastal numbers are lower
  bounds. The app's Groundings switch adds it as a second model.
- Drifters seldom come close to shore. A coastal cell is about 200 km across, and the 9 sampled ones have 2–71
  drifters each.
- The data end in October 2022, and the main estimate pools all years and seasons.

## Data sources

Hourly Global Drifter Program data, version 2.01 (Elipot et al. 2016, 2022,
[doi:10.25921/x46c-3620](https://doi.org/10.25921/x46c-3620)), accessed through clouddrift. Region R: 42°S–4°S,
60°W–20°E, 19,269,758 hourly readings from 1,975 drifters, September 1994 to October 2022, of which 1,892 drifters
give the 776,777 usable 15-day moves. Coastline: Natural Earth 1:110m (public domain).
