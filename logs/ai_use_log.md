# AI use log

One entry per session: what was asked, what the AI produced, and what we checked or changed ourselves.

## 2026-09-26 - Weeks 1-2 audit (Claude Code)
- **Asked:** loop the adequacy check over 1/2/3 degree grids; decide how much of a teammate's EDA notebook belongs in the Week 2 EDA; audit Weeks 1-2 (Computational and Quantitative) against the course marking guidance.
- **Produced:** cached region extract with a partial-download check (`src/ingest/load.py`); shared grid module and unit tests (`src/gridding/grid.py`, `tests/test_grid.py`); resolution loop and summary table (`src/inference/adequacy.py`); data-quality EDA (`src/inference/data_quality.py`); revised coverage and sparsity scripts.
- **Checked:** region totals reproduce the recorded 19,269,758 observations / 1,975 drifters; 1 and 2 degree adequacy figures match the earlier manual runs exactly; unit tests pass on synthetic data; all figures inspected by eye.
- **Still to verify by the team:** literature on windage of undrogued drifters before choosing the drogue rule; the SE-based framing of N=100 in the proposal.

## 2026-09-26 - Lecturer feedback and Week 3 planning (Claude Code)
- **Asked:** assess five lecturer suggestions (hexagonal cells, neighbour-informed adequacy, N=100 vs a CI-based N, why the plan subsampled, a coarse-grid baseline), make the drogue call, and turn the Week 3 plan into clear per-stream instructions.
- **Produced:** decision log (project_plan.txt section 7); rewritten Weeks 3-8; adequacy loop extended to 4 and 5 degree; notes on no-pooling and cell area in `adequacy.py` and `grid.py`.
- **Checked:** adequacy at 1-5 degree and at N=385 recomputed on the cached full region; GDP per-drifter variables `end_date` and `typedeath` confirmed in the live dataset; grid tests still pass.
- **Still to verify by the team:** literature on undrogued windage, typical lags, hexagonal grids in drifter studies, and the width of the Benguela coastal strip (listed in Week 3 Business).

## 2026-09-28 - Week 3 Computational: transition table (Claude Code)
- **Asked:** implement Week 3 Computational steps 1-6 from project_plan.txt on a new branch (`transMatrix`): the transition table, the count matrix with a "left R" state, unit tests, and the 4 deg naive P and 2 deg counts.
- **Produced:** `src/ingest/drifter_records.py` (per-drifter `end_date`/`typedeath` and each drifter's continuous stretches of hourly record, cached separately in data/interim/); `src/gridding/transitions.py` (`build_transitions`, `transition_counts`, `row_normalise`); `tests/test_transitions.py`; `PROCESSED_DIR` in `src/common/config.py`. Outputs in data/processed/: `transitions_tau15d.npz` (803,368 drifter-day starts from 1,971 drifters), `transition_counts_2deg_tau15d.npz`, `naive_P_4deg_tau15d.npz`.
- **Changed from the plan (our call, with reasons):**
  - Step 1 asked for `end_date` to decide "left R" vs "stopped reporting". Checked first: `end_date` (the DAC's "last good date") matched the last hourly reading within 1 hour for only 7% of R's drifters and is missing for 131, and 49% of R's drifters have gaps in their hourly record (8,751 gaps, median 20 h). So "still reporting at start + tau" is decided from the continuous record stretches instead. On the real table, the `end_date` rule would have called 18,626 rows exits, 6,577 of which are actually gaps and 922 ended records, and missed 527 real exits (~60% too many exits overall).
  - The per-drifter data went in a new module and cache rather than `load.py` (plan wording), so the existing region cache stays valid and `load.py` keeps one job. Zac asked for and agreed this.
  - Rows whose destination is unknown are kept with an outcome code (record ended / gap) instead of dropped, so their share is reportable and ended records with typedeath = 1 (ran aground) can feed the Week 6 coastline state. They are excluded from counts.
- **Result:** at tau = 15 days, 95.2% of starts are still in R, 1.5% left R, 2.4% record ended, 0.9% gap. At 4 deg, 161 cells have transitions and the median chance of staying in the same cell is 0.53.
- **Checked:** 11 unit tests pass (5 new, on hand-made tracks: stay, move, exit, record ended, gap; rows of naive P sum to 1). Independent brute-force lookup on 300 random rows plus 50 of each rare outcome: 0 mismatches. "Left R" starts lie a median 0.7 deg from the box edge (10.6 deg for the rest). Median 15-day displacement 153 km (~0.12 m/s), consistent with data_quality.py speeds.
- **Still to verify by the team:** start times fixed at 00:00 UTC (one per drifter-day) is a choice, not tested against other hours; "left R" is treated as absorbing in naive P (drifters aren't followed back in), to be revisited if re-entry matters for the long-run maps in Week 4.

## 2026-10-02 - Week 3 Quantitative code and results (Claude, claude.ai chat)
- **Asked:** write the Week 3 quant tasks (estimator, evaluation design, adequacy on transitions, association EDA) against our repo, then walk me through running and interpreting them.
- **Produced by the AI:** `src/inference/transition_data.py`, `split.py`, `estimators.py`, `evaluate.py`, `transition_adequacy.py`, `displacement_eda.py`; `tests/test_quant_week3.py`.
- **Checked by the AI before handover:** tests passed and all four scripts ran end to end on a synthetic table. That caught two bugs, fixed before use: the shrinkage prior gave zero probability to cells no training drifter reached, and the persistence eps search range was too narrow.
- **Changed after running on our real data:** the loader was rewritten to read Computational's `outcome` codes (stayed in R / left R kept; record ended / gap dropped) by importing them from `transitions.py`. `transition_adequacy.py` got a fallback that rebuilds the Week 1 proxy from the transition table, because loading the region cache on my laptop failed with a MemoryError.
- **Checked by me:** ran all four scripts on the real transition table (803,368 rows); 18/18 tests pass. Sanity checks: kept/valid/test counts add up to the 776,777 usable transitions; the split is 20%/20% as designed; median transitions per drifter (15 at 2°, 29.8 at 4°) match the Week 2 cell-crossing times; the table-based Week 1 proxy (44.1%, 80.2%) is close to the official Week 1 figures (46%, 80%). Went through each result with the AI to understand it before committing.
- **Still to do:** confirm season definitions with Business; be able to explain the estimator and evaluation unaided for the poster session.
