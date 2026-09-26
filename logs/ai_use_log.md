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
