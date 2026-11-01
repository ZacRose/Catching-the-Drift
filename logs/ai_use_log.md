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

## 2026-10-01 - Week 3 review and plain-language pass (Claude Code)
- **Asked:** quality review of the Week 3 Computational code and the whole repo; then rewrite the plan, docstrings and figure text in plain language without changing results, and apply the small fixes from the review (not Week 4 work).
- **Produced:** `project_plan.txt` rewritten (same sections, decisions and numbers; added a short glossary and a list of open review issues under Week 4). `extract_region()` now also checks ve, vn and time for failed download chunks. `build_transitions()` checks the region and records caches list the same drifters. Grid sizes in `transitions.py` come from config plus a named baseline constant. Three new tests (re-entering R, record-boundary targets, shuffled input). Self-contained captions on `sparsity_distributions.png` and `data_quality.png`; figure titles use a readable region label (`label` in `region.yaml`). Small doc fixes (how to load the saved table, P^n row sums, north-south crossing time, stale references).
- **Checked:** 14 unit tests pass. The Week 3 outputs rebuild identically from live data, before and after the code changes. Liam's local region cache had 10,944 NaN ve values (one contiguous block, 2 drifters, 2017-19); the live dataset has none for those drifters, so it was a failed chunk. The cache was rebuilt: 0 NaN, every other array identical to before. All scripts rerun; printed results match the plan (adequacy CSV unchanged). Figures inspected by eye.
- **Still to verify by the team:** the open issues listed under Week 4 in the plan (coastal cells, P^n row sums, overlapping 15-day windows, censoring at the coast, missing end positions and end-of-window drogue status, decision 8 evidence not in code).

## 2026-10-02 - Week 3 Quantitative code and results (Claude, claude.ai chat)
- **Asked:** write the Week 3 quant tasks (estimator, evaluation design, adequacy on transitions, association EDA) against our repo, then walk me through running and interpreting them.
- **Produced by the AI:** `src/inference/transition_data.py`, `split.py`, `estimators.py`, `evaluate.py`, `transition_adequacy.py`, `displacement_eda.py`; `tests/test_quant_week3.py`.
- **Checked by the AI before handover:** tests passed and all four scripts ran end to end on a synthetic table. That caught two bugs, fixed before use: the shrinkage prior gave zero probability to cells no training drifter reached, and the persistence eps search range was too narrow.
- **Changed after running on our real data:** the loader was rewritten to read Computational's `outcome` codes (stayed in R / left R kept; record ended / gap dropped) by importing them from `transitions.py`. `transition_adequacy.py` got a fallback that rebuilds the Week 1 proxy from the transition table, because loading the region cache on my laptop failed with a MemoryError.
- **Checked by me:** ran all four scripts on the real transition table (803,368 rows); 18/18 tests pass. Sanity checks: kept/valid/test counts add up to the 776,777 usable transitions; the split is 20%/20% as designed; median transitions per drifter (15 at 2°, 29.8 at 4°) match the Week 2 cell-crossing times; the table-based Week 1 proxy (44.1%, 80.2%) is close to the official Week 1 figures (46%, 80%). Went through each result with the AI to understand it before committing.
- **Still to do:** confirm season definitions with Business; be able to explain the estimator and evaluation unaided for the poster session.

## 2026-10-03 - Combining Week 3 work into one branch (Claude Code)
- **Asked:** bring Vidisha's Week 3 Quantitative work (`week3-shrinkage`) and Stella's undrogued-only check (`Stella`) onto Liam's plain-language branch (`readability-week3`) as a new branch `week3-unified`, review both against the Week 3 plan, and write any plan and log additions in the same plain style.
- **Produced:**
  - Vidisha's files brought in unchanged first, then tidied without changing any result: `TAU_DAYS`, `PROCESSED_DIR`, the grid sizes and the naive P function now come from the shared code instead of being redefined; `load_split()` refuses a table with drifters missing from the saved split; plain-English docstrings; readable titles and self-contained captions on `alpha_sensitivity.png` and `transition_adequacy_map.png`; one new test (22 in total).
  - Stella's change re-applied by hand to `data_quality.py` (her branch started before the Week 3 work, so merging it would have clashed with the rewritten plan).
  - `project_plan.txt`: Week 3 Quantitative section marked done with results, four glossary terms, decision 6a reworded, Week 4 open issues g-k (from an AI code review), Week 8 drogue check updated.
- **Checked:** Vidisha's CSVs reproduce exactly on the combined branch, before and after the tidy-up. The only change is that the adequacy CSV's Week 1 column now comes from the hourly cache (46.0% at 2 deg, matching the official Week 1 figure) instead of her laptop fallback (44.1%). Per-cell drifter counts equal the `start_drifters` arrays saved by Computational. Stella's 22.4% (2 deg, undrogued only) reproduces. All figures inspected by eye.
- **Changed from the branches (our call, with reasons):** Stella's 6a said "literature review confirms" undrogued drifters behave like oil. An AI literature search found support (Novelli et al. 2020, Mar. Pollut. Bull. 150:110805; Grodsky et al. 2011, GRL 38:L10606; Laxague et al. 2018, GRL 45:245-249) but from a few Gulf of Mexico studies, mostly read as abstracts only, so the wording was softened. It also found that GDP debris and transition-matrix studies pool both drifter types (van Sebille et al. 2012, ERL 7:044040; McAdam & van Sebille 2018, JGR Oceans 123:514-532) or check them separately (Maximenko et al. 2012, Mar. Pollut. Bull. 65:51-62).
- **Still to verify by the team:** read those papers in full before citing them; open issues g-k (fairer baseline, scores for thin vs well-sampled cells, intervals on scores, counting units for alpha, methods wording for the prior).

## 2026-10-03 - Review of the Business literature and significance drafts (Claude Code)
- **Asked:** review Michelle's Business drafts (literature review, case review, background, significance; written in Google Docs) and add them to the repo before the README is written.
- **Produced:** `docs/literature/business_literature_review.md`, `docs/report/background.md`, `docs/proposal/significance.md`. Michelle's text is kept unchanged apart from formatting and one typo, with review notes and suggested wording under each. Also: full APA references, and the Week 3 Business status and region justification pointers in `project_plan.txt`.
- **Checked:** an AI literature search checked each source against Crossref metadata, abstracts and the BCC web page. Full texts were blocked, so nothing was read in full. Results: Hutchings et al. 2009 is consistent with its abstract, though the four components need confirming. Drouin & Lozier 2019 abstract names only GDP drifters and OSCAR, with no subsurface floats, and its headline is a sizeable Drake Passage share of Benguela water. The BCC page lists the activities but doesn't call shipping "major"; it does flag international shipping routes as a pollution risk. Novelli et al. 2020 is accurate, but the drifters were probably CARTHE, not GDP buoys.
- **Still to verify by the team:** read all four papers in full before the README cites them; source the offshore-oil point; add the client case; confirm the GDP drogue depth from `gdp-overview.pdf`.

## 2026-10-06 - Proposal Draft 1 review, Draft 2, and the release-map figure (Claude Code)
- **Asked:** mark the team's Proposal Draft 1 (PDF) against the four proposal criteria (clarity, significance, likelihood of success, feasibility); check every citation and number; suggest how to fit figures into 3 pages; judge whether it reads as human-written; say whether the data counts will change. Then write a 3-page Draft 2 with every change highlighted (kept outside the repo, in the workspace folder), and add the code for its new figure to the repo.
- **Produced:**
  - Draft 2 PDFs (highlighted and clean) and a change list, outside the repo. Main changes: named users and an honest planning-tool scope; an Objectives paragraph; the 3.3% unknown destinations framed as the missing data; a softer literature-gap claim; RQ3 caveat; a Week x stream timeline; the bar chart replaced by a three-panel map.
  - `src/linalg/release_map.py` (Figure 1: drifters per 2 deg cell; where material released in one 4 deg cell is after P^6 and P^24); `release_point` in `config/region.yaml`; `tests/test_release_map.py` (3 tests on a toy matrix). Output: `outputs/figures/release_map_tau15d.png`. Branch `week4-release-map`, not yet committed.
- **Checked:**
  - Citations read in full where available: McAdam & van Sebille 2018 (Europe PMC full text: §1, §2.1, §2.2, eq. 2, §3.1 all support our sentences); Drouin & Lozier 2019 (NSF PAR PDF: pp. 7082, 7085, 7088-89, 7091-92 support ours, with two wording fixes: the Benguela "sources its waters from" those inflows, and they keep each drifter only until drogue loss); van Sebille et al. 2012 (IOP PDF: §2.1 pools drifters; the coastal release is in §2.3, not §2.2); the BCC page and the World Oil article (all claims present; World Oil says "one of the most active deepwater exploration areas globally", not "global hotspot"). Hutchings et al. 2009 (19-34 deg S) and Novelli et al. 2020 (wave tank, 25% vs 5% landfall) checked against abstracts only. All [1] claims checked against `gdp-overview.pdf`.
  - Every number in the draft matches the plan and committed CSVs, except one wording fix: tau is the time a median-speed drogued drifter takes to cross a 2 deg cell north to south (not "the median crossing time").
  - Release map, release cell 26-30 deg S, 12-16 deg E: 34.9% still in the cell after 15 days; centre of the remaining material at 24.6 deg S, 3.1 deg E after 90 days and 24.9 deg S, 15.0 deg W after 360 days; 1.3% left R after a year; no mass lost to row-less cells from this start. 25/25 tests pass. Figure inspected by eye.
- **Not verified:** Garcia-Pineda et al. 2020 (the paper was blocked): the "drogued at approximately 0.5 metres" claim is unconfirmed, and one search snippet mentions currents at 1 m.
- **Still to verify by the team:** the [9] drogue depth (read §2.1 via the library or reword); the dataset access date; stream owners in the timeline; the release point is provisional until open issue (a) defines the coastal cells.

## 2026-10-12 - Poster study-region map (Claude Code)
- **Asked:** replace the poster's plain box map with drifter tracks plus the box, following a teammate's peer feedback, so one figure shows the data, the region and the Benguela flow.
- **Produced:** `src/gridding/trajectory_map.py` (daily tracks, mean-flow arrows on 4° cells with at least 100 drifters, region box), `tests/test_trajectory_map.py` (2 tests), `outputs/figures/trajectory_map.png`, and the Natural Earth 1:110m land outline in `data/external/` (cartopy could not be installed on Python 3.14, so the coastline is drawn with plain matplotlib).
- **Checked:** the daily readings count (803,368 from 1,971 drifters) matches the Week 3 drifter-day starts. Arrows near Namibia point northwest and along 40°S point east, which matches the expected Benguela and South Atlantic Current directions. Tracks stop at the box edge because the region cache only holds readings inside R. 24 tests pass.
- **Still to verify by the team:** whether the "Benguela Current" label position and the arrow length scale read well at poster size.

## 2026-10-21 - Catch-up after the proposal and poster, and a repo tidy-up (Claude Code)
- **Asked:** read the repo plus the submitted proposal and poster, bring `project_plan.txt` up to date, then tidy the codebase without losing any substance: fix anything inaccurate, and remove or move only what is certainly safe.
- **Produced:**
  - `project_plan.txt`: team names and streams; Week 4 and Week 5 status blocks (research questions, objectives, which open issues the proposal settled, the carry-over to Week 6); Weeks 6-7 matched to the proposal timeline; "quasi-stationary distribution" where the plan said "stationary"; glossary, typo and status fixes.
  - Status notes on the three Business drafts, saying which review notes the proposal applied.
  - `trajectory_map.mean_flow` reuses `grid.per_cell_counts`, and `transitions.save_grid_result` reuses `BOX_KEYS`.
  - The submitted proposal and poster moved to `docs/proposal/` and `docs/poster/`.
  - `.gitkeep` files removed from folders that already hold tracked files.
- **Checked:** 24 tests pass. The old and new `mean_flow` give identical arrays on 200,000 synthetic readings. Every number added to the plan was checked against the proposal PDF or the committed outputs. The leftover `data/interim/*.npz.bak` was confirmed to be the faulty pre-rebuild cache (10,944 NaN `ve`), but it was not deleted.
- **Still to verify by the team:** that the Week 4/5 status blocks and the Week 6 carry-over match what the team agreed; and the person-to-stream mapping (proposal names vs poster nicknames).

## 2026-10-21 - Week 6 Quantitative (Claude Code)
- **Asked:** implement the Week 6 Quantitative task in full: a neighbour-only baseline and test scores split by cells under and over 100 drifters (open issues g, h), fit the 2 deg shrinkage model, and map the posterior effective sample size in drifters beside the naive count (issue j).
- **Produced:** `estimators.neighbour_P`; in `evaluate.py`, `tune_neighbour`, `tune_alpha` and `seen_cells`, plus a `cells` column in `evaluation_tau15d.csv`; `src/inference/shrinkage_fit.py` writing `data/processed/shrinkage_P_2deg_tau15d.npz`, `effective_sample_size_tau15d.csv` and `effective_sample_size_map.png`; `tests/test_quant_week6.py` (5 tests); the "empirical Bayes" note in `estimators.py` (issue k).
- **Choices made by the AI (for the team to confirm):**
  - the sparse/well-sampled split uses the adequacy map's all-drifter count, not the training set's (which would leave only 22% of cells adequate);
  - the neighbour baseline's leak goes to every state, because a first version leaked only to cells seen in training and gave -inf to a few test moves;
  - effective drifters = own drifters + alpha / median moves per drifter (15).
- **Checked:** 29 tests pass. The Week 3 "all" rows of the evaluation CSV and `alpha_sensitivity.png` reproduce exactly. The refitted counts match `transition_counts_2deg_tau15d.npz` exactly. Every row of the fitted P sums to 1 with no zeros. The figure was inspected by eye.
- **Still to verify by the team:** the split definition and the effective-sample-size formula. Results: shrinkage beats neighbour-only by about 0.9 per move, and the prior is worth only 3.75 drifters (41.3% -> 44.0% adequate). None of the score gaps has an interval yet (issue i).
- **End of session (2026-10-21):** nothing is committed. Everything from both 2026-10-21 entries is uncommitted on `week5-poster-map`, and Liam will split it into week5 and `week6-quant` commits. `project_plan.txt` (the "Where we are" paragraph and the "Week 6 status" block) and the workspace `CLAUDE.md` record the state for the next session. 29 tests pass.

## 2026-10-29 - Week 6 Computational: coastal cells, drifter bootstrap, coastal release maps (Claude Code)
- **Asked (Zac):** pull the latest repo; explain the bootstrap; then implement Week 6 Computational from `project_plan.txt` in full, without mistakes, and update this log and the plan. Zac asked not to bring in `week7-quant`.
- **Produced (branch `week6-computational`, from main):**
  - Liam's `week4-release-map` merged in, since Week 6 redraws its maps. The only conflict was this log; his 2026-10-06 entry now sits in date order. His Figure 1 numbers reproduce exactly.
  - `src/gridding/coast.py` and `coastal_zone` in `config/region.yaml` (issue a): coastal cells are those between 34S and 18S that the coastline passes through on the Natural Earth outline. Gives 13 cells at 2 deg, 9 with drifters. It also reports issue d (groundings at the coast). Outputs: `coastal_cells.csv`, `coastal_outcomes_tau15d.csv`.
  - `src/inference/bootstrap.py` (issues c, i): resamples whole drifters; 1000 replicates; a fast weighted count per replicate; per-entry SE and 95% intervals for the 2 deg shrinkage P and the 4 deg naive P, computed in row blocks to stay under 250 MB. Outputs: `*_bootstrap.npz` in data/processed, `bootstrap_P_summary_tau15d.csv`, `bootstrap_uncertainty_map.png`.
  - `src/inference/score_intervals.py` (issue i): intervals on every held-out score and on every paired model difference, from resampling the test drifters. `evaluate.py` gained `fit_models()`, split out of `evaluate_grid()`. Outputs: `score_intervals_tau15d.csv`, `score_intervals.png`.
  - `src/linalg/coastal_release.py`: P, P^6 and P^24 from the coastal cells for the 4 deg naive and 2 deg shrinkage matrices, each with bootstrap intervals. `release_map.propagate` can now spread a release over several cells. Outputs: `coastal_release_tau15d.csv`, `coastal_release_tau15d.png`.
  - `tests/test_comp_week6.py` (10 tests); docstring updates in `estimators.py`, `transitions.py` and `release_map.py`; plan updated (where we are, the Week 6 Computational status block, decision 4 runtime, stale branch notes).
- **Choices made by the AI (for the team to confirm):**
  - coastal band 34S-18S, from the proposal's cited 19-34S upwelling, and the "coastline passes through" rule. Drifters rarely reach the shore, so these cells are thin (2-71 drifters).
  - no "ran aground" state: only 5 drifters ran aground in the coastal cells (32 starts), too few to estimate it, so the share is reported instead;
  - the release is spread evenly over the coastal cells that have drifters;
  - alpha is held at 56.2 in the bootstrap, with the prior rebuilt each replicate; percentile intervals.
- **Checked:**
  - 42 tests pass, 10 of them new. One new test failed at first because its own reference calculation was wrong, not the code. Another exposed a float32 rounding in the row distances, which are now measured in float64.
  - Two replicates were rebuilt by literally copying the drawn drifters' rows: counts identical to the fast method at 2 and 4 deg. Replicates are reproducible, and every replicate row sums to 1.
  - The refitted 2 deg P matches the saved product. Coastal cell drifter counts match the committed `start_drifters`. `evaluation_tau15d.csv` is byte-identical after the `evaluate.py` change, and all 54 point estimates in `score_intervals_tau15d.csv` match it exactly. Cell totals (596 sampled, 247 adequate at 2 deg; 161/127 at 4 deg) match the adequacy outputs. All figures were inspected by eye.
  - Regenerated `alpha_sensitivity.png` and `release_map_tau15d.png` differ from the committed files only in rendering (a 1-pixel height difference, anti-aliasing), so the committed versions were kept.
- **Results:**
  - The bootstrap SE is about 3 times the SE from treating moves as independent.
  - Decision 3 holds: 97% of adequate 2 deg cells and all adequate 4 deg cells have every SE at or below 0.05.
  - Shrinkage beats neighbour-only by 0.90 (0.83 to 0.98). On the moves naive can score it is reliably worse, by 0.008 (0.0065 to 0.010).
  - Coastal releases mostly leave the coastal zone within 3 months (0.63% left) and are centred near 23S, 17W after a year, with 0.89% (0.61-1.5) gone from the box.
- **Not done:** "fix issues raised at the poster": the feedback isn't in the repo. Nothing is committed yet.
- **Still to verify by the team:**
  - the four choices above;
  - the bootstrap understates uncertainty for cells with only 1-2 drifters (documented in `bootstrap.py`);
  - environment: the tests and scripts were run in the `DATA3001` conda environment (Python 3.11), not the Python 3.14 that `environment.yml` names.

## 2026-10-29 - Week 7 Computational: transition tables at other time steps (Claude Code)
- **Asked (Liam):** implement the Week 7 Computational task in full ("build the transition table at other time steps"), make every script, output and plan entry clear to the team, and update the plan, CLAUDE.md and this log. `week7-quant` is pushed but deliberately not merged; the team will merge it themselves.
- **Produced (branch `week7-comp`, from main at pull request #11):**
  - `src/gridding/lags.py`: builds the table at 5, 10, 15, 30, 60 and 90 days with the existing `build_transitions`, runs three checks (rebuild of the committed 15-day table, composition a + b, brute-force lookups), and writes `outputs/results/lag_checks.csv`, `lag_summary.csv` and `outputs/figures/lag_summary.png`.
  - `data/processed/transitions_tau{5,10,30,60,90}d.npz` (about 11 MB each, same format as the 15-day table). The 15-day file is untouched.
  - `transitions.table_path()`, so every lag's file is named in one place; `tests/test_comp_week7.py` (5 tests); the plan's "Where we are", Week 4 and Week 6 status lines (now on main), a "Week 7 status - Computational" block, a Week 8 note and decision 4's runtime.
- **Choices made by the AI (for the team to confirm):**
  - the lags: 30 days for the Markov check, 90 days for the 3-month headline map, 60 days in between, and 5 and 10 days for the Week 8 shorter-step and 1 deg checks;
  - committing the five new tables (about 56 MB in all) so teammates without the region cache can use them;
  - adding the 1 deg grid to the summary, because Week 8 reruns at 1 deg;
  - not fitting P at the new lags, which is Week 8's job.
- **Checked:**
  - Every check passed: the rebuilt 15-day table is identical to the committed one; 0 composition mismatches over 3.0 million checked starts; 0 brute-force mismatches in 600 rows. The 15-day summary row reproduces the Week 3 numbers.
  - `week7-quant`'s `markov_check.chain_transitions` was run, in a temporary worktree that has since been removed, against the new 30-day table. They agree on 99.75% of rows, and every difference is a case its docstring says chaining can't settle (exit share 2.22% chained vs 2.11% real).
  - A dry-run merge (`git merge-tree`) shows `week7-quant` already conflicts with main only in `project_plan.txt` and this log, and its code merges cleanly.
  - 47 tests pass. The figure was inspected by eye, and its first version had a caption overlapping the axis labels, which was fixed. The figure's colours passed a colour-blind check, and each line also has its own marker shape.
- **Results:** longer steps lose starts to ended records (2.4% at 15 days, 4.9% at 30, 14.5% at 90, only 1.2 points of which is the dataset stopping), so long-lag matrices rest on longer-lived drifters. Adequate 2 deg cells drop from 41% to 31% between 15 and 90 days. 1 deg is under 3% adequate at every lag. At 5 days, 79% of 4 deg moves stay in their cell.
- **Still to verify by the team:** the four choices above; and the environment. Everything ran in the system Python 3.14.5 on Liam's machine (Week 6 used a Python 3.11 conda environment). The committed 15-day table rebuilt identically here, but the scripts weren't rerun under 3.11.

## 2026-11-01 - Week 7 Quantitative: Markov check, long-run pattern, mixing time, prior sensitivity, draft product (Claude, claude.ai)
- **Asked (Vidisha):** implement Week 7 Quantitative from `project_plan.txt` on top of the `week7-comp` work (shared as a zip), and update this log and the plan. Partway through, Vidisha asked to set the older `week7-quant` branch aside and put Week 7 and Week 8 Quantitative on one new branch (Week 8 has its own entry below).
- **Produced (branch `week7-8-quant`, on top of `week7-comp`):**
  - `src/inference/markov_check.py`: P(15 d)^2 vs P(30 d) and P(15 d)^6 vs P(90 d) at 4 deg naive and 2 deg shrinkage, using the real 30- and 90-day tables, fitted on the same starts at both lags. A held-out score test with intervals from resampling test drifters, and per-cell differences (stay, exit, distance) with 1000-replicate drifter bootstrap intervals. Outputs: `markov_check_tau15d.csv`, `markov_check.png`.
  - `src/linalg/long_run.py`: quasi-stationary distribution per unit area, lambda_1, time in the box, half-life, spectral and total-variation mixing times, for 4 deg naive, 2 deg naive and 2 deg shrinkage, with 1000-replicate bootstrap intervals. Outputs: `long_run_tau15d.csv`, `long_run_qsd.png`, `long_run_*_tau15d.npz`.
  - `src/inference/prior_sensitivity.py`: every headline number at alpha x 0.1-10, with an even-split prior and with no prior, shaded against the product's bootstrap intervals. Outputs: `prior_sensitivity_tau15d.csv`, `prior_sensitivity.png`.
  - `src/linalg/product.py` (the Week 7 "draft product" loader); `tests/test_quant_week7.py` (10 tests); an optional `rows` mask in `load_transitions()` and `split="even"` in `prior_rows()` (defaults unchanged); the plan's "Where we are" line and a "Week 7 status - Quantitative" block.
- **Choices made by the AI (for the team to confirm):**
  - the held-out score is the main Markov test. A split-half noise floor for the total variation was tried in a prototype and dropped: the chained matrix is smoother than a direct one, so the floor came out larger than the chained-vs-direct gap (0.21 vs 0.19 at 2 deg / 30 d), which would wrongly suggest no Markov gap at all;
  - matched starts across lags (otherwise the 90-day matrix rests only on long-lived drifters);
  - the total variation is reported without an interval, because on the first run the full-data value fell below its own bootstrap interval (resampling noise only adds to it);
  - QSD over cells with drifters only; t_mix at total variation 0.25 from the coastal release as the headline mixing time;
  - the 4 deg naive long-run intervals reported with and without replicates where a cell becomes a trap.
- **Checked:**
  - 57 tests pass (10 new). The new tests check against closed-form answers (eigenvectors, the two-cell mixing time 0.5 (1 - 2a)^n, an exact Chapman-Kolmogorov chain gives 0) and all passed on the first run.
  - The QSD from the eigen-decomposition and from power iteration agree, checked in the script for every model. `prior_sensitivity.py` checks that alpha x 1 reproduces the saved product.
  - The shrinkage product refits identically. `evaluation_tau15d.csv` and `effective_sample_size_tau15d.csv` are byte-identical after the code changes. `score_intervals_tau15d.csv` differs only at the 15th significant figure (numpy 2.5 here vs the team's), so the committed versions are kept.
  - The bootstrap first gave nonsense 4 deg naive intervals (lambda_1 up to 1, centre at 12S 14E). The AI traced this to one 2-drifter cell off Angola and reported it rather than trimming it away. All three figures were inspected by eye; layout fixes were made to two of them (a floating colour bar, one off-scale point squashing a scatter).
  - The product demo from 28S 14E is consistent with the Week 4 Figure 1 numbers.
- **Results:**
  - At 2 deg, P(15 d)^k predicts held-out 30- and 90-day moves better than direct fits (-0.040 and -0.100 per move, direct minus chained). It over-states exits per cell by about 0.8 points at 30 days and 2.8 at 90.
  - The long-run pattern is centred at 30.2S, 16.4W, with half of it in 11% of the area. Expected time in the box 16 years (14-19); the coastal start is forgotten after 2.7 years (2.4-2.9).
  - The long run doesn't depend on the prior, but the coastal numbers do at alpha x 10 or x 0.1.
- **Environment:** run in Python 3.13 with numpy 2.5 and scipy 1.18 (`long_run.py` needs scipy, now added to `environment.yml`; clouddrift already installs it). Not rerun in the team's conda environment.
- **Still to verify by the team:** the choices above; the "garbage patch" comparison needs a citation before it goes in the report.

## 2026-11-01 - Week 8 Quantitative: robustness and region swap (Claude, claude.ai)
- **Asked (Vidisha):** set the older `week7-quant` branch aside and do Week 7 and Week 8 Quantitative together on one branch from the `week7-comp` zip.
- **Produced (branch `week7-8-quant`):**
  - `src/inference/robustness.py`: 13 settings (grid 1/2/4 deg; lags 5-90 days; undrogued only and drogued only; 2010+), each changing one thing and refitted as the product. Headline numbers with 200-replicate drifter bootstrap intervals (50 at 1 deg). Outputs: `robustness_tau15d.csv`, `robustness.png`.
  - `src/inference/region_swap.py`: a second box (Brazil-Malvinas Confluence) built exactly from the existing tables, with every quant step run on it unchanged. Outputs: `region_swap_tau15d.csv`, `region_swap.png`.
  - `tests/test_quant_week8.py` (4 tests); a "Week 8 status - Quantitative" block in the plan; branch wording in "Where we are".
- **Choices made by the AI (for the team to confirm):**
  - the settings list;
  - fewer bootstrap replicates (200; 50 at 1 deg) to keep the run to about 8 minutes;
  - shrinkage at 1 deg towards 2 deg (one level up, as 2 towards 4);
  - drogue status taken at the start of each move;
  - a region swap inside our extract instead of a new extract (which needs Computational and the cloud store);
  - for naive P, intervals over the replicates without a trap.
- **Checked:**
  - The base setting reproduces the product's alpha and the Week 6 and 7 numbers exactly.
  - The surprising 1.2% adequacy for 2 deg 2010+ was confirmed by an independent count (plain floor arithmetic, median 52 drifters per cell).
  - The new tests caught a wrong claim in the AI's own docstring: the 4 deg settings change one thing from the 4 deg baseline, not from the product. Both the docstring and the test were fixed.
  - A first region-swap check reported FAIL because it expected no zeros in the absorbing "left the box" row (which is [0, ..., 0, 1] by design). The check was wrong, not the pipeline, and was narrowed to cell rows.
  - A first draft of the plan's Week 8 summary had two claims that didn't match the CSV (the largest pattern difference, and which 2010+ numbers fall outside the interval). Both were corrected against the CSV.
  - The figures were inspected by eye. 61 tests pass.
- **Results:**
  - The long-run pattern is the same in every setting (centre 29-30S, 16-21W).
  - Drogue status changes the coastal release most: undrogued material is about 10 deg further west after a year and 7-8 times as likely to have left the box. This answers RQ3, but drogued-only is thin.
  - Time in the box rises with the lag (10 to 26 years). 1 deg rests on its prior (2% of cells adequate), and so does a 2 deg 2010+ matrix (1%).
  - The code ran unchanged on the second box.
- **Still to verify by the team:** the choices above; whether the report frames the product as pooled "floating material and water" with the undrogued run as the floating-material case; the 1:10m coastline for 1 deg.
