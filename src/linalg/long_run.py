"""
long_run.py

Related task: Week 7 Quantitative - the long-run pattern and the mixing time (proposal section 4,
MATH2501): where does material that stays in the box end up, how long does it stay, and how fast
does the system forget where it started?

Why not the ordinary stationary distribution: "left R" is absorbing (decision 11), so in the very
long run everything has left and the stationary distribution of P is all on "left R". The useful
question is where the material that is STILL in the box sits, once it has been there long enough to
forget its release point. That is the quasi-stationary distribution (QSD).

The maths, in plain terms. Let Q be P with the "left R" row and column removed, and also the cells
with no drifters (their rows are guesses: absent for naive P, the 4 deg prior alone for the
shrinkage P). Q's rows sum to a little under 1, the shortfall being what leaves each step.
- Q's largest eigenvalue lambda_1 (real and positive, Perron-Frobenius) is the share of the
  material that stays in each step once the pattern has settled. 1 - lambda_1 is lost per step, so
  the expected time left in the box is 1 / (1 - lambda_1) steps and the half-life is
  ln 2 / (-ln lambda_1) steps.
- Its left eigenvector q (scaled to sum to 1) is the QSD: x Q^n / (total still in) -> q from any
  start x. Divided by cell area it is a density, shown as "times the box average"; high values
  mark where floating material gathers (decision 1: cells shrink with latitude, so this matters).
- Mixing time. The gap to the next eigenvalue sets how fast any start converges to q: the
  distance shrinks by about |lambda_2| / lambda_1 per step, so 1 / ln(lambda_1 / |lambda_2|) steps
  is the e-folding time ("spectral" mixing time). That is an asymptotic rate, so the time is also
  measured directly: from a start x, the total variation distance between x Q^n / (total) and q,
  and t_mix = the first n at which it is at most 1/4 (the usual cut-off, Levin, Peres & Wilmer).
  Reported from the Benguela coastal release (coastal_release.py), and for every start cell (median
  and worst case).

Three matrices, as the naive-vs-regularised comparison for the long run: the 4 deg naive P (the
baseline), the 2 deg naive P, and the 2 deg shrinkage P (the product). Intervals: N_BOOT drifter
bootstrap replicates (bootstrap.replicate_matrices), each refitting P exactly as the original;
the set of cells kept is fixed at the full-data one, so a replicate that draws no drifters for a
thin naive cell gives it an empty row there (material arriving is lost), as in coastal_release.py.

What the bootstrap showed (2026-11-01): the 4 deg naive P has a cell off Angola (cell 158, about
12S 14E) with only 2 drifters, one of which loiters there. In 21% of replicates only the loiterer is
drawn, the cell keeps everything that reaches it (lambda_1 = 1) and the long-run pattern collapses
into it, so the 4 deg naive intervals are also given over the replicates without such a trap. The
2 deg shrinkage P has no traps (its prior always lets some material out), but the same loitering
drifter (2 deg cell 636, about 11S 13E) makes a slow mode in about 9% of replicates, which is the
long upper tail of lambda_2 and of the spectral mixing time. The mixing time measured from the coast
(years_tmix_from_coast) is not affected.

Caveats for the report:
- Q assumes the 15-day step is Markov; markov_check.py measures how far that holds.
- Material leaving the box never comes back (decision 11b), so lambda_1 is a lower bound on how
  much stays and the time in the box is a lower bound.
- The QSD describes material that has been in the box for years; at 15 days to a year the release
  maps (coastal_release.py) are the right tool.

Outputs:
- outputs/results/long_run_tau15d.csv (every number with its interval);
- data/processed/long_run_<model>_tau15d.npz: QSD, density, 95% intervals per cell, kept cells and
  grid metadata (transitions.save_grid_result), for the product and the 4 deg baseline;
- outputs/figures/long_run_qsd.png.

Run: python3 -m src.linalg.long_run   (about 6 minutes, mostly the 2 deg bootstrap; needs
shrinkage_P_2deg_tau15d.npz)
"""
import csv
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.linalg

from src.common.config import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.coast import coastal_cells, land_fraction, land_paths
from src.gridding.grid import grid_edges, per_cell_counts
from src.gridding.trajectory_map import _degrees, draw_land
from src.gridding.transitions import BASELINE_CELL_SIZE, TAU_DAYS, save_grid_result
from src.inference.bootstrap import LEVEL, N_BOOT, percentile_interval, replicate_matrices, shrinkage_fit
from src.inference.estimators import count_matrix, naive_P
from src.inference.transition_data import load_transitions, parent_states, to_states

TMIX_EPS = 0.25  # total variation cut-off for t_mix
MAX_STEPS = 24 * 30  # 30 years of 15-day steps; t_mix beyond this is reported as not reached
EARTH_RADIUS_KM = 6371.0
DAYS_PER_YEAR = 365.25


def cell_areas_km2(region, cell_size):
    """Area of every cell inside the box (km^2); cells overhanging the box count only the part inside."""
    lat_edges, lon_edges = grid_edges(region, cell_size)
    lat_edges = np.clip(lat_edges, region["lat_min"], region["lat_max"])
    lon_max = region["lon_max"] + 360 if region["lon_min"] > region["lon_max"] else region["lon_max"]
    lon_edges = np.clip(lon_edges, region["lon_min"], lon_max)
    band = np.diff(np.sin(np.deg2rad(lat_edges)))
    width = np.deg2rad(np.diff(lon_edges))
    return (EARTH_RADIUS_KM ** 2 * band[:, None] * width[None, :]).ravel()


def substochastic(P, keep):
    """Q: P among the kept cells only (exit and every other cell dropped)."""
    return P[np.ix_(keep, keep)]


def quasi_stationary(Q):
    """Leading eigenvalue, QSD and second-largest eigenvalue modulus of a substochastic matrix.

    Returns:
        (lambda_1, q, |lambda_2|): q is the left eigenvector for lambda_1, non-negative, summing to 1.
    """
    values, vectors = scipy.linalg.eig(Q.T, check_finite=False)
    order = np.argsort(-np.abs(values))
    lam1 = values[order[0]]
    if abs(lam1.imag) > 1e-10 or lam1.real <= 0:
        raise ValueError(f"Leading eigenvalue {lam1} is not real and positive; is Q a substochastic matrix?")
    q = np.real(vectors[:, order[0]])
    q = np.abs(q) / np.abs(q).sum()  # Perron vector: one sign throughout, up to rounding
    return float(lam1.real), q, float(np.abs(values[order[1]])) if values.size > 1 else 0.0


def leading_eigenvalues(Q):
    """lambda_1 and |lambda_2| only (no eigenvectors: cheaper, for the bootstrap)."""
    values = scipy.linalg.eigvals(Q, check_finite=False)
    mods = np.sort(np.abs(values))[::-1]
    return float(mods[0]), float(mods[1]) if mods.size > 1 else 0.0


def qsd_power(Q, x0=None, tol=1e-12, max_iter=20000):
    """QSD and lambda_1 by power iteration (x <- x Q / |x Q|); cheap, for the bootstrap.

    x0: starting guess (e.g. the full-data QSD, so a replicate converges in fewer steps); uniform if None.
    """
    x = np.full(Q.shape[0], 1.0 / Q.shape[0]) if x0 is None else np.asarray(x0, dtype=float) / np.sum(x0)
    lam = 0.0
    for _ in range(max_iter):
        y = x @ Q
        lam = y.sum()
        y /= lam
        if np.abs(y - x).sum() < tol:
            return float(lam), y
        x = y
    warnings.warn("power iteration for the QSD did not converge")
    return float(lam), x


def tv_to_qsd(Q, q, start, max_steps=MAX_STEPS):
    """Total variation between the conditional distribution (still in the box) and q, at steps 1..max_steps.

    start: a distribution over the kept cells, or a (n_starts x n_kept) array of them (one per row).
    Returns:
        array (max_steps,) or (n_starts, max_steps).
    """
    x = np.atleast_2d(np.asarray(start, dtype=float))
    out = np.empty((x.shape[0], max_steps))
    for n in range(max_steps):
        x = x @ Q
        with np.errstate(invalid="ignore", divide="ignore"):  # a start whose material has all left: NaN, never mixes
            x /= x.sum(axis=1, keepdims=True)
        out[:, n] = 0.5 * np.abs(x - q).sum(axis=1)
    return out[0] if np.ndim(start) == 1 else out


def steps_to_mix(Q, q, start, eps=TMIX_EPS, max_steps=MAX_STEPS):
    """First step at which the conditional distribution from one start is within eps of q (NaN if never).
    Same as first_below(tv_to_qsd(...)) but stops as soon as it gets there (for the bootstrap)."""
    x = np.asarray(start, dtype=float)
    for n in range(1, max_steps + 1):
        x = x @ Q
        x /= x.sum()
        if 0.5 * np.abs(x - q).sum() <= eps:
            return float(n)
    return float("nan")


def first_below(tv, eps=TMIX_EPS):
    """First step (1-based) at which tv <= eps, along the last axis; NaN if never."""
    tv = np.atleast_2d(tv)
    hit = tv <= eps
    steps = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, np.nan).astype(float)
    return steps[0] if steps.size == 1 else steps


def years(steps):
    """15-day steps -> years."""
    return np.asarray(steps, dtype=float) * TAU_DAYS / DAYS_PER_YEAR


def centre(q, keep, region, cell_size):
    """Probability-weighted mean (lat, lon) of the kept cells' centres."""
    lat_edges, lon_edges = grid_edges(region, cell_size)
    n_lon = len(lon_edges) - 1
    row, col = np.divmod(keep, n_lon)
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])[row]
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])[col]
    return float((q * lat_c).sum()), float((q * lon_c).sum())


def area_holding_half(q, area):
    """Share of the kept area that holds half the QSD: the densest cells first."""
    density = q / area
    order = np.argsort(-density)
    reach = np.searchsorted(np.cumsum(q[order]), 0.5) + 1
    return float(area[order[:reach]].sum() / area.sum())


def summarise(P, keep, area, release, region, cell_size, full=True, q_start=None):
    """Every long-run quantity for one P.

    full=False (the bootstrap) uses power iteration from q_start and eigenvalues only, and skips the
    per-start-cell mixing times.
    """
    Q = substochastic(P, keep)
    if full:
        lam1, q, lam2 = quasi_stationary(Q)
    else:
        lam1, q = qsd_power(Q, q_start)
        _, lam2 = leading_eigenvalues(Q)
    n_cells = P.shape[0] - 1
    loss = 1 - lam1
    to_exit = float(q @ P[keep, n_cells])
    lat, lon = centre(q, keep, region, cell_size)
    x0 = np.zeros(keep.size)
    x0[np.isin(keep, release)] = 1.0 / np.isin(keep, release).sum()
    out = {
        "lambda_1": lam1,
        "pct_lost_per_step": 100 * loss,
        "pct_lost_per_step_to_exit": 100 * to_exit,
        "pct_lost_per_step_to_cells_without_drifters": 100 * (loss - to_exit),
        "years_in_box_expected": float(years(1 / loss)),
        "years_half_life": float(years(np.log(2) / -np.log(lam1))),
        "lambda_2_modulus": lam2,
        "years_mixing_efold": float(years(1 / np.log(lam1 / lam2))) if lam2 > 0 else 0.0,
        "years_tmix_from_coast": float(years(steps_to_mix(Q, q, x0))),
        "qsd_centre_lat": lat,
        "qsd_centre_lon": lon,
        "pct_area_holding_half": 100 * area_holding_half(q, area[keep]),
    }
    if full:
        tmix = years(first_below(tv_to_qsd(Q, q, np.eye(keep.size))))
        out["years_tmix_median_start"] = float(np.nanmedian(tmix))
        out["years_tmix_worst_start"] = float(np.nanmax(tmix)) if np.isfinite(tmix).all() else float("nan")
        out["starts_not_mixed"] = int(np.isnan(tmix).sum())
    return out, q


def bootstrap(start, end, drifter_id, n_states, fit, keep, area, release, region, cell_size, q_hat,
              n_boot=N_BOOT):
    """Replicates of every summarise() quantity and of the QSD density, P refitted per replicate.
    q_hat (the full-data QSD) is only the power iteration's starting guess."""
    values, densities = {}, np.empty((n_boot, keep.size))
    for b, P_b in enumerate(replicate_matrices(start, end, drifter_id, n_states, fit, n_boot)):
        out, q = summarise(P_b, keep, area, release, region, cell_size, full=False, q_start=q_hat)
        for k, v in out.items():
            values.setdefault(k, np.empty(n_boot))[b] = v
        densities[b] = density_ratio(q, area[keep])
    return values, densities


def density_ratio(q, area):
    """QSD per unit area, as a multiple of the average over the kept cells (1 = average)."""
    density = q / area
    return density / (q.sum() / area.sum())


TRAP_TOL = 1e-9  # lambda_1 this close to 1: some cells keep all their material (a closed set, a "trap")

UNITS = {"lambda_1": "per 15-day step", "lambda_2_modulus": "per 15-day step", "qsd_centre_lat": "deg",
         "qsd_centre_lon": "deg", "starts_not_mixed": "cells"}


def plot(region, panels, curves, out_path):
    """(a, b) QSD density on the 2 deg product and 4 deg baseline; (c) its 95% interval width; (d) mixing curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), layout="constrained")
    norm = colors.LogNorm(vmin=0.1, vmax=10)
    for ax, (title, cell_size, keep, dens, _) in zip((axes[0, 0], axes[0, 1]), panels):
        lat_edges, lon_edges = grid_edges(region, cell_size)
        grid = np.full((len(lat_edges) - 1) * (len(lon_edges) - 1), np.nan)
        grid[keep] = dens
        mesh = ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_invalid(grid.reshape(len(lat_edges) - 1, -1)),
                             cmap="viridis", norm=norm)
        _frame(ax, region, f"{title}: where material still in the box gathers")
    fig.colorbar(mesh, ax=axes[0, 1], shrink=0.85, extend="both",
                 label="long-run density, times the box average (log scale)")

    title, cell_size, keep, dens, (low, high) = panels[0]
    ax = axes[1, 0]
    lat_edges, lon_edges = grid_edges(region, cell_size)
    grid = np.full((len(lat_edges) - 1) * (len(lon_edges) - 1), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        grid[keep] = (high - low) / dens
    mesh = ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_invalid(grid.reshape(len(lat_edges) - 1, -1)),
                         cmap="magma_r", vmin=0, vmax=1.5)
    _frame(ax, region, f"{title}: width of the 95% interval, relative to the estimate")
    fig.colorbar(mesh, ax=ax, shrink=0.85, extend="max", label="(high - low) / estimate")

    ax = axes[1, 1]
    t = years(np.arange(1, MAX_STEPS + 1))
    for label, tv, style in curves:
        ax.plot(t, tv, label=label, **style)
    ax.axhline(TMIX_EPS, color="grey", lw=1, ls=":")
    ax.text(11.8, TMIX_EPS + 0.01, f"t_mix cut-off ({TMIX_EPS})", ha="right", va="bottom", fontsize=8, color="grey")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1)
    ax.set_xlabel("years after release")
    ax.set_ylabel("total variation to the long-run pattern")
    ax.set_title(f"(d) How fast the {panels[0][0]} forgets where material started", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    for ax, letter in zip((axes[0, 0], axes[0, 1], axes[1, 0]), "abc"):
        ax.set_title(f"({letter}) " + ax.get_title(loc="left"), fontsize=10, loc="left")
    fig.text(0.01, -0.05,
             f"Quasi-stationary distribution of the {TAU_DAYS}-day matrix restricted to cells with drifters, {region['label']}: "
             "the pattern that material still in the box settles into, whatever its release point. Divided by cell area.\n"
             f"(c): from {N_BOOT} drifter bootstrap replicates. (d): conditional distribution of material still in the box, "
             "from the Benguela coastal cells and from every start cell (median, and the slowest 10%).", fontsize=8.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _frame(ax, region, title):
    draw_land(ax)
    ax.set_xlim(region["lon_min"], region["lon_max"])
    ax.set_ylim(region["lat_min"], region["lat_max"])
    ax.set_aspect("equal")
    ax.set_facecolor("#d9d9d9")
    ax.xaxis.set_major_formatter(_degrees("lon"))
    ax.yaxis.set_major_formatter(_degrees("lat"))
    ax.set_title(title, fontsize=10, loc="left")


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    table = load_transitions(region, verbose=False)
    product = np.load(os.path.join(PROCESSED_DIR, f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d.npz"))
    alpha = float(product["alpha"])
    paths = land_paths()

    models = [  # (name, stem for the saved file or None, grid, fit, bootstrap?)
        (f"{fine:g} deg shrinkage", f"shrinkage_P_{fine:g}deg", fine, shrinkage_fit(parent_states(region, fine, coarse), alpha), True),
        (f"{coarse:g} deg naive", f"naive_P_{coarse:g}deg", coarse, naive_P, True),
        (f"{fine:g} deg naive", None, fine, naive_P, False),
    ]
    rows, panels, curves = [], [], []
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for name, stem, size, fit, boot in models:
        start, end, n_cells = to_states(table, region, size)
        _, drifters = per_cell_counts(start, table["drifter_id"], n_cells)
        P = fit(count_matrix(start, end, n_cells + 1))
        if stem and stem.startswith("shrinkage") and not np.allclose(P, product["P"]):
            raise ValueError("Refitted shrinkage P differs from the saved product; rerun src.inference.shrinkage_fit.")
        keep = np.flatnonzero(drifters > 0)
        area = cell_areas_km2(region, size)
        zone = coastal_cells(region, size, land_fraction(region, size, paths))
        release = zone[drifters[zone] > 0]
        est, q = summarise(P, keep, area, release, region, size)
        lam_p, q_p = qsd_power(substochastic(P, keep))
        if abs(lam_p - est["lambda_1"]) > 1e-9 or np.abs(q_p - q).sum() > 1e-6:
            raise ValueError(f"{name}: power iteration and eigen-decomposition disagree on the QSD.")
        dens = density_ratio(q, area[keep])
        print(f"{name}: {keep.size} cells kept; " + ", ".join(f"{k} {v:.4g}" for k, v in est.items()))

        reps, dens_low, dens_high = {}, np.full(keep.size, np.nan), np.full(keep.size, np.nan)
        if boot:
            reps, dens_reps = bootstrap(start, end, table["drifter_id"], n_cells + 1, fit, keep, area, release,
                                        region, size, q)
            dens_low, dens_high = percentile_interval(dens_reps, LEVEL)
        # A replicate can draw only a thin cell's drifters that never left it; under naive P that cell then
        # keeps everything that reaches it (lambda_1 = 1) and the long-run pattern collapses into it.
        trapped = reps["lambda_1"] >= 1 - TRAP_TOL if reps else np.zeros(0, dtype=bool)
        if trapped.any():
            print(f"  {100 * trapped.mean():.1f}% of replicates have a trap (lambda_1 = 1)")
        sets = [(name, np.ones(trapped.size, dtype=bool))]
        if trapped.any():
            sets.append((f"{name} (replicates without a trap)", ~trapped))
        for label, use in sets:
            for k, v in est.items():
                if label != name and k not in reps:
                    continue  # point-only quantities are already in the first set
                low, high = percentile_interval(reps[k][use], LEVEL) if k in reps else ("", "")
                rows.append({"model": label, "quantity": k, "estimate": round(v, 6),
                             "ci_low": round(float(low), 6) if low != "" else "",
                             "ci_high": round(float(high), 6) if high != "" else "",
                             "unit": UNITS.get(k, "years" if k.startswith("years") else "%" if k.startswith("pct") else ""),
                             "cells_kept": keep.size, "n_boot": int(use.sum()) if k in reps else 0})
        if reps:
            rows.append({"model": name, "quantity": "pct_replicates_with_trap", "estimate": round(100 * trapped.mean(), 2),
                         "ci_low": "", "ci_high": "", "unit": "%", "cells_kept": keep.size, "n_boot": N_BOOT})
        if stem:
            path = os.path.join(PROCESSED_DIR, f"long_run_{stem}_tau{TAU_DAYS}d.npz")
            save_grid_result(path, region, size, TAU_DAYS, kept_cells=keep, qsd=q, density_ratio=dens,
                             density_ci_low=dens_low, density_ci_high=dens_high, lambda_1=est["lambda_1"],
                             lambda_2_modulus=est["lambda_2_modulus"], n_boot=N_BOOT, level=LEVEL)
            print(f"  saved {path}")
            panels.append((name.replace(" deg", "°"), size, keep, dens, (dens_low, dens_high)))
        if size == fine and boot:
            Q = substochastic(P, keep)
            x0 = np.zeros(keep.size)
            x0[np.isin(keep, release)] = 1.0 / release.size
            every = tv_to_qsd(Q, q, np.eye(keep.size))
            curves = [("from the Benguela coastal cells", tv_to_qsd(Q, q, x0), {"color": "#d62728", "lw": 2}),
                      ("every start cell: median", np.median(every, axis=0), {"color": "#1f77b4", "lw": 1.5}),
                      ("every start cell: slowest 10%", np.quantile(every, 0.9, axis=0),
                       {"color": "#1f77b4", "lw": 1.5, "ls": "--"})]

    for r in rows:
        print(r)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table_out = os.path.join(RESULTS_DIR, f"long_run_tau{TAU_DAYS}d.csv")
    with open(table_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_out = os.path.join(FIGURES_DIR, "long_run_qsd.png")
    plot(region, panels, curves, fig_out)
    print(f"Saved {table_out} and {fig_out}")
