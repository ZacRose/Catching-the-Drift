"""Week 9 app: the data exporter and the interval check on hand-made data, and (if Chrome and the
exported data are present) the page's own maths against src/linalg/product.py and the shore model."""
import base64
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import numpy as np
import pytest

from src.app.check_intervals import brute_force, forward_quantities
from src.app.export_app_data import (APP_DATA_PATH, QUANTITIES, b64_float32, bootstrap_intervals, functionals,
                                     great_circle_km, groundings, quantities, saved_season_counts, shore_fit, start_matrix,
                                     step_back, zone_cells)
from src.common.config import PROCESSED_DIR, REPO_ROOT
from src.forecasting.seasonal_matrices import chain_season, season_matrices
from src.gridding.coast import RAN_AGROUND
from src.gridding.transitions import ENDED, IN_R, LEFT_R
from src.inference.drogue_comparison import UNDROGUED_STEM, undrogued_only
from src.inference.estimators import naive_P
from src.linalg.connectivity import absorbing

REGION = {"lat_min": -4.0, "lat_max": 4.0, "lon_min": 0.0, "lon_max": 8.0, "cell_size_deg": 2.0}  # 4 x 4 cells
CHROME = next((p for p in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                           shutil.which("google-chrome") or "", shutil.which("chromium") or "",
                           shutil.which("chrome") or "") if p and os.path.exists(p)), None)


def chain(n=16, n_shore=0, seed=0):
    """A random chain over n cells, an absorbing 'left' state and n_shore absorbing 'aground' states."""
    rng = np.random.default_rng(seed)
    k = n + 1 + n_shore
    P = rng.random((k, k))
    P[:n, n:] *= 0.05
    P /= P.sum(axis=1, keepdims=True)
    P[n:] = 0.0
    P[n:, n:] = np.eye(1 + n_shore)
    return P


def test_backward_functionals_match_forward_steps():
    n, n_shore = 16, 2
    P = chain(n, n_shore)
    lat_c, lon_c = np.linspace(-3, 3, n), np.linspace(1, 7, n)
    benguela, coast = np.array([0, 1, 5]), np.array([1, 2])
    F0 = functionals(lat_c, lon_c, benguela, coast, n_shore)
    S = start_matrix([3, -1], np.array([4, 9]), P.shape[0])
    assert np.allclose(S[1, [4, 9]], 0.5) and np.isclose(S[1].sum(), 1) and S[0, 3] == 1
    Q = absorbing(P, coast)
    for k, F in enumerate(step_back(P, F0, 5), start=1):
        x = S @ np.linalg.matrix_power(P, k)  # forward, one row per start
        h = S @ np.linalg.matrix_power(Q, k)
        q = quantities(S @ F, np.zeros(2), np.zeros(2))
        assert np.allclose(q, forward_quantities(x, h, n, lat_c, lon_c, benguela, coast, np.zeros(2), np.zeros(2)))
        inside = x[:, :n].sum(axis=1)
        assert np.allclose(q[0], 100 * x[:, n]) and np.allclose(q[6], 100 * x[:, n + 1:].sum(axis=1))
        assert np.allclose(q[3], (x[:, :n] * lat_c).sum(axis=1) / inside)


def test_shore_fit_scales_the_water_rows():
    n_cells = 3
    counts = np.zeros((6, 6))  # 3 cells, left, 2 coasts
    counts[0, [0, 1]] = [6, 2]   # cell 0: 8 known moves...
    counts[0, 4] = 2             # ...and 2 groundings on coast 0
    counts[1, 1] = 5             # cell 1: no groundings
    counts[2, [2, 5]] = [1, 3]   # cell 2: 1 move, 3 groundings on coast 1
    parent = np.array([0, 0, 1, 2])  # cells 0 and 1 share a coarse cell; cell 2 is alone
    P = shore_fit(naive_P, n_cells, 2, parent, alpha=0)(counts)  # no shrinkage: raw shares
    assert np.allclose(P.sum(axis=1), 1)
    assert np.allclose(P[0, :4], [0.6, 0.2, 0, 0]) and np.isclose(P[0, 4], 0.2)
    assert np.allclose(P[1, :4], [0, 1, 0, 0]) and P[1, 4:].sum() == 0
    assert np.isclose(P[2, 2], 0.25) and np.isclose(P[2, 5], 0.75)
    assert np.allclose(P[4:, 4:], np.eye(2)) and P[3, 3] == 1
    P = shore_fit(naive_P, n_cells, 2, parent, alpha=10)(counts)
    # cell 1 has no groundings but borrows its parent's: 2 of the parent's 15 moves ran aground on coast 0
    assert np.isclose(P[1, 4], (0 + 10 * 2 / 15) / (5 + 10)) and np.isclose(P[1, :4].sum(), 1 - P[1, 4])
    # cell 2's parent is cell 2 itself, so shrinking changes nothing there
    assert np.isclose(P[2, 5], 0.75) and np.allclose(P.sum(axis=1), 1)


def test_groundings_go_to_the_nearest_coastal_cell():
    land = np.zeros(16)
    land[[3, 7]] = [0.5, 1.0]  # cell 3 (row 0, col 3) is coastal; cell 7 is all land
    stored = {  # drifter 1 runs aground next to cell 3; drifter 2 "runs aground" far from land; drifter 3 doesn't
        "drifter_id": np.array([1, 1, 1, 2, 2, 3]),
        "start_time": np.array([0, 1, 2, 0, 1, 0]) * np.timedelta64(1, "D") + np.datetime64("2000-01-01"),
        "start_lon": np.array([1.0, 3.0, 5.0, 1.0, 1.0, 1.0]), "start_lat": np.array([-3.0, -3.0, -3.0, 3.0, 3.0, -3.0]),
        "outcome": np.array([IN_R, ENDED, ENDED, IN_R, ENDED, ENDED]),
        "typedeath": np.array([RAN_AGROUND] * 5 + [3]),
    }
    g = groundings(stored, REGION, 2.0, land)
    assert g["n_kept"] == 1 and g["n_dropped"] == 1
    assert g["drifter_id"].tolist() == [1, 1] and g["shore"].tolist() == [3, 3]  # both of drifter 1's final starts
    assert g["start_lon"].tolist() == [3.0, 5.0]


def test_fast_intervals_match_brute_force():
    """The exporter's route (drifter_pairs, backward steps, percentile_interval) against the slow one
    (row weights, forward steps, np.quantile), on the same draws."""
    rng = np.random.default_rng(1)
    n_cells, n_shore = 16, 2
    n_states = n_cells + 1 + n_shore
    start = rng.integers(0, n_cells, 600)
    end = np.where(rng.random(600) < 0.9, rng.integers(0, n_cells + 1, 600), n_cells + 1 + rng.integers(0, n_shore, 600))
    drifter_id = rng.integers(0, 40, 600) * 7 + 3
    def water(c):  # naive P with a little added everywhere, so no row is empty; "left" stays absorbing
        c = c + 0.01
        c[-1] = 0
        c[-1, -1] = 1
        return naive_P(c)
    model = {"start": start, "end": end, "drifter_id": drifter_id, "n_states": n_states,
             "fit": shore_fit(water, n_cells, n_shore, np.arange(n_cells + 1) // 4, alpha=3.0)}
    lat_c, lon_c = np.linspace(-3, 3, n_cells), np.linspace(1, 7, n_cells)
    benguela, coast = np.array([0, 1, 5]), np.array([1, 2])
    starts = [2, 7, -1]
    S = start_matrix(starts, np.array([4, 9]), n_states)
    s_lat, s_lon = np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])
    F0 = functionals(lat_c, lon_c, benguela, coast, n_shore)
    low, high = bootstrap_intervals(start, end, drifter_id, n_states, model["fit"], S, F0, s_lat, s_lon,
                                    n_steps=6, n_boot=40, seed=7)
    reps = brute_force(model, S, n_cells, lat_c, lon_c, benguela, coast, s_lat, s_lon, steps=[1, 3, 6], n_boot=40, seed=7)
    lo, hi = np.quantile(reps, [0.025, 0.975], axis=0)
    for s, n in enumerate([1, 3, 6]):
        assert np.allclose(low[:, n - 1], lo[:, s], rtol=1e-4, atol=1e-4)
        assert np.allclose(high[:, n - 1], hi[:, s], rtol=1e-4, atol=1e-4)
    assert len(QUANTITIES) == low.shape[0] == 8


def test_great_circle_and_zone_cells():
    assert np.isclose(great_circle_km(0, 0, 0, 1), 111.19, atol=0.01)  # one degree of the equator
    assert np.isclose(great_circle_km(-30, 10, -30, 10), 0)
    # cells wholly inside 0-4N, 2-6E on the 4 x 4 grid: rows 2-3, columns 1-2
    assert zone_cells(REGION, 2.0, {"lat_min": 0, "lat_max": 4, "lon_min": 2, "lon_max": 6}).tolist() == [9, 10, 13, 14]


def test_float32_round_trip():
    a = np.array([[0.25, 1e-7], [3.5, -2.0]])
    back = np.frombuffer(base64.b64decode(b64_float32(a)), dtype="<f4")
    assert np.allclose(back, a.ravel(), rtol=1e-6)


def test_page_is_one_view_with_its_info_button():
    with open(os.path.join(REPO_ROOT, "app", "index.html")) as f:
        page = f.read()
    assert 'id="infoBtn"' in page and 'id="infoDialog"' in page
    assert 'role="tablist"' not in page and "data-view" not in page



@pytest.mark.skipif(CHROME is None or not os.path.exists(APP_DATA_PATH),
                    reason="needs Google Chrome and app/data/app_data.js (python3 -m src.app.export_app_data)")
def test_page_maths_matches_python():
    from src.app.export_app_data import build_models
    from src.common.config import load_region_bounds
    from src.gridding.coast import coastal_cells, land_fraction, land_paths
    from src.inference.transition_data import TRANSITIONS_PATH, load_transitions
    from src.linalg.product import load_product

    page = os.path.join(REPO_ROOT, "tests", "app_check.html")
    with tempfile.TemporaryDirectory() as profile:
        dom = subprocess.run([CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
                              "--virtual-time-budget=30000", "--dump-dom", f"file://{page}"],
                             capture_output=True, text=True, timeout=180).stdout
    found = re.search(r'<pre id="out">(.*?)</pre>', dom, re.S)
    assert found, "the check page printed nothing"
    out = json.loads(html.unescape(found.group(1)))
    assert out["ok"], out.get("error")

    region = load_region_bounds()
    prods = {"all": load_product(), "undrogued": load_product(drifters="undrogued")}
    land = land_fraction(region, prods["all"].cell_size, land_paths())
    coast = coastal_cells(region, prods["all"].cell_size, land)
    assert out["cell_of"] == prods["all"].cell_of(-28.0, 14.0)
    base = load_transitions(region, verbose=False)
    with np.load(TRANSITIONS_PATH) as z:
        stored = {k: z[k] for k in z.files}
    unt = ~stored["drogued"].astype(bool)
    built = {"all": build_models(region, base, stored, prods["all"].alpha, land)}
    shores = built["all"][0]["shore"]["shores"]
    built["undrogued"] = build_models(region, undrogued_only(base), {k: (v[unt] if v.shape == unt.shape else v) for k, v in stored.items()},
                                      prods["undrogued"].alpha, land, shores)
    for run in out["results"]:
        key, variant, season = run["set"], run["variant"], run["season"]
        models, n_cells, drifters, _ = built[key]
        P = models[variant]["P"]
        assert run["n_states"] == P.shape[0]
        release = coast[drifters[coast] > 0]
        prod = prods[key]
        if season is None:
            step_matrix = lambda step: P
        else:
            counts, beta = saved_season_counts(key, n_cells + 1)
            seasons = season_matrices(P, counts, beta, n_cells + 1)
            step_matrix = lambda step: seasons[chain_season(season, step)]
        for r in run["starts"]:
            x0 = np.zeros(P.shape[0])
            if r["start"]["kind"] == "coast":
                x0[release] = 1 / release.size
            else:
                x0[r["start"]["cell"]] = 1
            x, h = x0.copy(), x0.copy()
            done = 0
            for s in r["steps"]:
                for step in range(done, s["n"]):
                    x, h = x @ step_matrix(step), h @ absorbing(step_matrix(step), coast)
                done = s["n"]
                assert abs(s["mass"] - 1) < 1e-9
                water = x[:n_cells]
                assert np.isclose(s["stats"]["pct_left_box"], 100 * x[n_cells], rtol=1e-4, atol=1e-6)
                assert np.isclose(s["stats"]["pct_aground"], 100 * x[n_cells + 1:].sum(), rtol=1e-4, atol=1e-6)
                assert np.isclose(s["stats"]["centre_lat"], water @ prod.lat_centre / water.sum(), atol=1e-4)
                assert np.isclose(s["stats"]["centre_lon"], water @ prod.lon_centre / water.sum(), atol=1e-4)
                assert np.isclose(s["reached"], 100 * h[coast].sum(), rtol=1e-4, atol=1e-6)
                for cell, p in s["top"]:
                    assert np.isclose(p, x[cell], atol=1e-6)
                if variant == "water":  # the water model is the product itself, with its seasons
                    code = None if season is None else "DJF MAM JJA SON".split()[season]
                    assert np.allclose(x, prod.forecast(days=s["n"] * prod.tau_days, start=x0, season=code), atol=1e-6)
                assert s["areas"][0]["cells"] <= s["areas"][1]["cells"]
                assert np.isclose(sum(s["shares"]), 100 * water.sum(), rtol=1e-6)
                if s["interval"] is not None:
                    assert s["interval"][0] <= s["interval"][1]
                elif r["start"]["kind"] != "coast":
                    assert drifters[r["start"]["cell"]] == 0  # only a start with no drifters in this set has no interval
