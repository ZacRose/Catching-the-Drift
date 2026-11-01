"""Week 7 Quantitative: Markov check, long-run pattern and mixing time, prior variants, product.
Hand-made matrices with answers known in closed form, so no data files are needed."""
import numpy as np
import pytest

from src.inference.estimators import prior_rows
from src.inference.markov_check import cell_distances, compare, group_means, row_summaries
from src.inference.transition_data import load_transitions
from src.linalg.long_run import (
    TRAP_TOL, area_holding_half, cell_areas_km2, first_below, leading_eigenvalues, qsd_power,
    quasi_stationary, steps_to_mix, tv_to_qsd,
)
from src.linalg.product import Product

REGION = {"lat_min": -4.0, "lat_max": 4.0, "lon_min": 0.0, "lon_max": 8.0, "cell_size_deg": 2.0}


def two_state(a, leak):
    """Symmetric two-cell chain that swaps with probability a and loses `leak` from each cell per step."""
    return (1 - leak) * np.array([[1 - a, a], [a, 1 - a]])


def test_qsd_is_the_leading_left_eigenvector():
    Q = np.array([[0.5, 0.4, 0.05], [0.3, 0.6, 0.05], [0.1, 0.2, 0.6]])
    lam1, q, lam2 = quasi_stationary(Q)
    assert np.allclose(q @ Q, lam1 * q)
    assert np.all(q >= 0) and np.isclose(q.sum(), 1)
    assert lam2 < lam1 < 1
    lam_p, q_p = qsd_power(Q)
    assert np.isclose(lam_p, lam1) and np.allclose(q_p, q)
    assert np.allclose(leading_eigenvalues(Q), (lam1, lam2))


def test_uniform_leak_gives_lambda_and_stationary_pattern():
    S = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    lam1, q, _ = quasi_stationary(0.99 * S)
    assert np.isclose(lam1, 0.99)  # 1% lost per step everywhere
    w, v = np.linalg.eig(S.T)
    pi = np.real(v[:, np.argmax(np.real(w))])
    assert np.allclose(q, pi / pi.sum())  # same as S's ordinary stationary distribution


def test_mixing_time_matches_closed_form():
    a, leak = 0.1, 0.02
    Q = two_state(a, leak)
    lam1, q, lam2 = quasi_stationary(Q)
    assert np.isclose(lam1, 1 - leak) and np.isclose(lam2, (1 - leak) * (1 - 2 * a))
    assert np.allclose(q, 0.5)
    tv = tv_to_qsd(Q, q, np.array([1.0, 0.0]), max_steps=40)
    n = np.arange(1, 41)
    assert np.allclose(tv, 0.5 * (1 - 2 * a) ** n)  # distance shrinks by lambda_2 / lambda_1 per step
    expected = int(np.ceil(np.log(0.5) / np.log(1 - 2 * a)))  # first n with 0.5 (0.8)^n <= 0.25
    assert first_below(tv) == expected == steps_to_mix(Q, q, np.array([1.0, 0.0]))


def test_trap_shows_as_lambda_one_and_never_mixes():
    Q = np.array([[1.0, 0.0], [0.3, 0.6]])  # cell 0 keeps everything: a closed set
    lam1, _ = leading_eigenvalues(Q)
    assert lam1 >= 1 - TRAP_TOL
    lam1, q, _ = quasi_stationary(Q)
    assert np.allclose(q, [1, 0])


def test_cell_areas_and_concentration():
    areas = cell_areas_km2(REGION, 2.0)
    r = 6371.0
    box = r ** 2 * np.deg2rad(8) * (np.sin(np.deg2rad(4)) - np.sin(np.deg2rad(-4)))
    assert np.isclose(areas.sum(), box)
    assert np.isclose(areas.reshape(4, 4)[0, 0], areas.reshape(4, 4)[-1, 0])  # symmetric about the equator
    q = np.array([0.5, 0.25, 0.25])
    assert np.isclose(area_holding_half(q, np.ones(3)), 1 / 3)  # the densest cell alone holds half


def test_markov_compare_is_zero_for_an_exact_chain():
    P = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.0, 0.0, 1.0]])
    dist = np.array([[0.0, 100.0], [100.0, 0.0]])
    chained, direct, tv = compare(P, P @ P, 2, dist)
    assert np.allclose(tv, 0)
    for q in chained:
        assert np.allclose(chained[q], direct[q])
    s = row_summaries(P, dist)
    assert np.allclose(s["stay"], [0.6, 0.7]) and np.allclose(s["exit"], [0.1, 0.1])
    assert np.allclose(s["km"], [100 * 0.3 / 0.9, 100 * 0.2 / 0.9])
    assert group_means(np.array([1.0, np.nan, 3.0]), {"g": np.array([True, True, True])})["g"] == 2.0


def test_cell_distances():
    d = cell_distances(REGION, 2.0)
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0)
    assert abs(d[0, 1] - 6371.0 * np.deg2rad(2) * np.cos(np.deg2rad(3))) < 1  # neighbours along 3S


def test_even_split_prior():
    parent = np.array([0, 0, 0, 0, 1])  # four fine cells in one coarse cell, then exit
    counts = np.zeros((5, 5))
    counts[0, 1], counts[2, 2], counts[3, 4] = 10, 5, 1
    for split in ("arrivals", "even"):
        assert np.allclose(prior_rows(counts, parent, split=split).sum(axis=1), 1)
    even = prior_rows(counts, parent, split="even")
    assert np.allclose(even[0, :4], even[0, 0])  # the coarse cell's mass shared equally
    arrivals = prior_rows(counts, parent)
    assert arrivals[0, 1] > arrivals[0, 0]  # cell 1 had arrivals
    with pytest.raises(ValueError):
        prior_rows(counts, parent, split="area")


def test_load_transitions_row_mask(tmp_path):
    path = tmp_path / "t.npz"
    n = 4
    np.savez(path, drifter_id=np.arange(n), start_time=np.arange(n) * 86400, start_lon=np.full(n, 1.0),
             start_lat=np.full(n, 0.0), end_lon=np.array([1.0, np.nan, 3.0, 5.0]), end_lat=np.zeros(n),
             drogued=np.ones(n), outcome=np.array([0, 1, 2, 0]))
    region = {**REGION, "lon_max": 8.0}
    full = load_transitions(region, str(path), verbose=False)
    assert full["drifter_id"].tolist() == [0, 1, 3]  # outcome 2 (record ended) dropped
    masked = load_transitions(region, str(path), verbose=False, rows=np.array([True, False, True, False]))
    assert masked["drifter_id"].tolist() == [0]
    with pytest.raises(ValueError):
        load_transitions(region, str(path), verbose=False, rows=np.ones(3, dtype=bool))


def test_product_forecast_and_cells():
    P = np.array([[0.5, 0.5, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.0]])
    region = {"lat_min": 0.0, "lat_max": 2.0, "lon_min": 0.0, "lon_max": 4.0}
    prod = Product(P=P, se=np.zeros_like(P), ci_low=P, ci_high=P, start_drifters=np.array([5, 0]), alpha=1.0,
                   region=region, cell_size=2.0, tau_days=15, lat_centre=np.array([1.0, 1.0]),
                   lon_centre=np.array([1.0, 3.0]))
    assert prod.cell_of(1.0, 3.5) == 1
    assert prod.prior_only.tolist() == [False, True]
    dist = prod.forecast(lat=1.0, lon=0.5, days=30)
    assert np.allclose(dist, [0.25, 0.70, 0.05])
    s = prod.summary(dist, top=1)
    assert np.isclose(s["pct_left_box"], 5) and s["top_cells"][0][0] == 1
    with pytest.raises(ValueError):
        prod.forecast(lat=1.0, lon=0.5, days=20)
    with pytest.raises(ValueError):
        prod.cell_of(5.0, 1.0)
