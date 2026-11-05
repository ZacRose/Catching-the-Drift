"""
seasonal_robustness.py

Related task: Week 9 forensic review of RQ4. seasonal_forecast.py finds that offshore drift from the Benguela
zone is seasonal. Week 8 found drogue status matters more than any other choice in the project, but the seasonal
result had never been tested against it, or against the other choices it rests on. Here each choice is changed
in turn, and the summer-winter (DJF - JJA) and spring-winter (SON - JJA) gaps recomputed. Nothing in
seasonal_forecast.py changes, and its outputs aren't touched.

1. The seasonal signal:
   - drifter type: drogued only, undrogued only, and the seasons reweighted to the same drogue mix;
   - independent of the 15-day moves: the reported hourly velocity, added up over 15 days (needs the region cache);
   - outliers: dropping the fastest moves, and medians;
   - the box: nearer and further from the coast, north and south halves;
   - direction: the true offshore axis, not due west (the coast runs about COAST_ANGLE west of north);
   - the time step: 10 and 30-day moves, rescaled to 15 days;
   - any one year: every year left out in turn;
   - chance: the season labels shuffled within each year. If this can make a pattern as big, the seasons mean nothing.
   Intervals resample whole season-years, as season_differences().
2. The forecast (the seasonal mean against the annual mean): the settings MIN_DRIFTERS, FIRST_TEST_YEAR, the
   drifter type, RMSE instead of MAE, and shuffled season labels.
3. The release (Part D) by drifter type: does DJF still leave less in the coastal zone?

It also writes data/processed/seasonal_forecast_app.json, which src/app/export_app_data.py puts in the app's
Seasons view. The app shows the seasonal pattern, not a forecast: there is no forecast to show (the data
stop in October 2022, and the best method is the average of past years' same season).

Outputs: outputs/results/seasonal_robustness.csv and data/processed/seasonal_forecast_app.json.
Run: python3 -m src.forecasting.seasonal_robustness   (about 2 minutes; after src.forecasting.seasonal_forecast)
"""
import csv
import json
import os
import numpy as np

from src.common.config import INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.forecasting import seasonal_forecast as sf
from src.ingest.load import load_region

COAST_ANGLE = 19.0  # deg west of north: Cape Point to the Kunene mouth, about 6 deg of longitude over 17 of latitude
N_SHUFFLE = 5000
N_SHUFFLE_FORECAST = 2000
N_BOOT_RELEASE = 300
SECONDS_PER_MOVE = 15 * 86400.0
APP_JSON_PATH = os.path.join(PROCESSED_DIR, "seasonal_forecast_app.json")
SEASON_NAMES = ("Summer", "Autumn", "Winter", "Spring")  # DJF, MAM, JJA, SON (southern hemisphere)
DJF, MAM, JJA, SON = 0, 1, 2, 3


def load_with_drogue(region):
    """seasonal_forecast's table, plus the drogue status at the start of each move."""
    t = sf.load_table(region=region)
    z = np.load(sf.TABLE_PATH)
    t["drogued"] = z["drogued"][z["outcome"] <= sf.LEFT_R].astype(bool)
    return t


def subset_table(t, mask):
    """The rows of a loaded table where mask is True, with drifters renumbered."""
    out = {k: (v[mask] if isinstance(v, np.ndarray) and v.shape == mask.shape else v) for k, v in t.items()}
    out["rank"] = np.unique(out["drifter_id"], return_inverse=True)[1]
    out["n_drifters"] = int(out["rank"].max()) + 1
    return out


def season_year_sums(values, syear, season):
    """Per (season-year, season): sum of values and number of moves, and the years in order."""
    years = np.unique(syear)
    at = (np.searchsorted(years, syear), season)
    sums, counts = np.zeros((years.size, 4)), np.zeros((years.size, 4))
    np.add.at(sums, at, values)
    np.add.at(counts, at, 1)
    return sums, counts


def years_higher(series, a, b):
    """(years season a beats season b, years both have enough drifters to compare)."""
    both = [y for y, s in series if s == a and (y, b) in series]
    return sum(series[(y, a)] > series[(y, b)] for y in both), len(both)


def signal_row(test, setting, values, mask, t, n_boot, rng, min_drifters=sf.MIN_DRIFTERS):
    """One row: each season's pooled mean, and the two gaps with intervals resampling whole season-years."""
    sums, counts = season_year_sums(values[mask], t["syear"][mask], t["season"][mask])
    pooled = sums.sum(axis=0) / counts.sum(axis=0)
    draws = rng.integers(0, sums.shape[0], (n_boot, sums.shape[0]))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    q = [(100 - sf.LEVEL) / 2, 100 - (100 - sf.LEVEL) / 2]
    low1, high1 = np.nanpercentile(boot[:, DJF] - boot[:, JJA], q)
    low2, high2 = np.nanpercentile(boot[:, SON] - boot[:, JJA], q)
    series, _ = sf.season_year_series(values, mask, t, sf.group_rows(t), min_drifters=min_drifters)
    w1, n1 = years_higher(series, DJF, JJA)
    w2, n2 = years_higher(series, SON, JJA)
    return {"test": test, "setting": setting, "moves": int(mask.sum()), "drifters": int(np.unique(t["rank"][mask]).size),
            "DJF": round(float(pooled[DJF]), 2), "MAM": round(float(pooled[MAM]), 2), "JJA": round(float(pooled[JJA]), 2), "SON": round(float(pooled[SON]), 2),
            "DJF_low": round(float(np.nanpercentile(boot[:, DJF], q[0])), 2), "DJF_high": round(float(np.nanpercentile(boot[:, DJF], q[1])), 2),
            "MAM_low": round(float(np.nanpercentile(boot[:, MAM], q[0])), 2), "MAM_high": round(float(np.nanpercentile(boot[:, MAM], q[1])), 2),
            "JJA_low": round(float(np.nanpercentile(boot[:, JJA], q[0])), 2), "JJA_high": round(float(np.nanpercentile(boot[:, JJA], q[1])), 2),
            "SON_low": round(float(np.nanpercentile(boot[:, SON], q[0])), 2), "SON_high": round(float(np.nanpercentile(boot[:, SON], q[1])), 2),
            "djf_minus_jja": round(float(pooled[DJF] - pooled[JJA]), 2), "djf_jja_low": round(float(low1), 2), "djf_jja_high": round(float(high1), 2),
            "years_djf_higher": w1, "years_djf_compared": n1,
            "son_minus_jja": round(float(pooled[SON] - pooled[JJA]), 2), "son_jja_low": round(float(low2), 2), "son_jja_high": round(float(high2), 2),
            "years_son_higher": w2, "years_son_compared": n2}


def fixed_mix_gaps(values, mask, drogued, season):
    """Each season's mean with drogued and undrogued moves weighted as in all seasons together."""
    share = drogued[mask].mean()
    return np.array([share * values[mask & drogued & (season == s)].mean()
                     + (1 - share) * values[mask & ~drogued & (season == s)].mean() for s in range(4)])


def shuffled_gap(values, mask, t, n, rng):
    """(biggest gap between seasons, the same with season labels shuffled within each year, n times)."""
    sums, counts = season_year_sums(values[mask], t["syear"][mask], t["season"][mask])
    pooled = sums.sum(axis=0) / counts.sum(axis=0)
    rows = np.arange(sums.shape[0])[:, None]
    labels = np.tile(np.arange(4), (sums.shape[0], 1))
    gaps = np.empty(n)
    for k in range(n):
        order = rng.permuted(labels, axis=1)
        means = sums[rows, order].sum(axis=0) / counts[rows, order].sum(axis=0)
        gaps[k] = means.max() - means.min()
    return float(pooled.max() - pooled.min()), gaps


def forecast_row(setting, values, mask, t, groups, n_boot, rng, min_drifters=sf.MIN_DRIFTERS, first_test_year=sf.FIRST_TEST_YEAR):
    """The seasonal mean against the annual mean under one setting."""
    series, moves = sf.season_year_series(values, mask, t, groups, min_drifters=min_drifters)
    rows = sf.rolling_forecasts(series, first_test_year=first_test_year, moves=moves)
    if len(rows) < 8:
        return {"test": "forecast", "setting": setting, "n_forecasts": len(rows), "note": "too few forecasts to score"}
    skill = {r["method"]: r for r in sf.skill_with_ci(rows, n_boot, rng)}["Seasonal mean"]
    truth = np.array([r["truth"] for r in rows])
    bench, seasonal = np.array([r[sf.BENCH] for r in rows]), np.array([r["Seasonal mean"] for r in rows])
    rmse = 1 - np.sqrt(((seasonal - truth) ** 2).mean()) / np.sqrt(((bench - truth) ** 2).mean())
    return {"test": "forecast", "setting": setting, "n_forecasts": len(rows), "mae_annual_km": round(float(np.abs(bench - truth).mean()), 2),
            "mae_seasonal_km": round(float(np.abs(seasonal - truth).mean()), 2), "skill_pct": round(100 * float(skill["skill"]), 1),
            "skill_low": round(100 * float(skill["ci_low"]), 1), "skill_high": round(100 * float(skill["ci_high"]), 1),
            "rmse_skill_pct": round(100 * float(rmse), 1)}


def shuffled_skill(series, moves, n, rng):
    """(the seasonal mean's MAE skill, the same with season labels shuffled within each year, n times)."""
    def skill(s, m):
        rows = sf.rolling_forecasts(s, moves=m)
        truth = np.array([r["truth"] for r in rows])
        return 1 - np.abs(np.array([r["Seasonal mean"] for r in rows]) - truth).mean() / np.abs(np.array([r[sf.BENCH] for r in rows]) - truth).mean()
    years = sorted({k[0] for k in series})
    out = []
    for _ in range(n):
        order = {y: rng.permutation(4) for y in years}
        try:
            out.append(skill({(y, int(order[y][q])): v for (y, q), v in series.items()},
                             {(y, int(order[y][q])): v for (y, q), v in moves.items()}))
        except (ValueError, ZeroDivisionError):  # a shuffle can leave a season with too few earlier years to forecast
            continue
    return float(skill(series, moves)), np.array(out)


def app_block(signal_all, signal_undrogued, signal_drogued, forecast):
    """The app's Seasons view (the format in src/app/export_app_data.py's docstring)."""
    x = [f"{name} ({code})" for name, code in zip(SEASON_NAMES, sf.SEASONS)]
    line = lambda r: [r[c] for c in sf.SEASONS]
    high = round(0.5 * (signal_all["DJF"] + signal_all["SON"]), -1)
    low = round(signal_all["JJA"], -1)
    return {
        "title": "Does the drift change with the seasons?",
        "summary": (f"Yes, in one clear way. Material starting in the Benguela zone drifts further west in summer and spring (about {high:.0f} km per 15 days) "
                    f"than in winter (about {low:.0f} km). Summer beat winter in {signal_all['years_djf_higher']} of the {signal_all['years_djf_compared']} years with enough "
                    "drifters to compare."),
        "quantity": "Westward distance per 15 days from the Benguela zone, outlined on the map",
        "unit": "km",
        "series": [
            {"label": "All drifters (shaded: 95% range, resampling whole years)", "x": x, "y": line(signal_all),
             "low": [signal_all[f"{c}_low"] for c in sf.SEASONS], "high": [signal_all[f"{c}_high"] for c in sf.SEASONS], "kind": "observed"},
            {"label": "Lost their drogue (also pushed by the wind)", "x": x, "y": line(signal_undrogued), "kind": "observed"},
            {"label": "Drogue attached (follow the water)", "x": x, "y": line(signal_drogued), "kind": "observed", "dashed": True},
        ],
        "notes": [
            f"The pattern is mainly in drifters that have lost their drogue, which the wind pushes directly: {signal_undrogued['djf_minus_jja']:+.0f} km between summer and winter, "
            f"against {signal_drogued['djf_minus_jja']:+.0f} km for drifters that follow the water, whose range ({signal_drogued['djf_jja_low']:+.0f} to {signal_drogued['djf_jja_high']:+.0f} km) includes no difference.",
            f"This is a typical pattern, not a forecast for a particular date. Knowing the season cuts the typical error in a season's westward drift from {forecast['mae_annual_km']:.0f} km to "
            f"{forecast['mae_seasonal_km']:.0f} km, so one year can still differ a lot from the average.",
            "On the drift map you can release in any season, and the drift then changes as the seasons pass. Whole matrices need more data than one average, so the map's seasons are pulled towards the all-year drift, and show smaller gaps than this chart.",
        ],
    }


if __name__ == "__main__":
    region = load_region_bounds()
    t = load_with_drogue(region)
    zone = region["coastal_zone"]
    rng = np.random.default_rng(sf.SEED)
    n_boot = sf.N_BOOT
    stay = ~t["exited"]
    band = t["in_band"] & stay
    offshore = -t["east_km"]
    rows = []

    def signal(test, setting, values, mask, table=t, **kw):
        r = signal_row(test, setting, values, mask, table, n_boot, rng, **kw)
        rows.append(r)
        print(f"{test:12s} {setting:44s} DJF-JJA {r['djf_minus_jja']:+7.1f} [{r['djf_jja_low']:+.0f},{r['djf_jja_high']:+.0f}] {r['years_djf_higher']}/{r['years_djf_compared']}"
              f" | SON-JJA {r['son_minus_jja']:+7.1f} [{r['son_jja_low']:+.0f},{r['son_jja_high']:+.0f}] {r['years_son_higher']}/{r['years_son_compared']}")
        return r

    base = signal("baseline", "westward km, all drifters", offshore, band)
    north = signal("baseline", "northward km, all drifters", t["north_km"], band)

    # Drifter type. Year counts at 10+ drifters, as a thinner table has few years at 15+
    tables = {"drogued only": subset_table(t, t["drogued"]), "undrogued only": subset_table(t, ~t["drogued"])}
    by_type = {}
    for name, tt in tables.items():
        by_type[name] = signal("drifter type", f"westward km, {name} (years at 10+ drifters)", -tt["east_km"], tt["in_band"] & ~tt["exited"], tt, min_drifters=10)
        signal("drifter type", f"northward km, {name} (years at 10+ drifters)", tt["north_km"], tt["in_band"] & ~tt["exited"], tt, min_drifters=10)
    for name, values in (("westward", offshore), ("northward", t["north_km"])):
        mix = fixed_mix_gaps(values, band, t["drogued"], t["season"])
        rows.append({"test": "drifter type", "setting": f"{name} km, seasons reweighted to the same drogue mix", "moves": int(band.sum()),
                     "DJF": round(float(mix[DJF]), 2), "MAM": round(float(mix[MAM]), 2), "JJA": round(float(mix[JJA]), 2), "SON": round(float(mix[SON]), 2),
                     "djf_minus_jja": round(float(mix[DJF] - mix[JJA]), 2), "son_minus_jja": round(float(mix[SON] - mix[JJA]), 2)})
    for s, label in enumerate(sf.SEASONS):
        rows.append({"test": "drifter type", "setting": f"share of coastal-box moves drogued, {label}", "moves": int((band & (t["season"] == s)).sum()),
                     "note": f"{100 * t['drogued'][band & (t['season'] == s)].mean():.0f}%"})

    # The velocities, which don't use the 15-day positions at all
    if os.path.exists(os.path.join(INTERIM_DIR, f"{region['name']}.npz")):
        data = load_region(region)
        inz = ((data["lat"] >= zone["lat_min"]) & (data["lat"] <= zone["lat_max"]) & (data["lon"] >= zone["lon_min"]) & (data["lon"] <= zone["lon_max"]))
        months = data["time"][inz].astype("datetime64[M]")
        month = months.astype(int) % 12 + 1
        hourly = {"syear": months.astype("datetime64[Y]").astype(int) + 1970 + (month == 12), "season": sf.season_index(month),
                  "rank": np.unique(data["drifter_id"][inz], return_inverse=True)[1]}
        west = -data["ve"][inz].astype(float) * SECONDS_PER_MOVE / 1000
        for name, keep in (("all drifters", np.ones(west.size, dtype=bool)), ("drogued", data["drogue_status"][inz]), ("undrogued", ~data["drogue_status"][inz])):
            sums, counts = season_year_sums(west[keep], hourly["syear"][keep], hourly["season"][keep])
            pooled = sums.sum(axis=0) / counts.sum(axis=0)
            draws = rng.integers(0, sums.shape[0], (n_boot, sums.shape[0]))
            boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            low, high = np.nanpercentile(boot[:, DJF] - boot[:, JJA], [2.5, 97.5])
            rows.append({"test": "velocity", "setting": f"reported westward velocity x 15 days, {name}", "moves": int(keep.sum()),
                         "DJF": round(float(pooled[DJF]), 2), "MAM": round(float(pooled[MAM]), 2), "JJA": round(float(pooled[JJA]), 2), "SON": round(float(pooled[SON]), 2),
                         "djf_minus_jja": round(float(pooled[DJF] - pooled[JJA]), 2), "djf_jja_low": round(float(low), 2), "djf_jja_high": round(float(high), 2),
                         "son_minus_jja": round(float(pooled[SON] - pooled[JJA]), 2), "note": "hourly readings in the coastal box"})
            print(f"velocity     {name:44s} DJF-JJA {pooled[DJF] - pooled[JJA]:+7.1f} [{low:+.0f},{high:+.0f}]")

    # Outliers
    km = np.hypot(t["east_km"], t["north_km"])
    for q in (0.999, 0.99):
        cut = np.quantile(km[band], q)
        signal("outliers", f"dropping moves over {cut:.0f} km (fastest {100 * (1 - q):g}%)", offshore, band & (km <= cut))
    rows.append({"test": "outliers", "setting": "medians", "moves": int(band.sum()),
                 **{sf.SEASONS[s]: round(float(np.median(offshore[band & (t["season"] == s)])), 2) for s in range(4)}})

    # The box
    for name, m in (("10-20E, nearer the coast", t["start_lon"] >= 10), ("0-10E, further out", t["start_lon"] < 10),
                    ("26-18S, northern half", t["start_lat"] >= -26), ("34-26S, southern half", t["start_lat"] < -26)):
        signal("box", name, offshore, band & m)

    # Direction
    angle = np.radians(COAST_ANGLE)
    signal("direction", f"true offshore axis, coast {COAST_ANGLE:g} deg west of north", -t["east_km"] * np.cos(angle) - t["north_km"] * np.sin(angle), band)

    # Time step
    for tau in (10, 30):
        tt = sf.load_table(os.path.join(PROCESSED_DIR, f"transitions_tau{tau}d.npz"), region)
        signal("time step", f"{tau}-day moves, rescaled to km per 15 days", -tt["east_km"] * 15 / tau, tt["in_band"] & ~tt["exited"], tt)

    # Any one year
    gaps = []
    for y in np.unique(t["syear"][band]):
        m = band & (t["syear"] != y)
        means = [offshore[m & (t["season"] == s)].mean() for s in range(4)]
        gaps.append((means[DJF] - means[JJA], means[SON] - means[JJA]))
    gaps = np.array(gaps)
    rows.append({"test": "any one year", "setting": f"each of {len(gaps)} years left out", "moves": int(band.sum()),
                 "djf_jja_low": round(float(gaps[:, 0].min()), 2), "djf_jja_high": round(float(gaps[:, 0].max()), 2),
                 "son_jja_low": round(float(gaps[:, 1].min()), 2), "son_jja_high": round(float(gaps[:, 1].max()), 2),
                 "note": "low and high are the smallest and largest gaps, not an interval"})

    # Chance
    for name, values, mask in (("westward km", offshore, band), ("northward km", t["north_km"], band),
                               ("% of moves leaving R", 100.0 * t["exited"], np.ones_like(band))):
        observed, shuffled = shuffled_gap(values, mask, t, N_SHUFFLE, rng)
        rows.append({"test": "chance", "setting": f"{name}: biggest gap between seasons, labels shuffled within years", "moves": int(mask.sum()),
                     "observed_gap": round(observed, 3), "shuffled_median": round(float(np.median(shuffled)), 3), "shuffled_95th": round(float(np.quantile(shuffled, 0.95)), 3),
                     "pct_shuffles_at_least_as_big": round(100 * float(np.mean(shuffled >= observed)), 2)})
        print(f"chance       {name:44s} gap {observed:.2f}; shuffled median {np.median(shuffled):.2f}, 95th {np.quantile(shuffled, 0.95):.2f}; "
              f"as big in {100 * np.mean(shuffled >= observed):.2f}%")

    # The forecast
    groups = sf.group_rows(t)
    main_forecast = forecast_row("settings as seasonal_forecast.py", offshore, band, t, groups, n_boot, rng)
    rows.append(main_forecast)
    for md in (5, 10, 20, 30):
        rows.append(forecast_row(f"MIN_DRIFTERS = {md}", offshore, band, t, groups, n_boot, rng, min_drifters=md))
    for fy in (2005, 2010, 2012, 2015):
        rows.append(forecast_row(f"FIRST_TEST_YEAR = {fy}", offshore, band, t, groups, n_boot, rng, first_test_year=fy))
    for name, tt in tables.items():
        rows.append(forecast_row(f"{name}, seasons with 10+ drifters", -tt["east_km"], tt["in_band"] & ~tt["exited"], tt, sf.group_rows(tt), n_boot, rng, min_drifters=10))
    series, moves = sf.season_year_series(offshore, band, t, groups)
    observed, shuffled = shuffled_skill(series, moves, N_SHUFFLE_FORECAST, rng)
    rows.append({"test": "forecast", "setting": "season labels shuffled within years", "skill_pct": round(100 * observed, 1),
                 "shuffled_median": round(100 * float(np.median(shuffled)), 1), "shuffled_95th": round(100 * float(np.quantile(shuffled, 0.95)), 1),
                 "pct_shuffles_at_least_as_big": round(100 * float(np.mean(shuffled >= observed)), 2), "note": f"{shuffled.size} shuffles"})
    for r in rows:
        if r["test"] == "forecast":
            print({k: v for k, v in r.items() if k not in ("test",)})

    # The release, by drifter type
    for name, tt in tables.items():
        for a in (sf.ALPHA_RELEASE, 1000):
            _, _, _, summary = sf.release_table(tt, region, N_BOOT_RELEASE, rng, alpha=a)
            est, low, high = summary[("DJF minus all seasons", 6, "pct_in_coastal_zone")]
            rows.append({"test": "release", "setting": f"{name}, alpha {a}: DJF minus all seasons, % in the coastal zone after 3 months",
                         "djf_minus_all": round(float(est), 3), "djf_low": round(float(low), 3), "djf_high": round(float(high), 3),
                         "DJF": round(float(summary[("DJF", 6, "pct_in_coastal_zone")][0]), 3), "all_seasons": round(float(summary[("All seasons", 6, "pct_in_coastal_zone")][0]), 3)})
            print(rows[-1]["setting"], est, low, high)

    out = os.path.join(RESULTS_DIR, "seasonal_robustness.csv")
    sf.write_csv(out, rows)
    block = app_block(base, by_type["undrogued only"], by_type["drogued only"], main_forecast)
    with open(APP_JSON_PATH, "w") as f:
        json.dump(block, f, indent=1)
    print(f"Saved {out} and {APP_JSON_PATH}")
    print(json.dumps({k: block[k] for k in ("summary", "notes")}, indent=1))
