"""
displacement_eda.py

Related task: Week 3 Quantitative - EDA for "association with the response" (Data marking
criteria). The response is where material goes in tau = 15 days, so this looks at 15-day
displacement split three ways, each tied to a decision in project_plan.txt section 7:
- drogue status at the start (decision 6: keep undrogued drifters in the main estimate);
- season of the start, southern-hemisphere seasons (Business's first look at seasonality);
- start before vs from 2010 (decision 7: all years vs 2010+).

Displacement is computed only for transitions that stay in R (an exit has no end position).
Dropping exits biases displacement low, because the drifters that leave tend to be the fast
ones, so each group's exit share is reported beside it. Groups are not independent of each
other (e.g. the drogued share changes over the years), so a difference in one split may partly
reflect another; this is a first look, not a controlled comparison.

Run: python3 -m src.inference.displacement_eda
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from src.common.config import FIGURES_DIR, RESULTS_DIR, load_region_bounds
from src.inference.data_quality import KM_PER_DEG
from src.inference.transition_data import TAU_DAYS, load_transitions

SEASONS = {12: "DJF summer", 1: "DJF summer", 2: "DJF summer",
           3: "MAM autumn", 4: "MAM autumn", 5: "MAM autumn",
           6: "JJA winter", 7: "JJA winter", 8: "JJA winter",
           9: "SON spring", 10: "SON spring", 11: "SON spring"}
SEASON_ORDER = ["DJF summer", "MAM autumn", "JJA winter", "SON spring"]
SPLIT_YEAR = 2010


def displacement_km(table):
    """Eastward, northward and total displacement (km) over tau; NaN for exits.

    Flat-earth approximation at the midpoint latitude, accurate to well under 1% over the
    few hundred km a drifter moves in 15 days.
    """
    dlon = ((table["end_lon"] - table["start_lon"] + 180) % 360) - 180
    mid_lat = np.radians(0.5 * (table["start_lat"] + table["end_lat"]))
    east = dlon * KM_PER_DEG * np.cos(mid_lat)
    north = (table["end_lat"] - table["start_lat"]) * KM_PER_DEG
    east[table["exited"]] = np.nan
    north[table["exited"]] = np.nan
    return east, north, np.hypot(east, north)


def groupings(table):
    """name -> (labels per transition, label order)."""
    months = table["start_time"].astype("datetime64[M]").astype(int) % 12 + 1
    years = table["start_time"].astype("datetime64[Y]").astype(int) + 1970
    return {
        "Drogue status": (np.where(table["drogued"], "Drogued", "Drogue lost"), ["Drogued", "Drogue lost"]),
        "Season of start": (np.array([SEASONS[m] for m in months]), SEASON_ORDER),
        "Start year": (np.where(years < SPLIT_YEAR, f"Before {SPLIT_YEAR}", f"{SPLIT_YEAR}+"),
                       [f"Before {SPLIT_YEAR}", f"{SPLIT_YEAR}+"]),
    }


def group_summary(table, east, north, dist, split_name, labels, order):
    rows = []
    for label in order:
        g = labels == label
        stay = g & ~table["exited"]
        rows.append({
            "split": split_name,
            "group": label,
            "transitions": int(g.sum()),
            "drifters": int(np.unique(table["drifter_id"][g]).size),
            "pct_exited": round(float(100 * table["exited"][g].mean()), 1) if g.any() else float("nan"),
            "median_km": round(float(np.median(dist[stay])), 1) if stay.any() else float("nan"),
            "iqr_km": round(float(np.subtract(*np.percentile(dist[stay], [75, 25]))), 1) if stay.any() else float("nan"),
            "mean_east_km": round(float(np.mean(east[stay])), 1) if stay.any() else float("nan"),
            "mean_north_km": round(float(np.mean(north[stay])), 1) if stay.any() else float("nan"),
        })
    return rows


def plot(table, dist, splits, summaries, out_path):
    fig, axes = plt.subplots(1, len(splits), figsize=(15, 4.6), layout="constrained", sharey=True,
                             gridspec_kw={"width_ratios": [len(o) for _, o in splits.values()]})
    stay = ~table["exited"]
    for ax, (name, (labels, order)) in zip(axes, splits.items()):
        data = [dist[stay & (labels == label)] for label in order]
        ax.boxplot(data, showfliers=False, widths=0.6)
        rows = [r for r in summaries if r["split"] == name]
        ax.set_xticks(range(1, len(order) + 1),
                      [f"{r['group']}\n{r['drifters']} drifters\n{r['pct_exited']}% left R" for r in rows], fontsize=9)
        ax.set_title(name)
    axes[0].set_ylabel(f"{TAU_DAYS}-day displacement (km)")
    fig.suptitle(f"{TAU_DAYS}-day displacement of drifters that stay in R, by drogue status, season and era")
    fig.text(0.01, -0.06, "Boxes: median and interquartile range; whiskers 1.5 x IQR, outliers hidden. "
             "Transitions that leave R have no end position and are excluded; their share is shown per group.",
             fontsize=9)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    table = load_transitions(region)
    east, north, dist = displacement_km(table)
    splits = groupings(table)

    summaries = []
    for name, (labels, order) in splits.items():
        summaries += group_summary(table, east, north, dist, name, labels, order)
    for row in summaries:
        print(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    table_path = os.path.join(RESULTS_DIR, f"displacement_by_group_tau{TAU_DAYS}d.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "displacement_association.png")
    plot(table, dist, splits, summaries, fig_path)
    print(f"Saved {table_path} and {fig_path}")
