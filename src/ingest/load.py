"""
load.py

Related task: Week 1 Computational - "lead the data-adequacy count and get the raw dataset loading".
coverage.py imports load_gdp() from here.

Global Drifter Program (GDP) hourly dataset stored as xarray.Dataset (NOT a big table).
Each drifter has its own trajectory of readings, and all drifters' readings are concatenated one after another
into a single long list (clouddrift "ragged array").
coverage.py computes which part of that long list belongs to each drifter.
"""

import clouddrift as cd


def load_gdp():
    """Fetch the GDP hourly dataset from clouddrift's cloud-hosted copy.

    This streams the data lazily and matches Elipot's implementation

    Returns:
        xarray.Dataset: the full GDP dataset.
    """
    return cd.datasets.gdp1h()


if __name__ == "__main__":
    # Run "python3 -m src.ingest.load" to load raw dataset
    print("Loading GDP dataset...")
    ds = load_gdp()
    print(f"Loaded {ds.sizes['traj']} trajectories.")