# ingest.py
# library imports
import clouddrift as cd

def load_gdp():
    """Lazily open the latest file from the library of clouddrift datasets.
    
    Returns:
        xarray.Dataset: GDP dataset.

    Raises:
        AssertionError: If the dataset contains fewer than 19,000 trajectories.
    """
    ds = cd.datasets.gdp1h()

    # Extract actual trajectories count from xarray dimension
    actual_trajectories = ds.sizes['traj']
    baseline_expected = 19000 # confirmed at least 19000
    
    # Resilient assertion: checks if number of trajectories meets baseline or has grown
    assert actual_trajectories >= baseline_expected, (
        f"Data loss detected! Found {actual_trajectories} trajectories, "
        f" but expected at least {baseline_expected}."
    )

    return ds

if __name__ == "__main__":
    print("Loading GDP dataset...")
    try:
        ds = load_gdp()
        print(f"Success! Loaded {ds.sizes['traj']} trajectories.")
    except AssertionError as e:
        print(f"Error: {e}")