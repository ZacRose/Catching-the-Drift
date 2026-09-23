# load.py
# library imports
import clouddrift as cd

def load_gdp():
    """Lazily open the latest file from the library of clouddrift datasets.
    
    Returns:
        xarray.Dataset: GDP dataset.
    """
    return cd.datasets.gdp1h()

if __name__ == "__main__":
    print("Loading GDP dataset...")
    ds = load_gdp()
    print(f"Loaded {ds.sizes['traj']} trajectories.")