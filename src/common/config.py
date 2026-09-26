"""
config.py

Shared paths and the region config, so every script reads the same box and
writes outputs to the same place.
"""
import os
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGION_CONFIG = os.path.join(REPO_ROOT, "config", "region.yaml")
INTERIM_DIR = os.path.join(REPO_ROOT, "data", "interim")
FIGURES_DIR = os.path.join(REPO_ROOT, "outputs", "figures")
RESULTS_DIR = os.path.join(REPO_ROOT, "outputs", "results")


def load_region_bounds(yaml_path=REGION_CONFIG):
    """Load region R's box (name, lat/lon limits, working cell size) from config/region.yaml."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)
