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
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(REPO_ROOT, "outputs", "figures")
RESULTS_DIR = os.path.join(REPO_ROOT, "outputs", "results")


def load_region_bounds(yaml_path=REGION_CONFIG):
    """Load region R from config/region.yaml: name (file names), label (figure titles),
    lat/lon limits and the working cell size."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)
