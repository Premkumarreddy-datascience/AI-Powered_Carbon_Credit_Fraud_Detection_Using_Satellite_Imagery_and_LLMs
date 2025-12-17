import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"

# Create directories
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys (set these in .env file)
SENTINELHUB_CLIENT_ID = os.getenv("SENTINELHUB_CLIENT_ID", "")
SENTINELHUB_CLIENT_SECRET = os.getenv("SENTINELHUB_CLIENT_SECRET", "")
EARTHENGINE_TOKEN = os.getenv("EARTHENGINE_TOKEN", "")

# Satellite settings
SATELLITE_CONFIG = {
    "sentinel-2": {
        "bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
        "resolution": 10,
        "cloud_coverage": 0.3
    }
}

# Analysis parameters
ANALYSIS_CONFIG = {
    "ndvi_threshold": 0.3,
    "change_threshold": 0.15,
    "min_forest_area": 1.0,
    "time_window_days": 365
}