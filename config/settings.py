import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# LOAD ENVIRONMENT VARIABLES
load_dotenv()

project_root_env = os.getenv("PROJECT_ROOT")

if not project_root_env:
    raise EnvironmentError(
        "PROJECT_ROOT environment variable not set. "
        "Please define it in .env or system environment variables."
    )

PROJECT_ROOT = Path(project_root_env)

# DATA DIRECTORIES
DATA_DIR = PROJECT_ROOT / "satellite_module" / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_BASE_DIR = DATA_DIR / "outputs"
TRAINING_DATA_DIR = DATA_DIR / "training"
MODEL_DIR = PROJECT_ROOT / "satellite_module" / "models"

for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_BASE_DIR, 
                  TRAINING_DATA_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# RUNTIME OUTPUT FOLDER
def create_run_output_folder():
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    run_folder = OUTPUT_BASE_DIR / timestamp
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder

# GOOGLE EARTH ENGINE CONFIG
GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID", "")

# DEFAULT AOI
DEFAULT_AOI = {
    "min_lon": 78.3,
    "min_lat": 17.3,
    "max_lon": 78.6,
    "max_lat": 17.6
}

# SATELLITE CONFIGURATION
SATELLITE_CONFIG = {
    "sentinel-2": {
        "dataset": "COPERNICUS/S2_SR_HARMONIZED",
        "bands": {
            "red": "B4",
            "nir": "B8",
            "green": "B3",
            "blue": "B2",
            "red_edge1": "B5",
            "red_edge2": "B6",
            "red_edge3": "B7",
            "red_edge4": "B8A",
            "swir1": "B11",
            "swir2": "B12"
        },
        "resolution": 10,
        "max_cloud_percent": 10
    }
}

# ANALYSIS CONFIGURATION
ANALYSIS_CONFIG = {
    # Index thresholds
    "ndvi_threshold": 0.3,
    "change_threshold": 0.15,
    
    # Forest thresholds
    "forest_ndvi_threshold": 0.4,
    "forest_evi_threshold": 0.2,
    "forest_ndwi_threshold": 0.2,
    
    # Forest risk threshold
    "min_forest_area_km2": 1.0,
    
    # Earth Engine reduce scale
    "scale": 100,
    
    # Optional temporal window
    "time_window_days": 365
}

# ML TRAINING CONFIGURATION

# BIOMASS MODEL TRAINING CONFIG
BIOMASS_MODEL_CONFIG = {
    "model_type": "xgboost",  # Options: "xgboost", "random_forest"
    "save_path": MODEL_DIR / "biomass_model.pkl",
    "feature_importance_path": MODEL_DIR / "feature_importance.png",
    
    # Training parameters
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    
    # XGBoost hyperparameters
    "xgboost_params": {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1,
        "random_state": 42
    },
    
    # Random Forest hyperparameters
    "rf_params": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    }
}

# ANOMALY DETECTION CONFIG
ANOMALY_CONFIG = {
    "model_type": "isolation_forest",
    "save_path": MODEL_DIR / "anomaly_detector.pkl",
    
    "isolation_forest_params": {
        "n_estimators": 100,
        "max_samples": "auto",
        "contamination": 0.05,
        "bootstrap": False,
        "random_state": 42
    },
    
    "svm_params": {
        "kernel": "rbf",
        "nu": 0.05,
        "gamma": "scale"
    }
}

# FEATURE ENGINEERING CONFIG
FEATURE_CONFIG = {
    "indices": [
        "ndvi", "evi", "savi", "msavi2", "ndmi", "ndwi", "nbr"
    ],
    
    "texture": {
        "window_sizes": [3, 5],
        "properties": ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    },
    
    "temporal": {
        "window_months": [3, 6, 12],
        "stats": ["mean", "std", "min", "max", "trend"]
    },
    
    "topographic": {
        "use_dem": True,
        "features": ["elevation", "slope", "aspect", "roughness"]
    }
}

# GROUND TRUTH DATA SOURCES
GROUND_TRUTH_CONFIG = {
    "gedi": {
        "dataset": "LARSE/GEDI/GEDI04_A_002_MONTHLY",
        "bands": ["agbd"],
        "quality_filter": "l2_quality_flag == 1 and l4_quality_flag == 1"
    },
    
    "local_training_data": TRAINING_DATA_DIR / "field_plots.csv",
    
    "public_datasets": {
        "biomass_global": "https://data.globalforestwatch.org/datasets/aboveground-biomass",
        "forest_plots": "https://forestplots.net/"
    }
}

# EVALUATION METRICS
EVALUATION_CONFIG = {
    "regression_metrics": ["r2", "rmse", "mae", "mape"],
    "anomaly_metrics": ["precision", "recall", "f1_at_k"]
}

# PROJECT VALIDATION THRESHOLDS
VALIDATION_THRESHOLDS = {
    "high_risk_ratio": 1.5,
    "medium_risk_ratio": 1.2,
    "anomaly_contamination": 0.05,
    "min_training_samples": 100,
    "confidence_threshold": 0.7
}