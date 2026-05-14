from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import os

from satellite_module.config.settings import (
    BIOMASS_MODEL_CONFIG,
    ANOMALY_CONFIG,
    VALIDATION_THRESHOLDS,
    MODEL_DIR
)


class ChangeDetector:
    def __init__(self):
        # Rule-based thresholds
        self.ndvi_drop_threshold = 0.15
        self.high_risk_area_km2 = 1.0
        self.large_deforestation_threshold = 2.0
        
        # ML Models (initially None, load via load_models())
        self.biomass_model = None
        self.anomaly_detector = None
        self.biomass_model_name = None
        self.biomass_metrics = None
        self.anomaly_config = None
        
        print("ChangeDetector Ready (ML-enhanced version)")

    
    def detect_abrupt_change(self, ndvi_change_mean):
        if ndvi_change_mean is None:
            return False
        return ndvi_change_mean < -self.ndvi_drop_threshold

    def detect_anomalous_loss(self, forest_loss_sqkm):
        return forest_loss_sqkm > self.high_risk_area_km2

    def detect_boundary_irregularity(self, ndvi_change_mean):
        if ndvi_change_mean is None:
            return False
        return abs(ndvi_change_mean) > 0.3

    def detect_planned_clearing(self, forest_loss_sqkm):
        return forest_loss_sqkm > self.large_deforestation_threshold

    def calculate_risk_score(self, signals):
        score = 0

        if signals["abrupt_change"]:
            score += 0.35

        if signals["anomalous_loss"]:
            score += 0.30

        if signals["suspicious_regularization"]:
            score += 0.20

        if signals["planned_clearing"]:
            score += 0.15

        score = round(min(score, 1.0), 2)

        if score >= 0.6:
            level = "High"
        elif score >= 0.3:
            level = "Medium"
        else:
            level = "Low"

        confidence = round((1 - score) * 100, 1)

        return {
            "fraud_risk_score": score,
            "risk_level": level,
            "confidence_percent": confidence
        }

    def analyze(
        self,
        ndvi_stats_before,
        ndvi_stats_after,
        forest_area_before,
        forest_area_after,
        percentage_loss
    ):
        ndvi_change_mean = (
            ndvi_stats_after["mean"] - ndvi_stats_before["mean"]
        )

        forest_loss = forest_area_before - forest_area_after

        signals = {
            "abrupt_change": self.detect_abrupt_change(ndvi_change_mean),
            "anomalous_loss": self.detect_anomalous_loss(forest_loss),
            "suspicious_regularization": self.detect_boundary_irregularity(ndvi_change_mean),
            "planned_clearing": self.detect_planned_clearing(forest_loss)
        }

        risk_summary = self.calculate_risk_score(signals)

        return {
            "ndvi_change_mean": round(ndvi_change_mean, 4),
            "forest_loss_sqkm": round(forest_loss, 3),
            "percentage_loss": percentage_loss,
            "signals": signals,
            **risk_summary
        }

    # ==================== ML METHODS ====================

    def load_models(self):
        """Load trained ML models (handles dictionary format)"""
        print("\n" + "-"*50)
        print("Loading Trained Models")
        print("-"*50)
        
        # Find the latest trained models
        training_dirs = list(MODEL_DIR.glob("training_*"))
        anomaly_dirs = list(MODEL_DIR.glob("anomaly_*"))
        
        if not training_dirs:
            print(" No trained biomass models found!")
            print("  Please run train_biomass_model.py first")
            return
        
        if not anomaly_dirs:
            print(" No trained anomaly detectors found!")
            print("  Please run train_anomaly_detector.py first")
            return
        
        # Get latest directories
        latest_training = max(training_dirs, key=os.path.getctime)
        latest_anomaly = max(anomaly_dirs, key=os.path.getctime)
        
        print(f"Latest training directory: {latest_training.name}")
        print(f"Latest anomaly directory: {latest_anomaly.name}")
        
        # Find model files
        biomass_files = list(latest_training.glob("*_model_*.pkl"))
        anomaly_files = list(latest_anomaly.glob("anomaly_detector_*.pkl"))
        
        if not biomass_files:
            print(" No biomass model files found!")
            return
        
        if not anomaly_files:
            print(" No anomaly detector files found!")
            return
        
        # Load biomass model (prefer Random Forest if available)
        biomass_model_path = None
        for f in biomass_files:
            if "random_forest" in str(f).lower():
                biomass_model_path = f
                print(f" Found Random Forest model: {f.name}")
                break
        
        if not biomass_model_path:
            biomass_model_path = biomass_files[0]
            print(f" Using model: {biomass_files[0].name}")
        
        # Load the saved data
        biomass_data = joblib.load(biomass_model_path)
        anomaly_data = joblib.load(anomaly_files[0])
        
        # Store biomass model info
        if isinstance(biomass_data, dict):
            self.biomass_model = biomass_data
            self.biomass_model_name = biomass_data.get('name', 'Unknown')
            self.biomass_metrics = biomass_data.get('metrics', {}).get('test', {})
            print(f" Loaded biomass model: {self.biomass_model_name}")
            print(f"  - Features: {len(biomass_data.get('feature_cols', []))}")
            if self.biomass_metrics:
                print(f"  - Test R2: {self.biomass_metrics.get('r2', 'N/A'):.4f}")
        else:
            self.biomass_model = {'model': biomass_data, 'name': 'Unknown'}
            self.biomass_model_name = 'Unknown'
            self.biomass_metrics = {}
            print(f" Loaded biomass model directly")
        
        # Store anomaly detector info
        if isinstance(anomaly_data, dict):
            self.anomaly_detector = anomaly_data
            self.anomaly_config = anomaly_data.get('config', {})
            print(f" Loaded anomaly detector")
            if self.anomaly_config:
                print(f"  - Contamination: {self.anomaly_config.get('contamination', 'N/A')}")
        else:
            self.anomaly_detector = {'model': anomaly_data}
            self.anomaly_config = {}
            print(f" Loaded anomaly detector directly")

    def estimate_biomass(self, feature_vector):
        """
        Estimate biomass using trained ML model
        """
        if self.biomass_model is None:
            print("Biomass model not loaded. Call load_models() first.")
            return None
        
        # Get the actual model
        if isinstance(self.biomass_model, dict):
            if 'model' in self.biomass_model:
                model = self.biomass_model['model']
            else:
                model = self.biomass_model
        else:
            model = self.biomass_model
        
        # Convert to DataFrame
        df = pd.DataFrame([feature_vector])
        
        # Select features if we have the list
        if isinstance(self.biomass_model, dict) and 'feature_cols' in self.biomass_model:
            expected_cols = self.biomass_model['feature_cols']
            available_cols = [col for col in expected_cols if col in df.columns]
            if available_cols:
                df = df[available_cols]
        
        # Predict
        try:
            estimated_biomass = model.predict(df)[0]
            return float(estimated_biomass)
        except Exception as e:
            print(f"Error in biomass prediction: {e}")
            return None

    def detect_anomaly(self, feature_vector):
        """
        Detect if project is anomalous using Isolation Forest
        """
        if self.anomaly_detector is None:
            print("Anomaly detector not loaded. Call load_models() first.")
            return None
        
        # Get the actual detector
        if isinstance(self.anomaly_detector, dict):
            if 'model' in self.anomaly_detector:
                detector = self.anomaly_detector['model']
            else:
                detector = self.anomaly_detector
        else:
            detector = self.anomaly_detector
        
        df = pd.DataFrame([feature_vector])
        
        try:
            prediction = detector.predict(df)[0]
            score = detector.score_samples(df)[0]
            
            return {
                "is_anomaly": prediction == -1,
                "anomaly_score": float(score),
                "status": "ANOMALY" if prediction == -1 else "NORMAL"
            }
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "status": "ERROR"
            }

    # ==================== TEMPLATE-BASED REPORT ====================

    def generate_report(
        self,
        # Required parameters (no defaults) - MUST come first
        country,
        state,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        area_sq_m,
        area_km2,
        year_before,
        year_after,
        # Basic indices (required)
        ndvi_stats_before,
        ndvi_stats_after,
        evi_stats_before,
        evi_stats_after,
        ndwi_stats_before,
        ndwi_stats_after,
        # Forest metrics (required)
        forest_area_before,
        forest_area_after,
        # Optional parameters (with defaults) - MUST come after all required ones
        forest_pixels_before=None,
        forest_pixels_after=None,
        # Advanced indices (optional)
        savi_stats_before=None,
        savi_stats_after=None,
        msavi2_stats_before=None,
        msavi2_stats_after=None,
        ndmi_stats_before=None,
        ndmi_stats_after=None,
        nbr_stats_before=None,
        nbr_stats_after=None,
        re_ndvi_stats_before=None,
        re_ndvi_stats_after=None,
        rep_stats_before=None,
        rep_stats_after=None,
        # ML Results (optional)
        biomass_before=None,
        biomass_after=None,
        biomass_change=None,
        anomaly_result=None,
        # Analysis results (optional)
        analysis_results=None,
        # Additional metadata (optional)
        cloud_coverage_before=None,
        cloud_coverage_after=None,
        image_count_before=None,
        image_count_after=None
        ):
        """Generate comprehensive report using a template"""
        
        analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate forest loss
        forest_loss = None
        forest_loss_percent = None
        if forest_area_before is not None and forest_area_after is not None:
            forest_loss = forest_area_before - forest_area_after
            if forest_area_before > 0:
                forest_loss_percent = (forest_loss / forest_area_before * 100)
        
        # Format function for stats
        def format_stats(stats):
            if stats is None:
                return "No data available"
            return f"Mean={stats.get('mean', 'N/A'):.4f}, Min={stats.get('min', 'N/A'):.4f}, Max={stats.get('max', 'N/A'):.4f}"
        
        # ============ FORMAT ALL VALUES HERE ============
        
        # Basic indices
        ndvi_before_str = format_stats(ndvi_stats_before)
        ndvi_after_str = format_stats(ndvi_stats_after)
        evi_before_str = format_stats(evi_stats_before)
        evi_after_str = format_stats(evi_stats_after)
        ndwi_before_str = format_stats(ndwi_stats_before)
        ndwi_after_str = format_stats(ndwi_stats_after)
        
        # Advanced indices
        savi_before_str = format_stats(savi_stats_before) if savi_stats_before else "Not computed"
        savi_after_str = format_stats(savi_stats_after) if savi_stats_after else "Not computed"
        msavi2_before_str = format_stats(msavi2_stats_before) if msavi2_stats_before else "Not computed"
        msavi2_after_str = format_stats(msavi2_stats_after) if msavi2_stats_after else "Not computed"
        ndmi_before_str = format_stats(ndmi_stats_before) if ndmi_stats_before else "Not computed"
        ndmi_after_str = format_stats(ndmi_stats_after) if ndmi_stats_after else "Not computed"
        nbr_before_str = format_stats(nbr_stats_before) if nbr_stats_before else "Not computed"
        nbr_after_str = format_stats(nbr_stats_after) if nbr_stats_after else "Not computed"
        re_ndvi_before_str = format_stats(re_ndvi_stats_before) if re_ndvi_stats_before else "Not computed"
        re_ndvi_after_str = format_stats(re_ndvi_stats_after) if re_ndvi_stats_after else "Not computed"
        rep_before_str = format_stats(rep_stats_before) if rep_stats_before else "Not computed"
        rep_after_str = format_stats(rep_stats_after) if rep_stats_after else "Not computed"
        
        # Forest values
        if forest_area_before is not None:
            forest_before_val = f"{forest_area_before:.3f}"
            forest_before_ha = f"{forest_area_before * 100:.2f}"
        else:
            forest_before_val = "N/A"
            forest_before_ha = "N/A"
        
        if forest_area_after is not None:
            forest_after_val = f"{forest_area_after:.3f}"
            forest_after_ha = f"{forest_area_after * 100:.2f}"
        else:
            forest_after_val = "N/A"
            forest_after_ha = "N/A"
        
        forest_loss_val = f"{forest_loss:+.3f}" if forest_loss is not None else "N/A"
        forest_loss_pct = f"{forest_loss_percent:+.2f}" if forest_loss_percent is not None else "N/A"
        forest_pixels_before_val = forest_pixels_before if forest_pixels_before is not None else "N/A"
        forest_pixels_after_val = forest_pixels_after if forest_pixels_after is not None else "N/A"
        
        # ML metrics
        model_name = self.biomass_model_name if self.biomass_model_name else "N/A"
        
        if self.biomass_metrics:
            r2_val = f"{self.biomass_metrics.get('r2', 0):.4f}"
            rmse_val = f"{self.biomass_metrics.get('rmse', 0):.2f}"
            mae_val = f"{self.biomass_metrics.get('mae', 0):.2f}"
        else:
            r2_val = "N/A"
            rmse_val = "N/A"
            mae_val = "N/A"
        
        # Biomass values
        biomass_before_val = f"{biomass_before:.2f}" if biomass_before is not None else "N/A"
        biomass_after_val = f"{biomass_after:.2f}" if biomass_after is not None else "N/A"
        biomass_change_val = f"{biomass_change:+.2f}" if biomass_change is not None else "N/A"
        
        # Carbon stock
        if biomass_before and area_km2:
            carbon_before_val = f"{biomass_before * area_km2 * 0.5:.2f}"
        else:
            carbon_before_val = "N/A"
        
        if biomass_after and area_km2:
            carbon_after_val = f"{biomass_after * area_km2 * 0.5:.2f}"
        else:
            carbon_after_val = "N/A"
        
        # Anomaly results
        if anomaly_result:
            anomaly_status = anomaly_result.get('status', 'N/A')
            anomaly_score_val = f"{anomaly_result.get('anomaly_score', 0):.4f}"
            anomaly_interpretation = "This project shows unusual patterns that warrant investigation" if anomaly_result.get('is_anomaly', False) else "Project patterns appear normal"
        else:
            anomaly_status = "N/A"
            anomaly_score_val = "N/A"
            anomaly_interpretation = "No anomaly detection available"
        
        # Anomaly config
        if self.anomaly_config:
            contamination = self.anomaly_config.get('contamination', 'N/A')
            estimators = self.anomaly_config.get('n_estimators', 'N/A')
        else:
            contamination = "N/A"
            estimators = "N/A"
        
        # Risk assessment
        if analysis_results:
            risk_score = str(analysis_results.get('fraud_risk_score', 'N/A'))
            risk_level = analysis_results.get('risk_level', 'N/A')
            confidence = str(analysis_results.get('confidence_percent', 'N/A'))
            ndvi_change = str(analysis_results.get('ndvi_change_mean', 'N/A'))
            
            # Signals
            signals = analysis_results.get('signals', {})
            abrupt = "Yes" if signals.get('abrupt_change', False) else "No"
            anomalous = "Yes" if signals.get('anomalous_loss', False) else "No"
            suspicious = "Yes" if signals.get('suspicious_regularization', False) else "No"
            planned = "Yes" if signals.get('planned_clearing', False) else "No"
        else:
            risk_score = "N/A"
            risk_level = "N/A"
            confidence = "N/A"
            ndvi_change = "N/A"
            abrupt = "N/A"
            anomalous = "N/A"
            suspicious = "N/A"
            planned = "N/A"
        
        # Cloud and image counts
        cloud_before = f"{cloud_coverage_before:.1f}%" if cloud_coverage_before is not None else "N/A"
        cloud_after = f"{cloud_coverage_after:.1f}%" if cloud_coverage_after is not None else "N/A"
        images_before = image_count_before if image_count_before is not None else "N/A"
        images_after = image_count_after if image_count_after is not None else "N/A"
        
        # Determine recommendation based on risk level
        if risk_level == "High":
            recommendation = """HIGH RISK: Immediate investigation required
    • Multiple fraud signals detected
    • Project should be flagged for manual review
    • Consider site visit or high-resolution imagery analysis
    • Verify carbon credit claims against satellite evidence"""
        elif risk_level == "Medium":
            recommendation = """MEDIUM RISK: Monitor closely
    • Some fraud signals detected
    • Recommend quarterly monitoring
    • Compare with neighboring areas
    • Check historical patterns for context"""
        elif risk_level == "Low":
            recommendation = """LOW RISK: Project appears legitimate
    • No significant fraud signals detected
    • Standard monitoring recommended
    • Continue regular checks
    • Document as verified project"""
        else:
            recommendation = "Insufficient data for recommendation"
        
        # Add anomaly warning if needed
        if anomaly_result and anomaly_result.get('is_anomaly', False):
            recommendation += """
    
    ANOMALY DETECTED: Statistical outlier
    • This project shows unusual patterns compared to training data
    • Review feature contributions to understand why
    • Consider if natural variation or potential issue"""
        
        # Create the template with all placeholders
        template = f"""
    ================================================================================
            AI-POWERED CARBON CREDIT FRAUD DETECTION REPORT
    ================================================================================

    ANALYSIS METADATA
    --------------------------------------------------------------------------------
    Analysis Date & Time : {analysis_date}
    Country              : {country}
    State / Region       : {state}
    Selected Years       : {year_before} vs {year_after}

    AREA OF INTEREST (AOI)
    --------------------------------------------------------------------------------
    Bounding Box:
    Min Longitude : {min_lon}
    Min Latitude  : {min_lat}
    Max Longitude : {max_lon}
    Max Latitude  : {max_lat}

    Area:
    Square Meters        : {area_sq_m:,.2f}
    Square Kilometers    : {area_km2:,.3f}
    Hectares             : {area_km2 * 100:,.2f}

    SATELLITE DATA QUALITY
    --------------------------------------------------------------------------------
    Year {year_before}:
    Images available     : {images_before}
    Cloud coverage       : {cloud_before}

    Year {year_after}:
    Images available     : {images_after}
    Cloud coverage       : {cloud_after}

    ================================================================================
    VEGETATION INDICES ANALYSIS
    ================================================================================

    BASIC INDICES
    --------------------------------------------------------------------------------
    NDVI Before            : {ndvi_before_str}
    NDVI After             : {ndvi_after_str}
    NDVI Change            : {ndvi_change}

    EVI Before             : {evi_before_str}
    EVI After              : {evi_after_str}

    NDWI Before            : {ndwi_before_str}
    NDWI After             : {ndwi_after_str}

    ADVANCED INDICES
    --------------------------------------------------------------------------------
    SAVI Before            : {savi_before_str}
    SAVI After             : {savi_after_str}

    MSAVI2 Before          : {msavi2_before_str}
    MSAVI2 After           : {msavi2_after_str}

    NDMI Before            : {ndmi_before_str}
    NDMI After             : {ndmi_after_str}

    NBR Before             : {nbr_before_str}
    NBR After              : {nbr_after_str}

    Red Edge NDVI Before   : {re_ndvi_before_str}
    Red Edge NDVI After    : {re_ndvi_after_str}

    Red Edge Position Before : {rep_before_str}
    Red Edge Position After  : {rep_after_str}

    ================================================================================
    FOREST COVER ANALYSIS
    ================================================================================

    Forest Area:
    Before ({year_before})      : {forest_before_val} sq km ({forest_before_ha} hectares)
    After  ({year_after})       : {forest_after_val} sq km ({forest_after_ha} hectares)

    Forest Change:
    Absolute Change       : {forest_loss_val} sq km
    Percentage Change     : {forest_loss_pct}%

    Forest Pixels:
    Before                : {forest_pixels_before_val} pixels
    After                 : {forest_pixels_after_val} pixels

    ================================================================================
    BIOMASS ESTIMATION (ML MODEL)
    ================================================================================

    Model Used              : {model_name}

    Model Performance (Test Set):
    R² Score              : {r2_val}
    RMSE                  : {rmse_val} Mg/ha
    MAE                   : {mae_val} Mg/ha

    Estimated Biomass:
    Before ({year_before})      : {biomass_before_val} Mg/ha
    After  ({year_after})       : {biomass_after_val} Mg/ha
    Change                 : {biomass_change_val} Mg/ha

    Total Carbon Stock (estimated):
    Before                 : {carbon_before_val} tonnes C
    After                  : {carbon_after_val} tonnes C
    (using 0.5 conversion factor)

    ================================================================================
    ANOMALY DETECTION
    ================================================================================

    Model                   : Isolation Forest

    Configuration:
    Contamination         : {contamination}
    Estimators            : {estimators}

    Results:
    Status                : {anomaly_status}
    Anomaly Score         : {anomaly_score_val}
    Interpretation        : {anomaly_interpretation}

    ================================================================================
    FRAUD RISK ASSESSMENT
    ================================================================================

    Risk Score              : {risk_score}/1.0
    Risk Level              : {risk_level}
    Confidence              : {confidence}%

    Signals Triggered:
    --------------------------------------------------------------------------------
    Abrupt Change         : {abrupt}
    Anomalous Loss        : {anomalous}
    Suspicious Regularization : {suspicious}
    Planned Clearing      : {planned}

    Signal Definitions:
    --------------------------------------------------------------------------------
    • Abrupt Change        : Sudden NDVI drop > {self.ndvi_drop_threshold} (possible deforestation)
    • Anomalous Loss       : Forest loss > {self.high_risk_area_km2} km² (unusual area)
    • Suspicious Regularization : |NDVI change| > 0.3 (too perfect patterns)
    • Planned Clearing     : Forest loss > {self.large_deforestation_threshold} km² (large-scale)

    ================================================================================
    RECOMMENDATIONS
    ================================================================================

    {recommendation}

    ================================================================================
    TECHNICAL SUMMARY
    ================================================================================

    ML Models Used:
    • Biomass Estimation  : {model_name}
    • Anomaly Detection   : Isolation Forest

    Data Sources:
    • Satellite           : Sentinel-2 (10m resolution)
    • Training Data       : GEDI LiDAR biomass samples
    • Administrative      : FAO GAUL 2015

    Processing:
    • Cloud Filter        : <10% cloud cover
    • Seasonal Window     : June-September
    • Resolution          : 10m (bands), 100m (statistics)

    ================================================================================
    Report Generated by AI-Powered Carbon Credit Fraud Detection System
    ================================================================================
    """
        
        return template