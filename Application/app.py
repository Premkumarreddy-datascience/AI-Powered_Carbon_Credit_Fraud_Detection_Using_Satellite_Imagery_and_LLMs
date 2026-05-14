import sys
import os
import json
import ee
import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ------------------------- PATH SETUP -------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.environ['PROJECT_ROOT'] = str(project_root)

# ------------------------- IMPORTS FROM MODULES -------------------------
from satellite_module.config.settings import (
    GEE_PROJECT_ID, MODEL_DIR, OUTPUT_BASE_DIR,
    SATELLITE_CONFIG, ANALYSIS_CONFIG
)
from satellite_module.src.data_downloader import SatelliteDataDownloader
from satellite_module.src.image_processor import ImageProcessor
from satellite_module.src.change_detector import ChangeDetector
from satellite_module.src.visualizer import SatelliteVisualizer


class FraudDetectionApp:
    def __init__(self):
        print("="*70)
        print("CARBON CREDIT FRAUD DETECTION SYSTEM (ML + VISUAL)")
        print("="*70)

        # Earth Engine
        try:
            ee.Initialize(project=GEE_PROJECT_ID)
            print("✓ Earth Engine initialized (with project ID)")
        except:
            try:
                ee.Initialize()
                print("✓ Earth Engine initialized (default)")
            except Exception as e:
                print(f"✗ Earth Engine init failed: {e}")
                sys.exit(1)

        self.downloader = SatelliteDataDownloader()
        self.processor = ImageProcessor()
        self.detector = ChangeDetector()
        self.visualizer = SatelliteVisualizer()

        # Load ML models
        self.detector.load_models()

        # Extract expected feature columns from the biomass model
        if hasattr(self.detector, 'biomass_model') and isinstance(self.detector.biomass_model, dict):
            self.feature_cols = self.detector.biomass_model.get('feature_cols', None)
            if self.feature_cols:
                print(f"✓ Model expects {len(self.feature_cols)} features")
        else:
            self.feature_cols = None

        # Output directory
        self.timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        self.output_dir = OUTPUT_BASE_DIR / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directory: {self.output_dir}")

        self.current_data = {}

    # ------------------------- USER INPUT -------------------------
    def get_user_input(self):
        print("\n" + "-"*50)
        print("ENTER AREA OF INTEREST (AOI)")
        print("-"*50)
        while True:
            try:
                min_lon = float(input("  Min Longitude (e.g., -62.2): "))
                min_lat = float(input("  Min Latitude  (e.g., -10.3): "))
                max_lon = float(input("  Max Longitude (e.g., -62.0): "))
                max_lat = float(input("  Max Latitude  (e.g., -10.1): "))
                if min_lon >= max_lon or min_lat >= max_lat:
                    print("✗ Min must be less than Max. Try again.")
                    continue
                break
            except ValueError:
                print("✗ Please enter valid numbers")

        print("\nEnter years to compare (2015 onward):")
        while True:
            try:
                year_before = int(input("  Year 1 (e.g., 2019): "))
                year_after  = int(input("  Year 2 (e.g., 2022): "))
                if year_before < 2015 or year_after < 2015:
                    print("✗ Sentinel-2 data from 2015")
                    continue
                if year_before >= year_after:
                    print("✗ Year 1 must be less than Year 2")
                    continue
                break
            except ValueError:
                print("✗ Please enter valid years")
        return {'coords': (min_lon, min_lat, max_lon, max_lat), 'years': (year_before, year_after)}

    # ------------------------- SATELLITE DATA & INDICES -------------------------
    def fetch_and_process(self, coords, years):
        min_lon, min_lat, max_lon, max_lat = coords
        y_before, y_after = years
        geometry = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

        country, region = self.downloader.get_location_info(geometry)
        area_sq_m = geometry.area().getInfo()
        area_km2 = round(area_sq_m / 1e6, 3)
        print(f"\nLocation: {country}, {region}")
        print(f"Area: {area_km2} km²")

        print("\nDownloading seasonal composites...")
        img_before = self.downloader.get_seasonal_image(geometry, y_before)
        img_after  = self.downloader.get_seasonal_image(geometry, y_after)

        # Basic indices (for visualisation & rule-based)
        ndvi_before = self.processor.compute_ndvi(img_before)
        ndvi_after  = self.processor.compute_ndvi(img_after)
        evi_before  = self.processor.compute_evi(img_before)
        evi_after   = self.processor.compute_evi(img_after)
        ndwi_before = self.processor.compute_ndwi(img_before)
        ndwi_after  = self.processor.compute_ndwi(img_after)

        # Statistics for basic indices
        ndvi_stats_before = self.processor.compute_statistics(ndvi_before, "NDVI", geometry)
        ndvi_stats_after  = self.processor.compute_statistics(ndvi_after, "NDVI", geometry)
        evi_stats_before  = self.processor.compute_statistics(evi_before, "EVI", geometry)
        evi_stats_after   = self.processor.compute_statistics(evi_after, "EVI", geometry)
        ndwi_stats_before = self.processor.compute_statistics(ndwi_before, "NDWI", geometry)
        ndwi_stats_after  = self.processor.compute_statistics(ndwi_after, "NDWI", geometry)

        # Forest masks
        forest_before = self.processor.forest_mask(ndvi_before, evi_before, ndwi_before)
        forest_after  = self.processor.forest_mask(ndvi_after, evi_after, ndwi_after)

        forest_before_stats = self.processor.calculate_area_from_mask(forest_before, geometry)
        forest_after_stats  = self.processor.calculate_area_from_mask(forest_after, geometry)

        forest_area_before = forest_before_stats['area_sqkm']
        forest_area_after  = forest_after_stats['area_sqkm']
        percentage_loss = self.processor.calculate_percentage_loss(forest_area_before, forest_area_after)

        # ML feature vectors (27 features, missing auxiliaries set to 0)
        print("Computing ML feature vectors (sampling, matching model features)...")
        feature_vector_before = self.processor.get_mean_feature_vector_for_model(
            img_before, geometry, self.feature_cols, scale=100, num_points=2000
        )
        feature_vector_after = self.processor.get_mean_feature_vector_for_model(
            img_after, geometry, self.feature_cols, scale=100, num_points=2000
        )

        # Change maps (for visualisation)
        ndvi_change = self.processor.compute_change(ndvi_before, ndvi_after, "NDVI")
        evi_change  = self.processor.compute_change(evi_before, evi_after, "EVI")
        ndwi_change = self.processor.compute_change(ndwi_before, ndwi_after, "NDWI")

        self.current_data = {
            'geometry': geometry,
            'coords': coords,
            'years': (y_before, y_after),
            'country': country,
            'region': region,
            'area_sq_m': area_sq_m,
            'area_km2': area_km2,
            'ndvi_stats_before': ndvi_stats_before,
            'ndvi_stats_after': ndvi_stats_after,
            'evi_stats_before': evi_stats_before,
            'evi_stats_after': evi_stats_after,
            'ndwi_stats_before': ndwi_stats_before,
            'ndwi_stats_after': ndwi_stats_after,
            'forest_area_before': forest_area_before,
            'forest_area_after': forest_area_after,
            'percentage_loss': percentage_loss,
            'feature_vector_before': feature_vector_before,
            'feature_vector_after': feature_vector_after,
            'ndvi_before': ndvi_before,
            'ndvi_after': ndvi_after,
            'ndvi_change': ndvi_change,
            'evi_before': evi_before,
            'evi_after': evi_after,
            'evi_change': evi_change,
            'ndwi_before': ndwi_before,
            'ndwi_after': ndwi_after,
            'ndwi_change': ndwi_change,
            'forest_before': forest_before,
            'forest_after': forest_after,
        }

    # ------------------------- RULE-BASED & ML ANALYSIS -------------------------
    def run_rule_analysis(self):
        print("\n" + "-"*50)
        print("RUNNING RULE-BASED ANALYSIS")
        print("-"*50)
        results = self.detector.analyze(
            self.current_data['ndvi_stats_before'],
            self.current_data['ndvi_stats_after'],
            self.current_data['forest_area_before'],
            self.current_data['forest_area_after'],
            self.current_data['percentage_loss']
        )
        self.current_data['analysis_results'] = results
        print(f"Risk score: {results['fraud_risk_score']} ({results['risk_level']})")

    def run_ml_analysis(self):
        print("\n" + "-"*50)
        print("RUNNING ML MODELS")
        print("-"*50)
        biomass_before = self.detector.estimate_biomass(self.current_data['feature_vector_before'])
        biomass_after  = self.detector.estimate_biomass(self.current_data['feature_vector_after'])
        if biomass_before and biomass_after:
            biomass_change = biomass_after - biomass_before
        else:
            biomass_before = biomass_after = biomass_change = None
        anomaly_result = self.detector.detect_anomaly(self.current_data['feature_vector_before'])

        self.current_data['biomass_before'] = biomass_before
        self.current_data['biomass_after'] = biomass_after
        self.current_data['biomass_change'] = biomass_change
        self.current_data['anomaly_result'] = anomaly_result

        if biomass_before:
            print(f"Biomass: {biomass_before:.2f} → {biomass_after:.2f} Mg/ha")
        else:
            print("Biomass estimation failed")
        if anomaly_result:
            print(f"Anomaly status: {anomaly_result['status']} (score: {anomaly_result['anomaly_score']:.4f})")

    # ------------------------- VISUAL OUTPUTS -------------------------
    def generate_dashboard(self):
        print("\n" + "-"*50)
        print("GENERATING INTERACTIVE DASHBOARD")
        print("-"*50)
        dashboard = self.visualizer.create_dashboard(
            geometry=self.current_data['geometry'],
            ndvi_before=self.current_data['ndvi_before'],
            ndvi_after=self.current_data['ndvi_after'],
            ndvi_change=self.current_data['ndvi_change'],
            evi_before=self.current_data['evi_before'],
            evi_after=self.current_data['evi_after'],
            evi_change=self.current_data['evi_change'],
            ndwi_before=self.current_data['ndwi_before'],
            ndwi_after=self.current_data['ndwi_after'],
            ndwi_change=self.current_data['ndwi_change'],
            forest_before=self.current_data['forest_before'],
            forest_after=self.current_data['forest_after']
        )
        map_path = self.output_dir / "interactive_dashboard.html"
        self.visualizer.save_map(dashboard, map_path)
        print(f"✓ Dashboard saved: {map_path}")

    def export_static_maps(self):
        print("\n" + "-"*50)
        print("EXPORTING STATIC IMAGES")
        print("-"*50)
        self.visualizer.export_all_static_maps(
            geometry=self.current_data['geometry'],
            output_folder=self.output_dir,
            ndvi_before=self.current_data['ndvi_before'],
            ndvi_after=self.current_data['ndvi_after'],
            ndvi_change=self.current_data['ndvi_change'],
            evi_before=self.current_data['evi_before'],
            evi_after=self.current_data['evi_after'],
            evi_change=self.current_data['evi_change'],
            ndwi_before=self.current_data['ndwi_before'],
            ndwi_after=self.current_data['ndwi_after'],
            ndwi_change=self.current_data['ndwi_change']
        )
        print("✓ Static images saved (PNG)")

    # ------------------------- REPORT & DATA -------------------------
    def save_report(self):
        print("\n" + "-"*50)
        print("SAVING DETAILED REPORT")
        print("-"*50)
        report = self.detector.generate_report(
            country=self.current_data['country'],
            state=self.current_data['region'],
            min_lon=self.current_data['coords'][0],
            min_lat=self.current_data['coords'][1],
            max_lon=self.current_data['coords'][2],
            max_lat=self.current_data['coords'][3],
            area_sq_m=self.current_data['area_sq_m'],
            area_km2=self.current_data['area_km2'],
            year_before=self.current_data['years'][0],
            year_after=self.current_data['years'][1],
            ndvi_stats_before=self.current_data['ndvi_stats_before'],
            ndvi_stats_after=self.current_data['ndvi_stats_after'],
            evi_stats_before=self.current_data['evi_stats_before'],
            evi_stats_after=self.current_data['evi_stats_after'],
            ndwi_stats_before=self.current_data['ndwi_stats_before'],
            ndwi_stats_after=self.current_data['ndwi_stats_after'],
            forest_area_before=self.current_data['forest_area_before'],
            forest_area_after=self.current_data['forest_area_after'],
            analysis_results=self.current_data['analysis_results'],
            biomass_before=self.current_data.get('biomass_before'),
            biomass_after=self.current_data.get('biomass_after'),
            biomass_change=self.current_data.get('biomass_change'),
            anomaly_result=self.current_data.get('anomaly_result')
        )
        report_path = self.output_dir / "final_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✓ Report saved: {report_path}")

    def save_json(self):
        serializable = {
            'timestamp': self.timestamp,
            'coords': self.current_data['coords'],
            'years': self.current_data['years'],
            'country': self.current_data['country'],
            'region': self.current_data['region'],
            'area_km2': self.current_data['area_km2'],
            'ndvi_change_mean': self.current_data['analysis_results']['ndvi_change_mean'],
            'forest_loss_sqkm': self.current_data['analysis_results']['forest_loss_sqkm'],
            'percentage_loss': self.current_data['percentage_loss'],
            'fraud_risk_score': self.current_data['analysis_results']['fraud_risk_score'],
            'risk_level': self.current_data['analysis_results']['risk_level'],
            'signals': self.current_data['analysis_results']['signals'],
            'biomass_before': self.current_data.get('biomass_before'),
            'biomass_after': self.current_data.get('biomass_after'),
            'biomass_change': self.current_data.get('biomass_change'),
            'anomaly_status': self.current_data.get('anomaly_result', {}).get('status'),
            'anomaly_score': self.current_data.get('anomaly_result', {}).get('anomaly_score')
        }
        json_path = self.output_dir / "analysis_results.json"
        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"✓ JSON data saved: {json_path}")

    # ------------------------- MAIN -------------------------
    def run(self):
        print("\n" + "="*70)
        print("STARTING COMPLETE ANALYSIS PIPELINE")
        print("="*70)
        user = self.get_user_input()
        self.fetch_and_process(user['coords'], user['years'])
        self.run_rule_analysis()
        self.run_ml_analysis()
        self.generate_dashboard()
        self.export_static_maps()
        self.save_report()
        self.save_json()
        print("\n" + "="*70)
        print(f"ANALYSIS COMPLETE! Outputs saved to: {self.output_dir}")
        print("="*70)


if __name__ == "__main__":
    app = FraudDetectionApp()
    app.run()