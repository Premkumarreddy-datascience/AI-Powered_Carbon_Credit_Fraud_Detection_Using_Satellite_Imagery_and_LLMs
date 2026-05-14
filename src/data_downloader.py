import ee
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from shapely.geometry import Point, box
import requests
import zipfile
import os
import json
import time  # Added for safe_get_info

from satellite_module.config.settings import (
    SATELLITE_CONFIG,
    DEFAULT_AOI,
    GROUND_TRUTH_CONFIG,
    TRAINING_DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
)


class SatelliteDataDownloader:

    def __init__(self):
        self.dataset = SATELLITE_CONFIG["sentinel-2"]["dataset"]
        self.max_cloud = SATELLITE_CONFIG["sentinel-2"]["max_cloud_percent"]
        self.bands = SATELLITE_CONFIG["sentinel-2"]["bands"]
        self.all_bands = list(SATELLITE_CONFIG["sentinel-2"]["bands"].values())
        
        # Initialize Earth Engine
        try:
            ee.Initialize()
        except Exception as e:
            print(f"Note: Earth Engine not initialized. Call ee.Initialize() first. Error: {e}")
        
        # GAUL boundaries for location info (added from prepare_training_data)
        self.boundaries = ee.FeatureCollection("FAO/GAUL/2015/level1")
            
        print("SatelliteDataDownloader Ready")
        print(f"Dataset: {self.dataset}")
        print(f"Bands: {len(self.all_bands)} bands available")

    
    def get_aoi(self, geometry=None):
        """Your existing method - kept unchanged"""
        if geometry:
            return geometry

        return ee.Geometry.Rectangle([
            DEFAULT_AOI["min_lon"],
            DEFAULT_AOI["min_lat"],
            DEFAULT_AOI["max_lon"],
            DEFAULT_AOI["max_lat"]
        ])

    def get_image(self, geometry, start_date, end_date):
        aoi = self.get_aoi(geometry)

        collection = (
            ee.ImageCollection(self.dataset)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.max_cloud))
        )

        image = collection.median()
        return image

    def get_seasonal_image(self, geometry, year):
        start_date = f"{year}-06-01"
        end_date   = f"{year}-09-30"
        return self.get_image(geometry, start_date, end_date)

    def compute_aoi_area(self, geometry):
        try:
            area_km2 = geometry.area().divide(1e6).getInfo()
            return round(area_km2, 2)
        except Exception:
            return None

    def get_metadata(self):
        return {
            "dataset": self.dataset,
            "cloud_filter": self.max_cloud,
            "timestamp": datetime.utcnow().isoformat()
        }

    # ============= METHODS MOVED FROM PREPARE_TRAINING_DATA.PY =============

    def safe_get_info(self, ee_object, retries=3):
        for attempt in range(retries):
            try:
                return ee_object.getInfo()
            except Exception:
                print("Earth Engine request failed, retrying...")
                time.sleep(5)
        return None

    def get_location_info(self, geometry):
        try:
            feature = self.boundaries.filterBounds(geometry).first()
            info = self.safe_get_info(feature)
            if info is None:
                return "Unknown", "Unknown"
            props = info["properties"]
            return props.get("ADM0_NAME", "Unknown"), props.get("ADM1_NAME", "Unknown")
        except:
            return "Unknown", "Unknown"

    def download_training_data_from_gedi(self, aoi_coords):
        aoi = ee.Geometry.Rectangle(aoi_coords)
        gedi_dataset = GROUND_TRUTH_CONFIG["gedi"]["dataset"]
        biomass_band = "agbd"
        all_samples = []

        for year in range(2019, 2023):
            print(f"\nProcessing GEDI year: {year}")
            try:
                gedi = (
                    ee.ImageCollection(gedi_dataset)
                    .filterBounds(aoi)
                    .filterDate(f"{year}-01-01", f"{year}-12-31")
                )
                count = self.safe_get_info(gedi.size())
                if count is None or count == 0:
                    print("No GEDI data for this year")
                    continue
                gedi_image = gedi.select([biomass_band]).mosaic()
                samples = gedi_image.sample(
                    region=aoi,
                    scale=25,
                    numPixels=20000,
                    geometries=True,
                    seed=42,
                    tileScale=4
                )
                info = self.safe_get_info(samples)
                if info is None:
                    print("Sampling failed for this year")
                    continue
                year_samples = 0
                for feat in info["features"]:
                    coords = feat["geometry"]["coordinates"]
                    props = feat["properties"]
                    if biomass_band in props:
                        all_samples.append({
                            "longitude": coords[0],
                            "latitude": coords[1],
                            "biomass_mg_ha": props[biomass_band],
                            "year": year
                        })
                        year_samples += 1
                print("Samples collected:", year_samples)
            except Exception as e:
                print("Error processing GEDI data:", e)
        df = pd.DataFrame(all_samples)
        if len(df) == 0:
            return None
        return df

    # ============= EXISTING ML METHODS =============

    def extract_features_at_points(self, points_gdf, year, buffer_size=20):
        """
        Extract Sentinel-2 features at specific point locations
        Used for creating training dataset
        """
        features_list = []
        
        # Get seasonal image
        image = self.get_seasonal_image(None, year)
        
        # Add all bands
        image = image.select(self.all_bands)
        
        # Convert GeoDataFrame to Earth Engine features
        for idx, row in points_gdf.iterrows():
            point = ee.Geometry.Point([row.longitude, row.latitude])
            
            # Extract pixel values
            sample = image.sample(
                region=point,
                scale=10,
                projection=None,
                factor=None,
                numPixels=1,
                dropNulls=True,
                geometries=False
            )
            
            try:
                values = sample.first().getInfo()['properties']
                values['longitude'] = row.longitude
                values['latitude'] = row.latitude
                if 'biomass' in row:
                    values['biomass'] = row.biomass
                features_list.append(values)
            except:
                continue
        
        return pd.DataFrame(features_list)

    def download_batch_aois(self, aoi_list, years, output_dir=None):
        """
        Download data for multiple AOIs for training
        """
        if output_dir is None:
            output_dir = RAW_DATA_DIR / "batch_downloads"
            output_dir.mkdir(exist_ok=True)
        
        for i, aoi_coords in enumerate(aoi_list):
            aoi = ee.Geometry.Rectangle(aoi_coords)
            aoi_data = {}
            
            for year in years:
                image = self.get_seasonal_image(aoi, year)
                
                # Compute indices
                ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
                evi = image.expression(
                    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                    {'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2')}
                ).rename('EVI')
                
                # Add to image
                image = image.addBands([ndvi, evi])
                
                # Get statistics
                stats = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi,
                    scale=100,
                    bestEffort=True
                ).getInfo()
                
                aoi_data[f'year_{year}'] = stats
            
            # Save
            with open(output_dir / f"aoi_{i}_data.json", 'w') as f:
                json.dump(aoi_data, f, indent=2)
        
        print(f"Downloaded {len(aoi_list)} AOIs to {output_dir}")

    def prepare_training_dataset(self, aoi, years, output_path=None):
        """
        Complete pipeline to prepare training dataset
        """
        if output_path is None:
            output_path = TRAINING_DATA_DIR / "training_dataset.csv"
        
        # Step 1: Download GEDI ground truth
        gedi_df = self.download_training_data_from_gedi(aoi)
        
        # Step 2: Extract Sentinel-2 features at GEDI points
        all_features = []
        for year in years:
            print(f"Extracting features for {year}...")
            features_df = self.extract_features_at_points(gedi_df, year)
            features_df['year'] = year
            all_features.append(features_df)
        
        # Step 3: Combine
        final_df = pd.concat(all_features, ignore_index=True)
        final_df.to_csv(output_path, index=False)
        print(f"Training dataset saved to {output_path}")
        
        return final_df