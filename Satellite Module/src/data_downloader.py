import numpy as np
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Polygon, box
import os
import sys
import json

class SimpleSatelliteDataDownloader:
    """Simplified satellite data downloader without rasterio dependency"""
    
    def __init__(self):
        print("Simple Satellite Data Downloader Initialized")
        print("Note: Using synthetic data for demonstration")
    
    def create_sample_data(self, output_path, height=500, width=500, add_deforestation=True):
        """Create realistic sample satellite data"""
        np.random.seed(42)
        
        print(f"Creating sample data: {height}x{width} pixels")
        
        # Base vegetation (healthy forest)
        base_ndvi = np.random.normal(0.7, 0.1, (height, width))
        
        # Add forest patches (clusters of high NDVI)
        for _ in range(15):
            center_x = np.random.randint(50, width-50)
            center_y = np.random.randint(50, height-50)
            radius = np.random.randint(30, 80)
            
            # Create circular forest patch
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            base_ndvi[mask] = np.random.normal(0.8, 0.05, np.sum(mask))
        
        # Add water bodies (low NDVI)
        for _ in range(8):
            center_x = np.random.randint(100, width-100)
            center_y = np.random.randint(100, height-100)
            radius = np.random.randint(20, 50)
            
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            base_ndvi[mask] = np.random.normal(-0.2, 0.1, np.sum(mask))
        
        # Create "before" data (healthy)
        ndvi_before = np.clip(base_ndvi, -1, 1)
        
        # Create "after" data (with deforestation if requested)
        ndvi_after = ndvi_before.copy()
        deforestation_patches = []
        
        if add_deforestation:
            # Add deforestation patches
            for i in range(12):
                center_x = np.random.randint(100, width-100)
                center_y = np.random.randint(100, height-100)
                radius = np.random.randint(15, 45)
                
                y, x = np.ogrid[:height, :width]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                ndvi_after[mask] = np.random.normal(0.1, 0.1, np.sum(mask))
                deforestation_patches.append({
                    'id': i,
                    'x': center_x,
                    'y': center_y,
                    'radius': radius,
                    'area_pixels': np.sum(mask)
                })
        
        # Create synthetic bands
        red_before = np.random.normal(0.15, 0.05, (height, width))
        green_before = ndvi_before * 0.8 + np.random.normal(0.1, 0.05, (height, width))
        blue_before = np.random.normal(0.1, 0.05, (height, width))
        nir_before = ndvi_before * 1.2 + 0.1
        
        red_after = np.random.normal(0.15, 0.05, (height, width))
        green_after = ndvi_after * 0.8 + np.random.normal(0.1, 0.05, (height, width))
        blue_after = np.random.normal(0.1, 0.05, (height, width))
        nir_after = ndvi_after * 1.2 + 0.1
        
        # Create data dictionary
        data = {
            'before': {
                'ndvi': ndvi_before,
                'red': red_before,
                'green': green_before,
                'blue': blue_before,
                'nir': nir_before
            },
            'after': {
                'ndvi': ndvi_after,
                'red': red_after,
                'green': green_after,
                'blue': blue_after,
                'nir': nir_after
            },
            'metadata': {
                'height': height,
                'width': width,
                'creation_date': datetime.now().isoformat(),
                'deforestation_patches': deforestation_patches if add_deforestation else [],
                'has_deforestation': add_deforestation
            }
        }
        
        # Save as numpy compressed file
        np.savez_compressed(output_path, **data)
        
        # Also save metadata as JSON
        metadata_path = output_path.replace('.npz', '_metadata.json')
        with open(metadata_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization in metadata
            meta_copy = data['metadata'].copy()
            meta_copy['deforestation_patches'] = deforestation_patches if add_deforestation else []
            json.dump(meta_copy, f, indent=2)
        
        print(f"Sample data saved: {output_path}")
        print(f"Metadata saved: {metadata_path}")
        print(f"Deforestation patches created: {len(deforestation_patches)}")
        
        return data
    
    def get_forest_cover_data(self, geometry, year=2020):
        """Get forest cover statistics for an area"""
        # Calculate area from geometry
        area_sq_degrees = geometry.area
        area_sq_km = area_sq_degrees * 111 * 111  # Rough conversion
        
        # Generate realistic statistics
        import random
        
        forest_percentage = random.uniform(65, 85)
        forest_area = area_sq_km * forest_percentage / 100
        
        return {
            'total_area_sqkm': round(area_sq_km, 2),
            'forest_area_sqkm': round(forest_area, 2),
            'forest_cover_percentage': round(forest_percentage, 1),
            'deforestation_area_sqkm': round(random.uniform(0, area_sq_km * 0.15), 2),
            'year': year,
            'num_forest_patches': random.randint(8, 25),
            'largest_patch_sqkm': round(random.uniform(forest_area * 0.1, forest_area * 0.3), 2),
            'data_source': 'synthetic_analysis'
        }
    
    def load_sample_data(self, filepath):
        """Load previously created sample data"""
        try:
            data = np.load(filepath, allow_pickle=True)
            print(f"Data loaded from: {filepath}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None