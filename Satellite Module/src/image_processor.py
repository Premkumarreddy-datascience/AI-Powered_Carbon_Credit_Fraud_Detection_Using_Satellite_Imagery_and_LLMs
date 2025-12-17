import numpy as np
from scipy import ndimage
import cv2
from skimage import exposure, filters
import geopandas as gpd
from shapely.geometry import Polygon
import os

class ImageProcessor:
    """Process and analyze satellite imagery"""
    
    @staticmethod
    def calculate_ndvi(red_band, nir_band):
        """Calculate Normalized Difference Vegetation Index"""
        ndvi = (nir_band.astype(float) - red_band.astype(float)) / \
               (nir_band.astype(float) + red_band.astype(float) + 1e-10)
        ndvi = np.clip(ndvi, -1, 1)
        return ndvi
    
    @staticmethod
    def calculate_evi(blue_band, red_band, nir_band):
        """Calculate Enhanced Vegetation Index"""
        L = 1  # Canopy background adjustment
        C1 = 6  # Coefficient for aerosol resistance
        C2 = 7.5  # Coefficient for aerosol resistance
        G = 2.5  # Gain factor
        
        evi = G * ((nir_band - red_band) / (nir_band + C1 * red_band - C2 * blue_band + L + 1e-10))
        evi = np.clip(evi, -1, 1)
        return evi
    
    @staticmethod
    def calculate_ndwi(green_band, nir_band):
        """Calculate Normalized Difference Water Index"""
        ndwi = (green_band.astype(float) - nir_band.astype(float)) / \
               (green_band.astype(float) + nir_band.astype(float) + 1e-10)
        ndwi = np.clip(ndwi, -1, 1)
        return ndwi
    
    @staticmethod
    def detect_vegetation(ndvi, threshold=0.3):
        """Detect vegetation areas based on NDVI threshold"""
        vegetation_mask = ndvi > threshold
        return vegetation_mask
    
    @staticmethod
    def segment_forest(ndvi, evi, ndwi, min_ndvi=0.4, min_evi=0.2):
        """Segment forest areas using multiple indices"""
        # Forest is high vegetation but not water
        forest_mask = (ndvi > min_ndvi) & (evi > min_evi) & (ndwi < 0.2)
        
        # Apply morphological operations to clean mask
        forest_mask = ndimage.binary_opening(forest_mask, structure=np.ones((3,3)))
        forest_mask = ndimage.binary_closing(forest_mask, structure=np.ones((5,5)))
        
        return forest_mask
    
    @staticmethod
    def calculate_forest_metrics(forest_mask, pixel_area=900):  # 30m x 30m = 900 sqm
        """Calculate forest area and fragmentation metrics"""
        # Label connected components
        labeled_array, num_features = ndimage.label(forest_mask)
        
        # Calculate areas of each patch
        patch_sizes = ndimage.sum(forest_mask, labeled_array, range(num_features + 1))
        
        # Total forest area
        total_area = np.sum(forest_mask) * pixel_area / 1e6  # Convert to sq km
        
        # Fragmentation metrics
        if num_features > 0:
            mean_patch_size = np.mean(patch_sizes[1:]) * pixel_area / 1e4  # hectares
            largest_patch_index = np.argmax(patch_sizes[1:]) + 1
            largest_patch_area = patch_sizes[largest_patch_index] * pixel_area / 1e6
        else:
            mean_patch_size = 0
            largest_patch_area = 0
        
        metrics = {
            'total_forest_area_sqkm': total_area,
            'num_forest_patches': num_features,
            'mean_patch_size_hectares': mean_patch_size,
            'largest_patch_area_sqkm': largest_patch_area,
            'forest_cover_percentage': (np.sum(forest_mask) / forest_mask.size) * 100
        }
        
        return metrics, labeled_array
    
    @staticmethod
    def detect_deforestation(ndvi_before, ndvi_after, change_threshold=0.15):
        """Detect deforestation between two time periods"""
        ndvi_change = ndvi_after - ndvi_before
        
        # Significant negative change indicates deforestation
        deforestation_mask = ndvi_change < -change_threshold
        
        # Filter small changes
        deforestation_mask = ndimage.binary_opening(deforestation_mask, structure=np.ones((3,3)))
        
        return deforestation_mask, ndvi_change
    
    @staticmethod
    def create_false_color_composite(red, green, blue, nir):
        """Create false color composite for visualization"""
        # SWIR, NIR, Red composite for vegetation analysis
        false_color = np.stack([nir, red, green], axis=0)
        false_color = exposure.rescale_intensity(false_color, out_range=(0, 255))
        return false_color.astype(np.uint8)
    
    @staticmethod
    def detect_water(ndwi, threshold=0.2):
        """Detect water bodies using NDWI"""
        water_mask = ndwi > threshold
        return water_mask