import ee
import numpy as np
from scipy import ndimage
import pandas as pd  # ensure pandas is imported

from skimage import feature, filters

from satellite_module.config.settings import (
    SATELLITE_CONFIG,
    ANALYSIS_CONFIG,
    FEATURE_CONFIG
)


class ImageProcessor:
    def __init__(self):
        # Band configuration
        self.bands = SATELLITE_CONFIG["sentinel-2"]["bands"]
        self.all_bands = list(self.bands.values())

        # Safe configuration loading with defaults
        self.ndvi_threshold = ANALYSIS_CONFIG.get("ndvi_threshold", 0.3)
        self.change_threshold = ANALYSIS_CONFIG.get("change_threshold", 0.15)

        self.forest_ndvi_threshold = ANALYSIS_CONFIG.get("forest_ndvi_threshold", 0.4)
        self.forest_evi_threshold = ANALYSIS_CONFIG.get("forest_evi_threshold", 0.2)
        self.forest_ndwi_threshold = ANALYSIS_CONFIG.get("forest_ndwi_threshold", 0.2)

        self.scale = ANALYSIS_CONFIG.get("scale", 100)
        
        # Feature config
        self.feature_config = FEATURE_CONFIG

        print("ImageProcessor initialized with ML feature extraction")

    
    def compute_ndvi(self, image):
        return image.normalizedDifference(
            [self.bands["nir"], self.bands["red"]]
        ).rename("NDVI")

    def compute_ndwi(self, image):
        return image.normalizedDifference(
            [self.bands["green"], self.bands["nir"]]
        ).rename("NDWI")

    def compute_evi(self, image):
        nir = image.select(self.bands["nir"])
        red = image.select(self.bands["red"])
        blue = image.select(self.bands["blue"])

        return image.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
            {
                "NIR": nir,
                "RED": red,
                "BLUE": blue
            }
        ).rename("EVI")

    def compute_change(self, image_before, image_after, band_name):
        return image_after.subtract(image_before).rename(f"{band_name}_Change")

    def compute_statistics(self, image, band_name, geometry):
        reducer = ee.Reducer.mean().combine(
            ee.Reducer.minMax(),
            sharedInputs=True
        )

        stats = image.reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=self.scale,
            bestEffort=True,
            maxPixels=1e7
        )

        result = stats.getInfo()

        return {
            "mean": round(result.get(f"{band_name}_mean", 0), 4),
            "min": round(result.get(f"{band_name}_min", 0), 4),
            "max": round(result.get(f"{band_name}_max", 0), 4),
        }

    def vegetation_mask(self, ndvi_image):
        return ndvi_image.gt(self.ndvi_threshold).rename("Vegetation")

    def forest_mask(self, ndvi, evi, ndwi):
        return (
            ndvi.gt(self.forest_ndvi_threshold)
            .And(evi.gt(self.forest_evi_threshold))
            .And(ndwi.lt(self.forest_ndwi_threshold))
        ).rename("Forest")

    def calculate_area_from_mask(self, mask_image, geometry):
        stats = mask_image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=self.scale,
            bestEffort=True,
            maxPixels=1e7
        )

        result = stats.getInfo()
        band_name = mask_image.bandNames().getInfo()[0]
        pixels = result.get(band_name, 0)

        if pixels is None:
            pixels = 0

        pixel_area_sqkm = (self.scale * self.scale) / 1e6
        area_sqkm = pixels * pixel_area_sqkm

        return {
            "pixels": int(pixels),
            "area_sqkm": round(area_sqkm, 3)
        }

    def calculate_percentage_loss(self, before_area, after_area):
        """Your existing method - kept unchanged"""
        if before_area == 0:
            return 0.0
        loss = before_area - after_area
        return round((loss / before_area) * 100, 2)

    #  NEW ADVANCED INDICES FOR ML 

    def compute_savi(self, image, L=0.5):
        """Soil Adjusted Vegetation Index"""
        nir = image.select(self.bands["nir"])
        red = image.select(self.bands["red"])
        
        savi = image.expression(
            '((NIR - RED) / (NIR + RED + L)) * (1 + L)',
            {'NIR': nir, 'RED': red, 'L': L}
        ).rename('SAVI')
        return savi

    def compute_msavi2(self, image):
        """Modified Soil Adjusted Vegetation Index 2"""
        nir = image.select(self.bands["nir"])
        red = image.select(self.bands["red"])
        
        msavi2 = image.expression(
            '(2 * NIR + 1 - sqrt((2 * NIR + 1)**2 - 8 * (NIR - RED))) / 2',
            {'NIR': nir, 'RED': red}
        ).rename('MSAVI2')
        return msavi2

    def compute_ndmi(self, image):
        """Normalized Difference Moisture Index"""
        nir = image.select(self.bands["nir"])
        swir1 = image.select(self.bands["swir1"])
        
        ndmi = image.normalizedDifference(
            [self.bands["nir"], self.bands["swir1"]]
            ).rename("NDMI")

        return ndmi

    def compute_nbr(self, image):
        """Normalized Burn Ratio"""
        nir = image.select(self.bands["nir"])
        swir2 = image.select(self.bands["swir2"])
        
        nbr = image.normalizedDifference(
            [self.bands["nir"], self.bands["swir2"]]
            ).rename("NBR")

        return nbr

    def compute_red_edge_indices(self, image):
        """Red edge indices for vegetation health"""
        re1 = image.select(self.bands["red_edge1"])
        re2 = image.select(self.bands["red_edge2"])
        re3 = image.select(self.bands["red_edge3"])
        nir = image.select(self.bands["nir"])
        
        # Red Edge NDVI
        re_ndvi = image.normalizedDifference(['B8', 'B5']).rename('RE_NDVI')
        
        # Red Edge Position
        rep = image.expression(
            '705 + 35 * ((0.5*(B4+B7) - B5) / (B6 - B5))',
            {'B4': image.select('B4'), 'B5': re1, 'B6': re2, 'B7': re3}
        ).rename('REP')
        
        return ee.Image([re_ndvi, rep])

    def compute_all_indices(self, image):
        """Compute all indices for ML feature extraction"""
        indices = ee.Image([
            self.compute_ndvi(image),
            self.compute_evi(image),
            self.compute_savi(image),
            self.compute_msavi2(image),
            self.compute_ndmi(image),
            self.compute_ndwi(image),
            self.compute_nbr(image)
        ])
        
        # Add red edge indices
        red_edge = self.compute_red_edge_indices(image)
        indices = indices.addBands(red_edge)
        
        return indices

    # ============= TEXTURE FEATURES FOR ML =============

    def extract_texture_features(self, image_array, window_size=5):
        """
        Extract GLCM texture features from numpy array
        For use with downloaded arrays, not EE objects
        """
        from skimage.feature import graycomatrix, graycoprops
        
        # Normalize to 0-255
        image_norm = ((image_array - image_array.min()) / 
                      (image_array.max() - image_array.min()) * 255).astype(np.uint8)
        
        # Compute GLCM
        glcm = graycomatrix(image_norm, [window_size], [0], 256, symmetric=True)
        
        features = {}
        for prop in self.feature_config["texture"]["properties"]:
            features[f"texture_{prop}"] = graycoprops(glcm, prop)[0, 0]
        
        return features

    # ============= FEATURE VECTOR CREATION =============

    def create_feature_vector(self, image, geometry):
        """
        Create comprehensive feature vector for ML models
        """
        # Compute all indices
        indices = self.compute_all_indices(image)
        
        # Add original bands
        combined = image.select(self.all_bands).addBands(indices)
        
        # Get all band names
        band_names = combined.bandNames().getInfo()
        
        # Compute statistics for each band
        feature_dict = {}
        
        for band in band_names:
            stats = self.compute_statistics(combined, band, geometry)
            for stat_name, value in stats.items():
                feature_dict[f"{band}_{stat_name}"] = value
        
        return feature_dict

    def batch_extract_features(self, image_list, geometry_list):
        """
        Extract features for multiple images/geometries
        Used for creating training datasets
        """
        all_features = []
        
        for i, (image, geometry) in enumerate(zip(image_list, geometry_list)):
            features = self.create_feature_vector(image, geometry)
            features['sample_id'] = i
            all_features.append(features)
        
        return all_features
    # ... (keep all existing imports and class definition)
# Add this new method anywhere inside the class:

    def get_mean_feature_vector(self, image, geometry, scale=100, num_points=2000):
 
    
        # Compute all advanced indices
        indices = self.compute_all_indices(image)
        combined = image.select(self.all_bands).addBands(indices)
        band_names = combined.bandNames().getInfo()   # should be 27 bands
        
        # Generate random points inside geometry
        points = ee.FeatureCollection.randomPoints(
            region=geometry,
            points=num_points,
            seed=42,
            maxError=1
        )
        
        # Sample the combined image at these points
        sampled = combined.sampleRegions(
            collection=points,
            scale=scale,
            geometries=False,
            tileScale=4
        )
        
        # Download the table (one request)
        data = sampled.getInfo()
        if 'features' not in data or len(data['features']) == 0:
            return {band: 0.0 for band in band_names}
        
        # Convert to pandas DataFrame
        rows = []
        for feat in data['features']:
            rows.append(feat['properties'])
        df = pd.DataFrame(rows)
        
        # Compute mean for each band
        feature_dict = {}
        for band in band_names:
            if band in df.columns:
                vals = df[band].dropna()
                feature_dict[band] = float(vals.mean()) if len(vals) > 0 else 0.0
            else:
                feature_dict[band] = 0.0
        
        return feature_dict
    def get_mean_feature_vector_for_model(self, image, geometry, feature_cols, scale=100, num_points=2000):
        
        # 1. Compute indices (9 bands)
        indices = self.compute_all_indices(image)
        # 2. Spectral bands (12 bands from self.all_bands)
        spectral = image.select(self.all_bands)
        combined = spectral.addBands(indices)
        
        # 3. Try to add auxiliary bands if they exist (normally they don't, but we try)
        aux_bands = ['AOT', 'WVP', 'SCL', 'TCI_R', 'TCI_G', 'TCI_B']
        existing_aux = [b for b in aux_bands if b in image.bandNames().getInfo()]
        if existing_aux:
            combined = combined.addBands(image.select(existing_aux))
        
        actual_bands = combined.bandNames().getInfo()
        
        # 4. Sample random points
        points = ee.FeatureCollection.randomPoints(
            region=geometry,
            points=num_points,
            seed=42,
            maxError=1
        )
        sampled = combined.sampleRegions(
            collection=points,
            scale=scale,
            geometries=False,
            tileScale=4
        )
        data = sampled.getInfo()
        
        if 'features' not in data or len(data['features']) == 0:
            return {col: 0.0 for col in feature_cols}
        
        rows = [feat['properties'] for feat in data['features']]
        df = pd.DataFrame(rows)
        
        # 5. Build result dictionary – map model column names to actual band names
        result = {}
        for col in feature_cols:
            # model columns are like 's2_B1', so strip 's2_' to get band name
            band = col.replace('s2_', '')
            if band in df.columns:
                vals = df[band].dropna()
                result[col] = float(vals.mean()) if len(vals) > 0 else 0.0
            else:
                result[col] = 0.0   # missing auxiliary bands become 0
        
        return result