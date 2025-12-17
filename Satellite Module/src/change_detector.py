import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy import ndimage, stats
import cv2
from skimage.feature import local_binary_pattern
from skimage.segmentation import watershed
import warnings
warnings.filterwarnings('ignore')

class ChangeDetector:
    """Detect suspicious changes and anomalies in satellite imagery"""
    
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.anomaly_detector = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
    
    def detect_abrupt_changes(self, time_series_data, sensitivity=2.0):
        """
        Detect abrupt changes in vegetation time series
        
        Parameters:
        -----------
        time_series_data : numpy array
            3D array of shape (time, height, width)
        sensitivity : float
            Sensitivity threshold for change detection
        
        Returns:
        --------
        change_mask : numpy array
            Binary mask of detected changes
        change_scores : numpy array
            Change magnitude scores
        """
        n_times, height, width = time_series_data.shape
        
        # Calculate mean and std for each pixel over time
        mean_series = np.mean(time_series_data, axis=0)
        std_series = np.std(time_series_data, axis=0)
        
        # Detect pixels with significant deviations
        z_scores = np.abs((time_series_data[-1] - mean_series) / (std_series + 1e-10))
        
        # Initial change mask
        change_mask = z_scores > sensitivity
        
        # Apply spatial consistency check
        change_mask = self._spatial_consistency_filter(change_mask)
        
        # Calculate change scores
        change_scores = z_scores * change_mask.astype(float)
        
        return change_mask, change_scores
    
    def _spatial_consistency_filter(self, binary_mask, min_size=5):
        """Filter out small, isolated changes"""
        # Label connected components
        labeled_mask, num_features = ndimage.label(binary_mask)
        
        # Remove small components
        component_sizes = ndimage.sum(binary_mask, labeled_mask, range(num_features + 1))
        small_components = component_sizes < min_size
        
        # Create filtered mask
        filtered_mask = binary_mask.copy()
        for label in np.where(small_components)[0]:
            if label > 0:  # Skip background
                filtered_mask[labeled_mask == label] = 0
        
        return filtered_mask
    
    def detect_anomalous_patterns(self, features):
        """
        Detect anomalous patterns using Isolation Forest
        
        Parameters:
        -----------
        features : numpy array
            Feature matrix of shape (n_samples, n_features)
            
        Returns:
        --------
        anomaly_scores : numpy array
            Anomaly scores (-1 for anomalies, 1 for normal)
        """
        # Fit and predict anomalies
        anomaly_scores = self.anomaly_detector.fit_predict(features)
        
        return anomaly_scores
    
    def detect_boundary_changes(self, forest_mask_before, forest_mask_after):
        """Detect suspicious boundary changes (possible fraud indicator)"""
        # Calculate boundary pixels
        from scipy.ndimage import binary_dilation, binary_erosion
        
        # Get boundaries
        boundary_before = binary_dilation(forest_mask_before) ^ forest_mask_before
        boundary_after = binary_dilation(forest_mask_after) ^ forest_mask_after
        
        # Detect significant boundary changes
        boundary_change = boundary_after.astype(int) - boundary_before.astype(int)
        
        # Expansion (positive) or contraction (negative)
        expansion = boundary_change > 0
        contraction = boundary_change < 0
        
        # Calculate boundary irregularity
        before_irregularity = self._calculate_boundary_irregularity(forest_mask_before)
        after_irregularity = self._calculate_boundary_irregularity(forest_mask_after)
        
        # Suspicious if boundary becomes too regular (possible manipulation)
        regularity_change = after_irregularity / (before_irregularity + 1e-10)
        
        results = {
            'expansion_area': np.sum(expansion),
            'contraction_area': np.sum(contraction),
            'boundary_irregularity_before': before_irregularity,
            'boundary_irregularity_after': after_irregularity,
            'regularity_change_ratio': regularity_change,
            'suspicious_regularization': regularity_change < 0.7  # 30% more regular
        }
        
        return results
    
    def _calculate_boundary_irregularity(self, binary_mask):
        """Calculate boundary irregularity index"""
        from skimage.measure import perimeter
        
        # Calculate perimeter and area
        perimeter_length = perimeter(binary_mask)
        area = np.sum(binary_mask)
        
        # Circularity index (1 = perfect circle, lower = more irregular)
        if area > 0:
            circularity = (4 * np.pi * area) / (perimeter_length ** 2)
            irregularity = 1 / circularity
        else:
            irregularity = 0
            
        return irregularity
    
    def detect_grid_patterns(self, deforestation_mask):
        """
        Detect grid-like deforestation patterns (indicative of planned clearing)
        """
        # Use Hough transform to detect lines
        edges = cv2.Canny((deforestation_mask * 255).astype(np.uint8), 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=50, 
            minLineLength=20, 
            maxLineGap=10
        )
        
        if lines is not None:
            # Calculate angle distribution
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            # Check for regular patterns (90-degree angles common in grids)
            angles = np.array(angles) % 90
            regular_pattern_score = np.sum(np.abs(angles) < 10) / len(angles)
        else:
            regular_pattern_score = 0
        
        return regular_pattern_score
    
    def calculate_fraud_risk_score(self, metrics_dict):
        """
        Calculate overall fraud risk score based on multiple metrics
        """
        risk_factors = []
        
        # 1. Abrupt change risk
        if 'abrupt_change_magnitude' in metrics_dict:
            risk_factors.append(min(metrics_dict['abrupt_change_magnitude'] * 10, 1))
        
        # 2. Boundary regularization risk
        if 'suspicious_regularization' in metrics_dict:
            risk_factors.append(0.8 if metrics_dict['suspicious_regularization'] else 0.2)
        
        # 3. Grid pattern risk
        if 'grid_pattern_score' in metrics_dict:
            risk_factors.append(metrics_dict['grid_pattern_score'])
        
        # 4. Anomaly detection risk
        if 'anomaly_score' in metrics_dict:
            risk_factors.append(1 - metrics_dict['anomaly_score'])  # Inverted
        
        # Calculate weighted risk score
        if risk_factors:
            weights = [0.3, 0.25, 0.25, 0.2]  # Adjust based on importance
            weighted_risk = sum(w * r for w, r in zip(weights, risk_factors))
        else:
            weighted_risk = 0
        
        return min(weighted_risk, 1.0)  # Cap at 1.0
    
    def detect_seasonal_anomalies(self, monthly_ndvi, month):
        """
        Detect anomalies compared to seasonal patterns
        month: 1-12 representing January-December
        """
        # For demo, return random anomalies
        np.random.seed(month)
        height, width = monthly_ndvi.shape
        anomaly_mask = np.random.random((height, width)) > 0.95
        
        return anomaly_mask