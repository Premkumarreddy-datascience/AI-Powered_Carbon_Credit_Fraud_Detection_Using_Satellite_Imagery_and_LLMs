import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box
import warnings
warnings.filterwarnings('ignore')

# Try to import optional packages with fallbacks
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: plotly not installed. Some interactive features limited.")

try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("Note: folium not installed. Interactive maps limited.")

try:
    import rasterio
    from rasterio.plot import show
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Note: rasterio not installed. GeoTIFF display limited.")

class SatelliteVisualizer:
    """Visualize satellite data and analysis results"""
    
    @staticmethod
    def plot_rgb_composite(rgb_array, title="RGB Composite", figsize=(12, 10)):
        """Plot RGB composite image"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Handle different array shapes
        if rgb_array.ndim == 3:
            if rgb_array.shape[0] == 3 or rgb_array.shape[0] == 4:  # CHW format
                rgb_normalized = rgb_array.transpose(1, 2, 0).astype(float)
            else:  # HWC format
                rgb_normalized = rgb_array.astype(float)
        else:
            raise ValueError(f"Expected 3D array, got {rgb_array.ndim}D")
        
        # Normalize to 0-1 range for display
        rgb_normalized = (rgb_normalized - rgb_normalized.min()) / \
                        (rgb_normalized.max() - rgb_normalized.min() + 1e-10)
        
        ax.imshow(rgb_normalized)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_vegetation_indices(ndvi, evi, ndwi, titles=None):
        """Plot vegetation indices side by side"""
        if titles is None:
            titles = ['NDVI', 'EVI', 'NDWI']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Custom colormap for vegetation
        veg_cmap = LinearSegmentedColormap.from_list(
            'veg_cmap', ['brown', 'yellow', 'green', 'darkgreen']
        )
        
        # Plot NDVI
        im1 = axes[0].imshow(ndvi, cmap=veg_cmap, vmin=-1, vmax=1)
        axes[0].set_title(titles[0], fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Plot EVI
        im2 = axes[1].imshow(evi, cmap=veg_cmap, vmin=-1, vmax=1)
        axes[1].set_title(titles[1], fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Plot NDWI
        water_cmap = LinearSegmentedColormap.from_list(
            'water_cmap', ['brown', 'white', 'blue']
        )
        im3 = axes[2].imshow(ndwi, cmap=water_cmap, vmin=-1, vmax=1)
        axes[2].set_title(titles[2], fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_change_detection(before_rgb, after_rgb, change_mask, 
                             deforestation_mask=None, figsize=(15, 10)):
        """Visualize change detection results"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Before image
        if before_rgb.ndim == 3 and before_rgb.shape[0] in [3, 4]:
            axes[0,0].imshow(before_rgb.transpose(1, 2, 0))
        else:
            axes[0,0].imshow(before_rgb)
        axes[0,0].set_title('Before Period', fontsize=12, fontweight='bold')
        axes[0,0].axis('off')
        
        # After image
        if after_rgb.ndim == 3 and after_rgb.shape[0] in [3, 4]:
            axes[0,1].imshow(after_rgb.transpose(1, 2, 0))
        else:
            axes[0,1].imshow(after_rgb)
        axes[0,1].set_title('After Period', fontsize=12, fontweight='bold')
        axes[0,1].axis('off')
        
        # Change mask
        axes[1,0].imshow(change_mask, cmap='Reds', alpha=0.7)
        axes[1,0].set_title('Change Detection Mask', fontsize=12, fontweight='bold')
        axes[1,0].axis('off')
        
        # Deforestation mask (if provided)
        if deforestation_mask is not None:
            axes[1,1].imshow(deforestation_mask, cmap='OrRd', alpha=0.7)
            axes[1,1].set_title('Deforestation Areas', fontsize=12, fontweight='bold')
        else:
            axes[1,1].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_forest_metrics(forest_mask, metrics, labeled_patches=None):
        """Visualize forest metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Forest mask
        axes[0].imshow(forest_mask, cmap='Greens')
        axes[0].set_title('Forest Cover', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Add forest patches if available
        if labeled_patches is not None:
            axes[0].contour(labeled_patches > 0, colors='red', linewidths=0.5, alpha=0.5)
        
        # Metrics bar plot
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # Convert to appropriate scale for visualization
        scaled_values = []
        for name, value in zip(metric_names, metric_values):
            if 'area' in name.lower() or 'percentage' in name.lower():
                scaled_values.append(value)
            else:
                scaled_values.append(value)
        
        axes[1].barh(range(len(metric_names)), scaled_values)
        axes[1].set_yticks(range(len(metric_names)))
        axes[1].set_yticklabels(metric_names)
        axes[1].set_xlabel('Value')
        axes[1].set_title('Forest Metrics', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_interactive_map(lat, lon, zoom=10):
        """Create interactive Folium map"""
        if not FOLIUM_AVAILABLE:
            print("Folium not available. Install with: pip install folium")
            return None
        
        m = folium.Map(location=[lat, lon], zoom_start=zoom)
        
        # Add tile layers
        folium.TileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add draw tools
        draw = plugins.Draw(
            export=False,
            position='topleft',
            draw_options={
                'polyline': False,
                'rectangle': True,
                'polygon': True,
                'circle': False,
                'marker': False,
                'circlemarker': False,
            }
        )
        draw.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add measure tool
        plugins.MeasureControl(position='topright').add_to(m)
        
        return m
    
    @staticmethod
    def plot_fraud_risk_heatmap(risk_scores, coordinates=None, figsize=(10, 8)):
        """Plot fraud risk heatmap"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(risk_scores, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_title('Fraud Risk Heatmap', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Risk Score (0=Low, 1=High)', fontsize=12)
        
        # Add risk categories
        risk_categories = {
            'Low': (0.0, 0.3),
            'Medium': (0.3, 0.7),
            'High': (0.7, 1.0)
        }
        
        # Add legend
        patches = []
        for category, (low, high) in risk_categories.items():
            color = im.cmap((low + high) / 2)
            patches.append(mpatches.Patch(color=color, label=f'{category} Risk'))
        
        ax.legend(handles=patches, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_animated_timeseries(time_series_data, timestamps):
        """Create animated time series visualization"""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: pip install plotly")
            # Create static version instead
            fig, axes = plt.subplots(1, len(time_series_data), figsize=(5*len(time_series_data), 5))
            if len(time_series_data) == 1:
                axes = [axes]
            
            for i, (data, timestamp) in enumerate(zip(time_series_data, timestamps)):
                axes[i].imshow(data, cmap='Greens', vmin=data.min(), vmax=data.max())
                axes[i].set_title(str(timestamp))
                axes[i].axis('off')
            
            plt.tight_layout()
            return fig
        
        fig = go.Figure()
        
        # Create frames for animation
        frames = []
        for i, (data, timestamp) in enumerate(zip(time_series_data, timestamps)):
            frame = go.Frame(
                data=[go.Heatmap(z=data, colorscale='Greens')],
                name=str(timestamp),
                traces=[0]
            )
            frames.append(frame)
        
        # Initial frame
        fig.add_trace(
            go.Heatmap(
                z=time_series_data[0],
                colorscale='Greens',
                colorbar=dict(title='NDVI')
            )
        )
        
        # Add frames and animation controls
        fig.frames = frames
        
        # Animation settings
        fig.update_layout(
            title='Vegetation Time Series Animation',
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[str(t)], 
                                {'frame': {'duration': 300, 'redraw': True},
                                 'mode': 'immediate',
                                 'transition': {'duration': 300}}],
                        'label': str(t),
                        'method': 'animate'
                    }
                    for t in timestamps
                ],
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 16},
                    'prefix': 'Date: ',
                    'visible': True,
                    'xanchor': 'right'
                }
            }]
        )
        
        return fig
    
    @staticmethod
    def plot_deforestation_analysis(before_ndvi, after_ndvi, deforestation_mask, 
                                   metrics, figsize=(15, 12)):
        """Comprehensive deforestation analysis visualization"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Before NDVI
        im1 = axes[0,0].imshow(before_ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0,0].set_title('NDVI - Before', fontsize=12, fontweight='bold')
        axes[0,0].axis('off')
        plt.colorbar(im1, ax=axes[0,0], fraction=0.046, pad=0.04)
        
        # After NDVI
        im2 = axes[0,1].imshow(after_ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0,1].set_title('NDVI - After', fontsize=12, fontweight='bold')
        axes[0,1].axis('off')
        plt.colorbar(im2, ax=axes[0,1], fraction=0.046, pad=0.04)
        
        # NDVI Change
        ndvi_change = after_ndvi - before_ndvi
        im3 = axes[0,2].imshow(ndvi_change, cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[0,2].set_title('NDVI Change', fontsize=12, fontweight='bold')
        axes[0,2].axis('off')
        plt.colorbar(im3, ax=axes[0,2], fraction=0.046, pad=0.04)
        
        # Deforestation mask
        axes[1,0].imshow(deforestation_mask, cmap='Reds')
        axes[1,0].set_title('Deforestation Areas', fontsize=12, fontweight='bold')
        axes[1,0].axis('off')
        
        # Statistics
        axes[1,1].axis('off')
        stats_text = f"""
        Deforestation Analysis:
        
        Total Area: {metrics.get('total_area', 0):.2f} sq km
        Forest Loss: {metrics.get('forest_loss', 0):.2f} sq km
        Loss Percentage: {metrics.get('loss_percentage', 0):.1f}%
        Patches Detected: {metrics.get('num_patches', 0)}
        Average Patch Size: {metrics.get('mean_patch_size', 0):.2f} ha
        """
        axes[1,1].text(0.1, 0.5, stats_text, fontsize=11, 
                       verticalalignment='center', transform=axes[1,1].transAxes)
        
        # Risk score if available
        if 'fraud_risk' in metrics:
            axes[1,2].axis('off')
            risk_text = f"""
            Fraud Risk Assessment:
            
            Overall Risk: {metrics['fraud_risk']:.3f}
            Risk Level: {metrics.get('risk_level', 'N/A')}
            
            Indicators:
            - Abrupt Change: {'✓' if metrics.get('abrupt_change', False) else '✗'}
            - Grid Pattern: {'✓' if metrics.get('grid_pattern', False) else '✗'}
            - Boundary Regularity: {'✓' if metrics.get('suspicious_boundary', False) else '✗'}
            """
            axes[1,2].text(0.1, 0.5, risk_text, fontsize=11,
                          verticalalignment='center', transform=axes[1,2].transAxes)
        else:
            axes[1,2].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def display_geotiff(geotiff_path):
        """Display GeoTIFF file if rasterio is available"""
        if not RASTERIO_AVAILABLE:
            print("Cannot display GeoTIFF: rasterio not installed")
            print("Install with: pip install rasterio")
            return None
        
        try:
            with rasterio.open(geotiff_path) as src:
                fig, ax = plt.subplots(figsize=(12, 10))
                show(src, ax=ax)
                ax.set_title(f"GeoTIFF: {geotiff_path}", fontsize=14)
                plt.tight_layout()
                return fig
        except Exception as e:
            print(f"Error displaying GeoTIFF: {e}")
            return None
    
    @staticmethod
    def plot_comparison_grid(images, titles, ncols=3, figsize=(15, 10)):
        """Plot multiple images in a grid"""
        nrows = (len(images) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        # Flatten axes if only one row
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (img, title) in enumerate(zip(images, titles)):
            row = idx // ncols
            col = idx % ncols
            
            if img.ndim == 3 and img.shape[0] in [3, 4]:  # CHW format
                axes[row, col].imshow(img.transpose(1, 2, 0))
            else:
                axes[row, col].imshow(img, cmap='viridis')
            
            axes[row, col].set_title(title, fontsize=11)
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for idx in range(len(images), nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def save_figure(fig, filepath, dpi=150, bbox_inches='tight'):
        """Save figure to file"""
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Figure saved to: {filepath}")
    
    @staticmethod
    def plot_ndvi_histogram(ndvi_data, title="NDVI Distribution", figsize=(10, 6)):
        """Plot histogram of NDVI values"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Flatten NDVI data
        ndvi_flat = ndvi_data.flatten()
        
        # Remove NaN values
        ndvi_flat = ndvi_flat[~np.isnan(ndvi_flat)]
        
        # Plot histogram
        ax.hist(ndvi_flat, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax.set_xlabel('NDVI Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(ndvi_flat)
        std_val = np.std(ndvi_flat)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5)
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5)
        
        ax.legend()
        plt.tight_layout()
        return fig