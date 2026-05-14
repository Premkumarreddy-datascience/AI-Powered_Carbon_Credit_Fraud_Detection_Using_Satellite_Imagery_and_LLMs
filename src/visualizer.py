import ee
import folium
import numpy as np
import requests
import io
import matplotlib.pyplot as plt


class SatelliteVisualizer:

    def __init__(self):
        print("SatelliteVisualizer Ready")

    # Internal EE Layer Helper
    @staticmethod
    def _add_ee_layer(map_object, ee_image, vis_params, name):
        map_id = ee.Image(ee_image).getMapId(vis_params)

        folium.raster_layers.TileLayer(
            tiles=map_id["tile_fetcher"].url_format,
            attr="Google Earth Engine",
            name=name,
            overlay=True,
            control=True,
            show=False
        ).add_to(map_object)

    # Interactive Dashboard
    def create_dashboard(
        self,
        geometry,
        ndvi_before, ndvi_after, ndvi_change,
        evi_before, evi_after, evi_change,
        ndwi_before, ndwi_after, ndwi_change,
        forest_before, forest_after
    ):

        center = geometry.centroid().coordinates().getInfo()[::-1]
        m = folium.Map(location=center, zoom_start=10)

        vis = {"min": -1, "max": 1, "palette": ["red", "yellow", "green"]}
        change_vis = {"min": -0.5, "max": 0.5, "palette": ["red", "white", "green"]}
        forest_vis = {"min":0, "max":1, "palette":["white", "darkgreen"]}
        forest_change_vis = {"min": -1, "max": 1, "palette": ["red", "white", "darkgreen"]}

        # AOI boundary styling
        aoi_geojson = geometry.getInfo()

        folium.GeoJson(
            data=aoi_geojson,
            name="AOI Boundary",
            style_function=lambda x:{
                "color": "black",
                "weight": 4,
                "fill": False
            }
        ).add_to(m)
        
        # NDVI
        self._add_ee_layer(m, ndvi_before, vis, "NDVI Before")
        self._add_ee_layer(m, ndvi_after, vis, "NDVI After")
        self._add_ee_layer(m, ndvi_change, change_vis, "NDVI Change")

        # EVI
        self._add_ee_layer(m, evi_before, vis, "EVI Before")
        self._add_ee_layer(m, evi_after, vis, "EVI After")
        self._add_ee_layer(m, evi_change, change_vis, "EVI Change")

        # NDWI
        self._add_ee_layer(m, ndwi_before, vis, "NDWI Before")
        self._add_ee_layer(m, ndwi_after, vis, "NDWI After")
        self._add_ee_layer(m, ndwi_change, change_vis, "NDWI Change")

        # Forest Masks
        self._add_ee_layer(m, forest_before, forest_vis, "Forest Before")
        self._add_ee_layer(m, forest_after, forest_vis, "Forest After")
        
        # Forest Change
        forest_change = forest_after.subtract(forest_before).rename("Forest_Change")
        self._add_ee_layer(m, forest_change, forest_change_vis, "Forest Change")

        folium.LayerControl(collapsed=False).add_to(m)

        return m

    # Save Interactive Map
    @staticmethod
    def save_map(map_object, output_path):
        map_object.save(str(output_path))

    # EE → NumPy Conversion
    @staticmethod
    def ee_to_numpy(image, geometry, band, scale=100):
        url = image.getDownloadURL({
            "scale": scale,
            "region": geometry,
            "format": "NPY"
        })

        response = requests.get(url)
        arr = np.load(io.BytesIO(response.content))

        return arr[band].astype(float)

    # Save Static Image
    @staticmethod
    def save_static_image(array, title, output_path):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(array, cmap="RdYlGn")
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Export All Index Images
    def export_all_static_maps(
        self,
        geometry,
        output_folder,
        ndvi_before, ndvi_after, ndvi_change,
        evi_before, evi_after, evi_change,
        ndwi_before, ndwi_after, ndwi_change
    ):

        index_list = [
            ("ndvi_before", ndvi_before, "NDVI"),
            ("ndvi_after", ndvi_after, "NDVI"),
            ("ndvi_change", ndvi_change, "NDVI_Change"),
            ("evi_before", evi_before, "EVI"),
            ("evi_after", evi_after, "EVI"),
            ("evi_change", evi_change, "EVI_Change"),
            ("ndwi_before", ndwi_before, "NDWI"),
            ("ndwi_after", ndwi_after, "NDWI"),
            ("ndwi_change", ndwi_change, "NDWI_Change"),
        ]

        for name, image, band in index_list:
            array = self.ee_to_numpy(image, geometry, band)
            self.save_static_image(
                array,
                title=name.replace("_", " ").upper(),
                output_path=output_folder / f"{name}.png"
            )

        print("All static images exported")