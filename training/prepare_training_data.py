import sys
import os
from pathlib import Path
from datetime import datetime
import time
import ee
import pandas as pd

RUN_TIMESTAMP = datetime.now().strftime("%d%m%Y_%H%M%S")

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ["PROJECT_ROOT"] = str(project_root)

print("=" * 60)
print("PATH CONFIGURATION")
print("=" * 60)
print(f"Project root: {project_root}")
print("=" * 60)

from satellite_module.config.settings import (
    TRAINING_DATA_DIR,
    GEE_PROJECT_ID
)

from satellite_module.src.data_downloader import SatelliteDataDownloader
from satellite_module.src.image_processor import ImageProcessor


AOI_LIST = [
    {"coords": (-65, -12, -55, -3), "category": "Stable_Forest"}
]


class TrainingDataPreparer:

    def __init__(self):
        print("\nInitializing Training Data Preparer")

        try:
            ee.Initialize(project=GEE_PROJECT_ID)
        except:
            ee.Initialize()

        print("Earth Engine initialized")

        self.downloader = SatelliteDataDownloader()
        self.processor = ImageProcessor()

        self.training_data_path = TRAINING_DATA_DIR / f"training_dataset_{RUN_TIMESTAMP}.csv"

    def extract_sentinel_features(self, gedi_df, aoi_info):
        coords = aoi_info["coords"]
        category = aoi_info["category"]

        geometry = ee.Geometry.Rectangle(coords)
        country, region = self.downloader.get_location_info(geometry)

        all_features = []

        for year in gedi_df["year"].unique():
            year_df = gedi_df[gedi_df["year"] == year]

            min_lon = year_df["longitude"].min() - 0.1
            max_lon = year_df["longitude"].max() + 0.1
            min_lat = year_df["latitude"].min() - 0.1
            max_lat = year_df["latitude"].max() + 0.1

            aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
            image = self.downloader.get_seasonal_image(aoi, year)
            indices = self.processor.compute_all_indices(image)
            image = image.addBands(indices)

            band_names = self.downloader.safe_get_info(image.bandNames())
            if band_names is None:
                continue

            points = ee.FeatureCollection([
                ee.Feature(
                    ee.Geometry.Point([row["longitude"], row["latitude"]]),
                    {
                        "biomass_mg_ha": row["biomass_mg_ha"],
                        "year": row["year"]
                    }
                )
                for _, row in year_df.iterrows()
            ])

            samples = image.sampleRegions(
                collection=points,
                scale=10,
                geometries=True,
                tileScale=4
            )

            info = self.downloader.safe_get_info(samples)
            if info is None:
                continue

            for feat in info["features"]:
                props = feat["properties"]
                coords = feat["geometry"]["coordinates"]

                feature_dict = {
                    "country": country,
                    "area_name": region,
                    "aoi_category": category,
                    "longitude": coords[0],
                    "latitude": coords[1],
                    "year": props.get("year"),
                    "biomass_mg_ha": props.get("biomass_mg_ha")
                }

                for band in band_names:
                    feature_dict[f"s2_{band}"] = props.get(band, None)

                all_features.append(feature_dict)

        return pd.DataFrame(all_features)

    def clean_dataset(self, df):
        if "biomass_mg_ha" not in df.columns:
            return pd.DataFrame()

        df = df.dropna(subset=["biomass_mg_ha"])
        df = df[(df["biomass_mg_ha"] > 0) & (df["biomass_mg_ha"] < 500)]

        ignore_cols = [
            "MSK_CLDPRB", "MSK_SNWPRB", "QA10", "QA20", "QA60",
            "MSK_CLASSI_OPAQUE", "MSK_CLASSI_CIRRUS", "MSK_CLASSI_SNOW_ICE"
        ]

        df = df.drop(
            columns=[c for c in df.columns if any(x in c for x in ignore_cols)],
            errors="ignore"
        )

        return df

    def run_pipeline(self):
        print("\nStarting dataset generation")

        if os.path.exists(self.training_data_path):
            os.remove(self.training_data_path)

        pd.DataFrame().to_csv(self.training_data_path, index=False)

        header_written = False
        total_samples = 0

        for i, aoi_info in enumerate(AOI_LIST, start=1):
            print("\n" + "="*70)
            print(f"AOI {i} / {len(AOI_LIST)}")
            print("AOI Coordinates:", aoi_info["coords"])

            geometry = ee.Geometry.Rectangle(aoi_info["coords"])
            country, region = self.downloader.get_location_info(geometry)

            print("Country:", country)
            print("Region:", region)
            print("-"*70)

            gedi_df = self.downloader.download_training_data_from_gedi(aoi_info["coords"])

            if gedi_df is None:
                print("No GEDI samples found")
                print("="*70)
                continue

            print("\nTotal GEDI samples collected from this AOI:", len(gedi_df))

            features_df = self.extract_sentinel_features(gedi_df, aoi_info)
            clean_df = self.clean_dataset(features_df)

            if clean_df.empty:
                print("No valid samples")
                print("="*70)
                continue

            clean_df.to_csv(
                self.training_data_path,
                mode="a",
                header=not header_written,
                index=False
            )

            header_written = True
            total_samples += len(clean_df)

            print("Samples added to dataset:", len(clean_df))
            print("Total dataset samples so far:", total_samples)
            print("="*70)

            time.sleep(2)


if __name__ == "__main__":
    preparer = TrainingDataPreparer()
    preparer.run_pipeline()