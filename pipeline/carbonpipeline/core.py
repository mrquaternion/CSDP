# carbonpipeline/core.py
import json
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})

from .Geometry.geometry import Geometry
from .config import CarbonPipelineConfig
from .Processing.processor import DataProcessor
from .downloader import DataDownloader
from .dataset import DatasetManager
from .Processing.processing_utils import AGG_SCHEMA


class CarbonPipeline:
    """Main pipeline orchestrator for carbon data processing."""
    
    def __init__(self):
        self.config = CarbonPipelineConfig()
        self.processor = DataProcessor(self.config)
        self.downloader = DataDownloader(self.config)
        self.dataset_manager = DatasetManager(self.config)

    async def run_download(
        self,
        coords_to_download: list[float],
        region_id: str,
        geometry: Geometry,
        start: str,
        end: str,
        preds: list[str],
        vrs: list[str],
        regions_to_process: dict[str | int, list[float]],
        processing_type: str,
        aggregation_type: str
    ) -> None:
        """
        Downloads ERA5 datasets for a specified area and time range.
        """
        start_adj = pd.to_datetime(start, errors="coerce")
        end_adj = pd.to_datetime(end, errors="coerce")
        if pd.isna(start_adj) or pd.isna(end_adj):
            raise ValueError(f"Invalid dates: start={start}, end={end}")

        groups = self.processor.get_request_groups(start_adj, end_adj, aggregation_type == "MONTHLY")
        unzip_dirs = await self.downloader.download_groups_async(groups, vrs, coords_to_download, aggregation_type == "MONTHLY", region_id)

        feature_entry = {
            "region_id": region_id,
            "start_date": start,
            "end_date": end,
            "geometry": geometry.geom_type.value,
            "unzip_sub_folders": unzip_dirs,
            "preds": preds,
            "rect_regions": regions_to_process,
        }

        manifest_path = Path(self.config.OUTPUT_MANIFEST)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Load or init manifest
        if manifest_path.is_file():
            with open(manifest_path, "r") as fp:
                try:
                    manifest = json.load(fp)
                    if not isinstance(manifest, dict):
                        manifest = {}
                except json.JSONDecodeError:
                    manifest = {}
        else:
            manifest = {}

        # Clean old per-feature keys (optional)
        for f in manifest.get("features", []):
            f.pop("processing_type", None)
            f.pop("aggregation_type", None)

        # Rebuild the object with desired key order:
        features = manifest.get("features", [])
        features.append(feature_entry)

        ordered_manifest = {
            "processing_type": processing_type,
            "aggregation_type": aggregation_type,
            "features": features
        }

        with open(manifest_path, 'w') as fp:
            json.dump(ordered_manifest, fp, indent=2)

        print(f"Appended new entry to manifest at {manifest_path}", flush=True)

    def run_area_process(
        self,
        merged_ds: xr.Dataset,
        preds: list[str],
        start: str,
        end: str,
        rect_regions: dict[str | int, list[float]],
        output_name: str,
        processing_type: str,
        aggregation_type: str
    ) -> None:
        """Process area data from manifest."""
        print(f"\n🔧 Starting area process for {output_name}", flush=True)
        merged_ds = self.dataset_manager.apply_column_rename(merged_ds)
        print("✅ Columns renamed", flush=True)

        # Handle CO2 data
        ds_co2 = self.dataset_manager.load_and_clean_co2_dataset()
        if ds_co2 is not None:
            print("➕ Adding CO2 column...", flush=True)
            merged_ds = self.dataset_manager.add_co2_column(merged_ds, ds_co2)

        # Handle WTD data
        ds_wtd = self.dataset_manager.load_and_clean_wtd_dataset(start, end)
        if ds_wtd is not None:
            print("➕ Adding WTD column...", flush=True)
            merged_ds = self.dataset_manager.add_wtd_column(merged_ds, ds_wtd)

        # Filter the dataset for the regions
        print("📐 Filtering dataset by bounding boxes...", flush=True)
        all_dss = self.dataset_manager.filter_coordinates(ds=merged_ds, regions=rect_regions)

        # Conversion to AMF predictors and intelligent chunk writing
        index = ['region_id', 'latitude', 'longitude', 'valid_time']
        print("✍️ Writing dataset chunks...", flush=True)
        tmp_dirs = self.dataset_manager.write_chunks(all_dss, preds, index, len(rect_regions))

        # Reopen the chunks for each region and create the NetCDF files
        print("💥 Concatenating chunks...", flush=True)
        region_dsets = self.dataset_manager.concat_chunks(tmp_dirs)

        # Aggregation --> not available for global option because too much data --> not optimized with chunk loading
        resample_methods = {"DAILY": "1D", "MONTHLY": "1ME"}
        if aggregation_type in resample_methods.keys():
            print(f"📊 Performing {aggregation_type} aggregation...", flush=True)
            save_paths = self.dataset_manager.parallel_aggregation(
                region_dsets,
                aggregation_type,
                output_name,
                AGG_SCHEMA,
                resample_methods
            )
            print(f"✅ Aggregation done for {len(save_paths)} regions", flush=True)
        else:
            self.dataset_manager.write_aggregated_ds(region_dsets, output_name, "", False)


    def run_point_process(
        self,
        data_fp: str,
        preds: list[str],
        merged_ds: xr.Dataset,
        start: str,
        end: str,
        output_name: str
    ) -> None:
        """
        Post-processes downloaded data for a single point.
        """
        df = self.processor.load_and_filter_dataframe(data_fp, start, end)
        if df.empty:
            print("No missing data found in the specified time range. Nothing to do.", flush=True)
            return None

        # Handle CO2 data (similar to area processing)
        ds_co2 = self.dataset_manager.load_and_clean_co2_dataset()
        if ds_co2 is not None:
            print("Adding CO2 column...", flush=True)
            merged_ds = self.dataset_manager.add_co2_column(merged_ds, ds_co2)

        # Handle WTD data (similar to area processing)
        ds_wtd = self.dataset_manager.load_and_clean_wtd_dataset(start, end)
        if ds_wtd is not None:
            print("Adding WTD column...", flush=True)
            merged_ds = self.dataset_manager.add_wtd_column(merged_ds, ds_wtd)
                
        dfm = self.dataset_manager.apply_column_rename(merged_ds).to_dataframe()
        dfr = self.dataset_manager.build_multiindex_dataframe(df, preds)
        
        for pred in preds:
            if pred in dfr.columns.get_level_values('variable'):
                era5_values = self.processor.convert_ameriflux_to_era5(dfm, pred)
                dfr.loc[:, (pred, "ERA5")] = era5_values

        ts = dfr.pop(("timestamp", "AMF"))
        dfr.insert(0, "timestamp", ts.droplevel('source'))

        if dfr is not None:
            self.dataset_manager.save_output(dfr, output_name)

    def load_features_from_manifest(self):
        """Load manifest file"""
        with open(self.config.OUTPUT_MANIFEST, "r") as fp:
            content = json.load(fp)
        return content

    def open_nc_all(self, output_name: str) -> dict[str, xr.Dataset]:
        """
        Open all NetCDF files for the given output_name (one per region).
        Returns a dict {region_id: Dataset}.
        """
        pattern = str(Path(self.config.OUTPUT_PROCESSED_DIR) / f"{output_name}_*.nc")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files found for {output_name} in {self.config.OUTPUT_PROCESSED_DIR}")

        dsets = {}
        for f in files:
            region_id = Path(f).stem.split("_")[-1]  # ex: output_name_region_1 -> "1"
            dsets[region_id] = xr.open_dataset(f, decode_times=True).load()
        return dsets

    @staticmethod
    def setup_manifest_and_dirs(manifest, *dirs) -> None:
        """Setup directories by removing and recreating them."""
        manifest_path = Path(manifest)
        if manifest_path.exists():
            manifest_path.unlink() # deletes the manifest at each run

        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)