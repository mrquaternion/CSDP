# carbonpipeline/core.py
import json
import os
import glob
from pathlib import Path
import shutil

import pandas as pd
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})

from .Geometry.geometry import Geometry
from .config import CarbonPipelineConfig
from .Processing.processor import DataProcessor
from .downloader import DataDownloader
from .dataset import DatasetManager


class CarbonPipeline:
    """Main pipeline orchestrator for download and processing flows."""
    
    def __init__(self):
        self.config = CarbonPipelineConfig()
        self.processor = DataProcessor(self.config)
        self.downloader = DataDownloader(self.config)
        self.dataset_manager = DatasetManager(self.config)

    async def run_download_point(
        self,
        download_bbox: list[float],
        region_id: str,
        geometry: Geometry,
        start: str,
        end: str,
        predictors: list[str],
        era5_vars: list[str],
        gapfilling: bool,
        ameriflux_csv: str,
        manifest_path: str
    ) -> None:
        """
        Download ERA5 data for a single EC station time range.

        Note: download_bbox is always a [N, W, S, E] region.
        """
        start_adj = pd.to_datetime(start, errors="coerce")
        end_adj = pd.to_datetime(end, errors="coerce")
        if pd.isna(start_adj) or pd.isna(end_adj):
            raise ValueError(f"Invalid dates: start={start}, end={end}")

        # Check if the dates are within the time range
        self.processor.validate_ameriflux_time_range(ameriflux_csv, start, end)

        groups = self.processor.build_request_groups(start_adj, end_adj, False)
        unzip_dirs = await self.downloader.download_request_groups_async(groups, era5_vars, download_bbox, False, region_id)

        feature_entry = {
            "region_id": region_id,
            "start_date": start,
            "end_date": end,
            "geometry": geometry.geom_type.value,
            "unzip_sub_folders": unzip_dirs,
            "preds": predictors
        }

        manifest_path = Path(manifest_path)
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
            f.pop("gapfilling", None)
            f.pop("data_file", None)

        # Rebuild the object with desired key order:
        features = manifest.get("features", [])
        features.append(feature_entry)

        ordered_manifest = {
            "gapfilling": gapfilling,
            "data_file": ameriflux_csv,
            "features": features
        }

        with open(manifest_path, 'w') as fp:
            json.dump(ordered_manifest, fp, indent=2)

        print(f"Appended new entry to manifest at {manifest_path}")


    async def run_download_area(
        self,
        download_bbox: list[float],
        region_id: str,
        geometry: Geometry,
        start: str,
        end: str,
        predictors: list[str],
        era5_vars: list[str],
        region_bboxes: dict[str | int, list[float]],
        geometry_mode: str,
        aggregation_type: str,
        manifest_path: str
    ) -> None:
        """Download ERA5 data for an area or multiple region boxes."""
        start_adj = pd.to_datetime(start, errors="coerce")
        end_adj = pd.to_datetime(end, errors="coerce")
        if pd.isna(start_adj) or pd.isna(end_adj):
            raise ValueError(f"Invalid dates: start={start}, end={end}")

        groups = self.processor.build_request_groups(start_adj, end_adj, aggregation_type == "MONTHLY")
        unzip_dirs = await self.downloader.download_request_groups_async(groups, era5_vars, download_bbox, aggregation_type == "MONTHLY", region_id)

        feature_entry = {
            "region_id": region_id,
            "start_date": start,
            "end_date": end,
            "geometry": geometry.geom_type.value,
            "unzip_sub_folders": unzip_dirs,
            "preds": predictors,
            "rect_regions": region_bboxes,
        }

        manifest_path = Path(manifest_path)
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

        # Clean old per-feature keys
        for f in manifest.get("features", []):
            f.pop("processing_type", None)
            f.pop("aggregation_type", None)

        # Rebuild the object with desired key order
        features = manifest.get("features", [])
        features.append(feature_entry)

        ordered_manifest = {
            "processing_type": geometry_mode,
            "aggregation_type": aggregation_type,
            "features": features
        }

        with open(manifest_path, 'w') as fp:
            json.dump(ordered_manifest, fp, indent=2)

        print(f"Appended new entry to manifest at {manifest_path}")

    def run_area_process(
        self,
        merged_ds: xr.Dataset,
        predictors: list[str],
        start: str,
        end: str,
        rect_regions: dict[str | int, list[float]],
        output_name: str,
        geometry_mode: str,
        aggregation_type: str,
        delete_source_after_aggregation: bool | None = None,
    ) -> None:
        """Process downloaded ERA5 data for area/box modes."""
        print(f"Processing {output_name}...")
        merged_era5_ds = self.dataset_manager.apply_column_rename(merged_ds)

        # Handle CO2 data
        ds_co2 = self.dataset_manager.load_and_clean_co2_dataset()
        if ds_co2 is not None:
            print("➕ Adding CO2 column...")
            merged_era5_ds = self.dataset_manager.add_co2_column(merged_era5_ds, ds_co2)

        # Handle WTD data
        ds_wtd = self.dataset_manager.load_and_clean_wtd_dataset(start, end)
        if ds_wtd is not None:
            print("➕ Adding WTD column...")
            merged_era5_ds = self.dataset_manager.add_wtd_column(merged_era5_ds, ds_wtd)

        if geometry_mode == "Box":
            region_datasets = self.dataset_manager.filter_coordinates(ds=merged_era5_ds, regions=rect_regions)
        else:
            merged_era5_df = merged_era5_ds.to_dataframe().reset_index()
            merged_era5_df["region_id"] = list(rect_regions.keys())[0]
            merged_era5_df = (
                merged_era5_df
                .set_index(["region_id", "latitude", "longitude", "valid_time"])
                .sort_index()
            )
            region_datasets = [merged_era5_df.to_xarray()]

        # Conversion to AMF predictors and intelligent chunk writing
        index_columns = ["region_id", "latitude", "longitude", "valid_time"]
        tmp_dirs = self.dataset_manager.write_chunks(region_datasets, predictors, index_columns)

        # Reopen the chunks for each region and create the NetCDF files
        region_datasets_by_id = self.dataset_manager.concat_chunks(tmp_dirs)

        # Aggregation --> not available for global option because too much data --> not optimized with chunk loading
        resample_rules = {"DAILY": "1D", "MONTHLY": "1ME"}
        if aggregation_type in resample_rules.keys(): # AGGREGATION
            if delete_source_after_aggregation is None:
                while True:
                    user_input = input("\nDo you want to delete the original files after aggregation? (Y/n): ").strip()
                    if user_input.upper() == "Y":
                        delete_source = True
                        break
                    elif user_input.lower() == "n":
                        delete_source = False
                        break
                    else:
                        print("Invalid input: please enter 'Y' to delete them, or 'n' to keep them.")
            else:
                delete_source = delete_source_after_aggregation

            self.dataset_manager.aggregate_dataset(region_datasets_by_id, resample_rules, aggregation_type, output_name, delete_source)
        else: # NO AGGREGATION
            for idx, ds in region_datasets_by_id.items():
                name = "_".join([output_name, idx])
                save_paths = self.dataset_manager.save_netcdf_daily(ds, name)
                for save_path in save_paths:
                    print(f"✅ File saved to {save_path}")

    def run_point_process(
        self,
        ameriflux_csv: str,
        merged_ds: xr.Dataset,
        predictors: list[str],
        start: str,
        end: str,
        region_id: str,
        gapfilling: bool,
        output_name: str
    ) -> None:
        """Post-process downloaded data for a single point."""
        ameriflux_df = self.processor.load_and_filter_ameriflux_csv(ameriflux_csv, start, end)
        era5_ds = self.dataset_manager.apply_column_rename(merged_ds)

        # The ndarray `era5_values` must be equal length of the gapfill dataframe
        # Downloading
        era5_hourly_df = (
            era5_ds.to_dataframe()
               .droplevel("latitude")
               .droplevel("longitude")
               .groupby("valid_time")
               .mean(numeric_only=True)
        )

        if gapfilling:
            gapfill_df = self.dataset_manager.build_multiindex_dataframe(ameriflux_df, predictors)
            for pred in predictors:
                if pred in gapfill_df.columns.get_level_values("variable"):
                    era5_values = self.processor.convert_ameriflux_to_era5(era5_hourly_df, pred)
                    gapfill_df.loc[:, (pred, "ERA5")] = era5_values
            
            cand = ("timestamp", "AMF")
            if cand in gapfill_df.columns:
                ts = pd.to_datetime(gapfill_df.pop(cand), errors="coerce")
                gapfill_df.insert(0, "timestamp", ts)  # put it first as a plain column
                gapfill_df = gapfill_df.set_index("timestamp")  # make it the index
            gapfill_df = gapfill_df.drop(columns=["year", "month", "day", "time"])

            self.dataset_manager.save_csv(gapfill_df, output_name)
        else:
            era5_ds = era5_ds.drop_vars(["year_month", "lat", "lon"])

            output_name = f"{output_name}_{region_id}"
            save_paths = self.dataset_manager.save_netcdf_daily(era5_ds, output_name)
            for save_path in save_paths:
                print(f"✅ File saved to {save_path}")

    def load_features_from_manifest(self, path):
        """Load manifest file as a dict."""
        with open(path, "r") as fp:
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
    def setup_manifest_and_dirs(path, *dirs) -> None:
        """Reset manifest and directories used for downloads."""
        manifest_path = Path(path)
        if manifest_path.exists():
            manifest_path.unlink() # deletes the manifest at each run

        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
