"""Dataset merge, augmentation, and export utilities for the pipeline."""
import glob
import os
import shutil
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr

from .Processing.constants import SHORTNAME_TO_FULLNAME
from .Processing.processor import DataProcessor
from .config import CarbonPipelineConfig
from .Processing.processing_utils import AGG_SCHEMA

class DatasetManager:
    """Manages dataset operations including merging, processing, and saving."""
    
    def __init__(self, config: CarbonPipelineConfig):
        self.config = config

    def merge_unzipped_netcdfs(self, dirs: list[str]) -> Union[xr.Dataset, None]:
        """Merge NetCDF files from a list of directories into one xarray Dataset."""
        netcdf_paths = [p for d in dirs for p in glob.glob(os.path.join(d, "*.nc"))]
        if not netcdf_paths:
            return None

        def _drop_unneeded_vars(ds: xr.Dataset) -> xr.Dataset:
            removable = [v for v in ["number", "expver"] if v in ds.variables]
            return ds.drop_vars(removable) if removable else ds

        return xr.open_mfdataset(
            sorted(netcdf_paths),
            engine="h5netcdf",
            combine="by_coords",
            preprocess=_drop_unneeded_vars,
            combine_attrs="override",
            chunks="auto",
        )

    def add_co2_column(self, ds_era5: xr.Dataset, ds_co2: xr.Dataset) -> xr.Dataset:
        """Add CO2 column aligned to the ERA5 grid and monthly time axis."""
        ds_co2_renamed = ds_co2.rename({"time": "valid_time", "lat": "latitude", "lon": "longitude"})
        ds_co2_renamed = self._add_year_month(ds_co2_renamed, "valid_time")
        ds_era5 = self._add_year_month(ds_era5, "valid_time")

        ds_co2_monthly = ds_co2_renamed.groupby("year_month").mean(dim="valid_time")
        co2_selected = ds_co2_monthly["xco2"].sel(year_month=ds_era5["year_month"])
        co2_selected = co2_selected.sel(
            latitude=ds_era5["latitude"],
            longitude=ds_era5["longitude"],
            method="nearest",
        )

        ds_era5["xco2"] = co2_selected
        return ds_era5

    def add_wtd_column(self, ds_era5: xr.Dataset, ds_wtd: xr.Dataset) -> xr.Dataset:
        """Add WTD column aligned to the ERA5 grid and monthly time axis."""

        # Remove unwanted columns
        ds_wtd = ds_wtd.drop_vars("spatial_ref")

        # Rename WTD indexes so it matches ERA5 indexes
        ds_wtd_renamed = ds_wtd.rename({"time": "valid_time", "y": "latitude", "x": "longitude"})

        # Add column year_month to both dataset
        ds_wtd_renamed = self._add_year_month(ds_wtd_renamed, "valid_time")
        ds_era5 = self._add_year_month(ds_era5, "valid_time")
        
        ds_wtd_monthly = ds_wtd_renamed.groupby("year_month").mean(dim="valid_time")
        wtd_selected = ds_wtd_monthly["wtd"].sel(year_month=ds_era5["year_month"])
        wtd_selected = wtd_selected.sel(
            latitude=ds_era5["latitude"],
            longitude=ds_era5["longitude"],
            method="nearest",
        )

        ds_era5["wtd"] = wtd_selected
        return ds_era5.drop_vars(["year_month"], errors="ignore")

    def _add_year_month(self, ds: xr.Dataset, time_coord: str) -> xr.Dataset:
        """Add year_month coordinate as datetime64[M] (truncated to month)."""
        return ds.assign_coords(year_month=ds[time_coord].dt.strftime("%Y-%m"))

    def load_and_clean_co2_dataset(self) -> xr.Dataset:
        """Load and clean CO2 dataset."""
        co2_files = glob.glob(os.path.join(self.config.CO2_DIR, "*.nc"))
        if not co2_files:
            return None
        co2_ds = xr.open_dataset(co2_files[0])
        co2_ds["xco2"] = co2_ds["xco2"].where(co2_ds["xco2"] < 1e10, np.nan)

        return co2_ds[["xco2"]]

    def load_and_clean_wtd_dataset(self, start: str, end: str) -> Union[xr.Dataset, None]:
        """Load and clean WTD dataset."""
        start_str = pd.to_datetime(start).strftime("%Y-%m")
        end_str = pd.to_datetime(end).strftime("%Y-%m")
        time_window = "_".join([start_str, end_str])
        wtd_full_path = os.path.join(self.config.WTD_DIR, time_window)

        wtd_files = glob.glob(os.path.join(wtd_full_path, "*.tif"))
        if not wtd_files:
            return None
        
        wtd_datasets = []
        for file_path in wtd_files:
            raster = rxr.open_rasterio(file_path, masked=True).squeeze("band", drop=True)
            scale_factor = int(np.ceil(self.config.ERA5_RES / self.config.WTD_RES))
            raster_coarse = raster.coarsen(x=scale_factor, y=scale_factor, boundary="trim").mean()
            wtd_ds = raster_coarse.to_dataset(name="wtd")
            # Extract date from filename and set as time coordinate
            date_str = os.path.basename(file_path).split("-")[2].split(".")[0]
            time_value = pd.to_datetime(date_str, format="%Y%m%d")
            wtd_ds = wtd_ds.expand_dims(time=[time_value])
            wtd_datasets.append(wtd_ds)
        
        return xr.concat(wtd_datasets, dim="time") if wtd_datasets else None

    def filter_coordinates(self, ds: xr.Dataset, regions: dict[str | int, list[float]]) -> list[xr.Dataset]:
        """Filter ERA5 dataset to region corner coordinates and remap to region bounds."""
        dataset_copy = ds.copy()

        lat_values = dataset_copy.coords["latitude"].values
        lon_values = dataset_copy.coords["longitude"].values

        region_datasets = []
        for region_id, (lat_max, lon_min, lat_min, lon_max) in regions.items():
            lat_max_grid = self._nearest_point(lat_max, lat_values)
            lon_max_grid = self._nearest_point(lon_max, lon_values)
            lat_min_grid = self._nearest_point(lat_min, lat_values, prev=lat_max_grid)
            lon_min_grid = self._nearest_point(lon_min, lon_values, prev=lon_max_grid)

            # Select corner coordinates from the ERA5 grid
            lats = list({lat_max_grid, lat_min_grid})
            lons = list({lon_max_grid, lon_min_grid})
            corner_points = dataset_copy.sel(latitude=lats, longitude=lons)

            # Map ERA5 grid coordinates back to the requested region bounds
            coord_mapping = {
                lat_max_grid: lat_max,
                lat_min_grid: lat_min,
                lon_max_grid: lon_max,
                lon_min_grid: lon_min
            }

            remapped_lats = np.array([
                coord_mapping.get(float(v), float(v)) for v in corner_points["latitude"].values
            ])
            remapped_lons = np.array([
                coord_mapping.get(float(v), float(v)) for v in corner_points["longitude"].values
            ])

            region_ds = corner_points.assign_coords(
                latitude=("latitude", remapped_lats),
                longitude=("longitude", remapped_lons),
            )
            region_ds = region_ds.expand_dims(region_id=[region_id])
            region_datasets.append(region_ds)

        return region_datasets

    @staticmethod
    def _nearest_point(target: float | int, candidate_points: np.ndarray, prev=None):
        """Return the closest candidate to the target, optionally excluding a value."""
        if prev is not None:
            filtered = candidate_points[candidate_points != prev]
            if filtered.size > 0:
                candidate_points = filtered
        return candidate_points[np.argmin(np.abs(candidate_points - target))]

    def apply_column_rename(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply column renaming to dataset."""
        rename_map = {
            k: v 
            for k, v in SHORTNAME_TO_FULLNAME.items() 
            if k in ds.data_vars
        }
        return ds.rename(rename_map)

    @staticmethod
    def build_multiindex_dataframe(dataframe: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
        """
        Restructures a DataFrame to have a MultiIndex for AmeriFlux vs ERA5 data.
        """
        ameriflux_cols = {p: f"AMF, {p}" for p in predictors if p in dataframe.columns}
        renamed_df = dataframe.rename(columns=ameriflux_cols)
        
        for predictor in predictors:
            era5_col_name = f"ERA5, {predictor}"
            if era5_col_name not in renamed_df.columns:
                 renamed_df[era5_col_name] = np.nan

        column_tuples = []
        for col in renamed_df.columns:
            if ", " in col:
                source, variable = col.split(", ", 1)
            else:
                source, variable = "AMF", col
            column_tuples.append((variable, source))

        renamed_df.columns = pd.MultiIndex.from_tuples(column_tuples, names=["variable", "source"])
        return renamed_df.sort_index(axis=1, level="variable")

    def write_chunks(self, region_datasets: list[xr.Dataset], predictors: list[str]) -> list[str]:
        """
        Write dataset in chunks.
        """
        tmp_root = "./outputs_tmp"
        tmp_directories = []
        shutil.rmtree(tmp_root, ignore_errors=True)
        os.makedirs(tmp_root, exist_ok=True)
        data_processor = DataProcessor(self.config)

        # Process in larger time chunks for efficiency
        for dataset in region_datasets:
            region_id = f"region_{dataset.coords['region_id'].values[0]}"
            tmp_dir = os.path.join(tmp_root, region_id)
            os.makedirs(tmp_dir, exist_ok=True)
            predictor_lookup: dict[str, xr.DataArray] = {}
            for predictor in predictors:
                predictor_da = data_processor.convert_dataset_to_era5(dataset, predictor)
                predictor_lookup[predictor] = predictor_da.astype(np.float32)

            output_path = os.path.join(tmp_dir, f"{region_id}.nc")
            chunk_ds = xr.Dataset(predictor_lookup)
            if "valid_time" in chunk_ds.dims:
                chunk_ds = chunk_ds.chunk({"valid_time": min(744, chunk_ds.sizes["valid_time"])})
            chunk_ds.to_netcdf(output_path, mode="w", format="NETCDF4", engine="h5netcdf")

            tmp_directories.append(tmp_dir)
        
        return tmp_directories

    @staticmethod
    def concat_chunks(tmp_dirs: list[str]) -> dict[str, xr.Dataset]:
        """Concatenate chunked NetCDFs per region."""
        region_datasets = {}
        for tmp_dir in tmp_dirs:
            file_paths = sorted(glob.glob(os.path.join(tmp_dir, "*.nc")))
            if not file_paths:
                print(f"No chunks found in {tmp_dir}, skipping.")
                continue

            datasets = [xr.open_dataset(path, engine="h5netcdf", chunks="auto") for path in file_paths]
            combined_ds = xr.combine_by_coords(datasets, combine_attrs="override")

            region_id = os.path.basename(tmp_dir)
            region_datasets[region_id] = combined_ds

        return region_datasets

    @staticmethod
    def save_csv(df: pd.DataFrame, out_name: str) -> None:
        """Save output in specified format."""
        output_path = f"{out_name}.csv"
        df.to_csv(output_path)
        print(f"✅ File saved to {output_path}")

    def aggregate_dataset(
        self,
        region_datasets: dict[str, xr.Dataset],
        resample_rules: dict[str, str],
        aggregation_type: str,
        output_name: str,
        delete_source: bool
    ):
        """Aggregate datasets per region using the configured schema."""
        for region_id, dataset in region_datasets.items():
            variable_names = list(dataset.data_vars.keys())
            agg_schema = {key: AGG_SCHEMA[key] for key in variable_names if key in AGG_SCHEMA}

            aggregated_ds = xr.Dataset({
                name: self._aggregate_resampled_variable(
                    dataset[predictor].resample(valid_time=resample_rules[aggregation_type]),
                    func,
                )
                for predictor, agg_types in agg_schema.items()
                for agg_dict in [agg_types.get(aggregation_type.lower(), {})]
                if agg_dict != "DROP"
                for name, func in agg_dict.items()
            })

            if aggregation_type == "MONTHLY":
                aggregated_ds["valid_time"] = aggregated_ds["valid_time"].to_index().to_period("M")

            print(f"✅ Aggregation done for region {region_id}")

            save_path = self.save_netcdf(
                dataset=aggregated_ds,
                output_name=f"{output_name}_{region_id}",
                aggregation_type=aggregation_type,
                delete_source=delete_source
            )

            print(f"✅ Aggregation saved to {save_path}")

    @staticmethod
    def _aggregate_resampled_variable(
        resampled,
        operation: str,
    ) -> xr.DataArray:
        """Apply a supported aggregation operation to a resampled DataArray."""
        if operation == "delta":
            return resampled.last() - resampled.first()
        aggregator = getattr(resampled, operation, None)
        if aggregator is None:
            raise ValueError(f"Unsupported aggregation operation: {operation}")
        return aggregator()

    def save_netcdf(
        self,
        dataset: xr.Dataset,
        output_name: str,
        aggregation_type: str | None = None,
        delete_source: bool | None = None,
    ) -> Path:
        """Persist a dataset to NetCDF with compression and optional cleanup."""
        if aggregation_type:
            filename = f"{output_name}_{aggregation_type.lower()}.nc"
        else:
            filename = f"{output_name}.nc"

        path = Path(self.config.OUTPUT_PROCESSED_DIR) / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        # Overwrite if exists
        if path.exists():
            print(f"⚠️ Overwriting existing{' aggregated' if aggregation_type else ''} file: {path}", flush=True)
            path.unlink()

        if "valid_time" in dataset.coords:
            dataset = dataset.assign_coords(
                valid_time=("valid_time", np.array(dataset["valid_time"].values, dtype="datetime64[ns]"))
            )

        encoding = {}
        for v in dataset.data_vars:
            enc = {"zlib": True, "complevel": 4}
            # If float64 isn't required, store as float32 to cut size in half
            if str(dataset[v].dtype).startswith("float64"):
                enc["dtype"] = np.float32
            encoding[v] = enc

        dataset.to_netcdf(path, encoding=encoding, engine="h5netcdf")

        if delete_source:
            src = Path(self.config.OUTPUT_PROCESSED_DIR) / f"{output_name}.nc"
            try:
                src.unlink()
            except FileNotFoundError:
                pass

        return path

    def save_netcdf_daily(
        self,
        dataset: xr.Dataset,
        output_name: str,
    ) -> list[Path]:
        """Persist one NetCDF file per day when valid_time is available."""
        if "valid_time" not in dataset.coords:
            return [self.save_netcdf(dataset=dataset, output_name=output_name)]

        valid_time = pd.to_datetime(dataset["valid_time"].values, errors="coerce")
        if len(valid_time) == 0:
            return [self.save_netcdf(dataset=dataset, output_name=output_name)]

        day_tokens = pd.Series(valid_time).dt.strftime("%Y-%m-%d")
        unique_days = [day for day in day_tokens.dropna().unique()]
        if not unique_days:
            return [self.save_netcdf(dataset=dataset, output_name=output_name)]

        saved_paths: list[Path] = []
        for day in unique_days:
            indexes = np.where(day_tokens == day)[0]
            if len(indexes) == 0:
                continue
            day_dataset = dataset.isel(valid_time=indexes)
            day_name = f"{output_name}_{day}"
            saved_paths.append(self.save_netcdf(dataset=day_dataset, output_name=day_name))

        return saved_paths
