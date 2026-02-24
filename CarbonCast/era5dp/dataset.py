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

        # Fallback w/o Dask: open each file and combine
        datasets = [
            xr.open_dataset(path, engine="netcdf4", drop_variables=["number", "expver"])
            for path in netcdf_paths
        ]
        merged = xr.combine_by_coords(datasets, combine_attrs="override")
        return merged

    def add_co2_column(self, ds_era5: xr.Dataset, ds_co2: xr.Dataset) -> xr.Dataset:
        """Add CO2 column aligned to the ERA5 grid and monthly time axis."""

        # Rename CO2 indexes so it matches ERA5 indexes
        ds_co2_renamed = ds_co2.rename({"time": "valid_time", "lat": "latitude", "lon": "longitude"})

        # Add column year_month to both dataset
        ds_co2_renamed = self._add_year_month(ds_co2_renamed, "valid_time")
        ds_era5_renamed = self._add_year_month(ds_era5, "valid_time")

        ds_co2_monthly = ds_co2_renamed.groupby('year_month').mean(dim='valid_time')

        # Cut for dates for which we only queried through ERA5
        unique_months_era5 = np.unique(ds_era5_renamed.year_month.values)
        ds_co2_monthly_cut = ds_co2_monthly.sel(year_month=unique_months_era5)
        ds_co2_sortby = ds_co2_monthly_cut.sortby(['latitude', 'longitude'], ascending=[False, False])

        ds_era5_coord_adjusted = self._assign_closest_lat_lon(ds_era5, ds_co2_monthly_cut, "latitude", "longitude")
        ds_era5_sortby = ds_era5_coord_adjusted.sortby(["lat", "lon"], ascending=[False, False])

        co2_selected = ds_co2_sortby["xco2"].sel(
            year_month=ds_era5_sortby["year_month"],
            latitude=ds_era5_sortby["lat"],
            longitude=ds_era5_sortby["lon"]
        )

        ds_era5_sortby["xco2"] = (("valid_time", "latitude", "longitude"), co2_selected.data)

        return ds_era5_sortby

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
        ds_wtd_sortby = ds_wtd_monthly.sortby(["latitude", "longitude"], ascending=[False, False])

        """print(ds_era5.isel(valid_time=slice(0, 5)).to_dataframe())"""

        ds_wtd_coord_adjusted = self._assign_closest_lat_lon(ds_wtd_sortby, ds_era5, "lat", "lon")

        # Reconstructing the good index
        ds_wtd_coord_adjusted = ds_wtd_coord_adjusted.set_index({
            "year_month": "year_month",
            "latitude": "lat",
            "longitude": "lon",
        })

        """print(ds_wtd_coord_reajusted.to_dataframe())"""

        # Manipulate the index
        wtd_df = ds_wtd_coord_adjusted.to_dataframe().reset_index()

        # Delete duplicates
        wtd_df = wtd_df.drop_duplicates(subset=["year_month", "latitude", "longitude"])
        wtd_ds_clean = wtd_df.set_index(["year_month", "latitude", "longitude"]).to_xarray()

        wtd_selected = wtd_ds_clean["wtd"].sel(
            year_month=ds_era5["year_month"],
            latitude=ds_era5["lat"],
            longitude=ds_era5["lon"]
        )

        ds_era5["wtd"] = (("valid_time", "latitude", "longitude"), wtd_selected.data)

        """print(ds_era5.isel(valid_time=slice(0, 5)).to_dataframe())"""
        """print(np.isnan(ds_era5["xco2"].values).all())"""
        """print(np.isnan(ds_era5["wtd"].values).all())"""

        return ds_era5.drop(["year_month", "lat", "lon"])

    def _add_year_month(self, ds: xr.Dataset, time_coord: str) -> xr.Dataset:
        """Add year_month coordinate as datetime64[M] (truncated to month)."""
        year_month_periods = pd.to_datetime(ds[time_coord].values).to_period("M")
        ds["year_month"] = (time_coord, year_month_periods)
        return ds

    def _assign_closest_lat_lon(
            self, 
            ds_projected_on: xr.Dataset, 
            ds_projecting: xr.Dataset,
            lat_name: str,
            lon_name: str
    ) -> xr.Dataset:
        """Assign closest lat/lon coordinates from a reference dataset."""
        ref_lats = np.unique(ds_projecting[lat_name].values)
        ref_lons = np.unique(ds_projecting[lon_name].values)
        
        return ds_projected_on.assign_coords(
            lat=("latitude", self._match_to_closest(ds_projected_on["latitude"].values, ref_lats)),
            lon=("longitude", self._match_to_closest(ds_projected_on["longitude"].values, ref_lons)),
        )

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

            region_df: pd.DataFrame = (
                corner_points
                .to_dataframe()
                .reset_index()
            )

            # Map ERA5 grid coordinates back to the requested region bounds
            coord_mapping = {
                lat_max_grid: lat_max,
                lat_min_grid: lat_min,
                lon_max_grid: lon_max,
                lon_min_grid: lon_min
            }

            region_df["latitude"] = region_df["latitude"].map(
                lambda x: coord_mapping.get(x, x)
            )
            region_df["longitude"] = region_df["longitude"].map(
                lambda x: coord_mapping.get(x, x)
            )

            region_df["region_id"] = region_id
            region_df = (
                region_df
                .set_index(["region_id", "latitude", "longitude", "valid_time"])
                .sort_index()
            )

            region_ds = region_df.to_xarray()
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

    def _match_to_closest(self, values, reference_points):
        """Match values to closest reference points."""
        reference_points = np.asarray(reference_points)
        return np.array([reference_points[np.abs(reference_points - v).argmin()] for v in values])

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

    def write_chunks(self, region_datasets: list[xr.Dataset], predictors: list[str], index_columns: list) -> list[str]:
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

            region_df = dataset.to_dataframe().reset_index().set_index(index_columns)
            predictor_lookup = {
                predictor: data_processor.convert_ameriflux_to_era5(region_df, predictor)
                for predictor in predictors
            }

            output_path = os.path.join(tmp_dir, f"{region_id}.nc")
            chunk_ds = pd.DataFrame(predictor_lookup, index=region_df.index).to_xarray()
            chunk_ds.to_netcdf(output_path, mode="w", format="NETCDF4", engine="netcdf4")

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

            datasets = [xr.open_dataset(path, engine="netcdf4") for path in file_paths]
            combined_ds = xr.combine_by_coords(datasets, combine_attrs="override").load()

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
                name: getattr(
                    dataset[predictor].resample(valid_time=resample_rules[aggregation_type]),
                    func
                )()
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

        dataset.to_netcdf(path, encoding=encoding, engine="netcdf4")

        if delete_source:
            src = Path(self.config.OUTPUT_PROCESSED_DIR) / f"{output_name}.nc"
            try:
                src.unlink()
            except FileNotFoundError:
                pass

        return path
