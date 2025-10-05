# carbonpipeline/dataset.py
import glob
import os
import shutil
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr
import rioxarray as rxr

from .api_request import CO2_FOLDERNAME
from .Processing.constants import SHORTNAME_TO_FULLNAME
from .Processing.processor import DataProcessor
from .config import CarbonPipelineConfig
from .Processing.processing_utils import AGG_SCHEMA

class DatasetManager:
    """Manages dataset operations including merging, processing, and saving."""
    
    def __init__(self, config: CarbonPipelineConfig):
        self.config = config

    def merge_unzipped(self, dirs: list[str]) -> Union[xr.Dataset, None]:
        paths = [p for d in dirs for p in glob.glob(os.path.join(d, "*.nc"))]
        if not paths:
            return None

        # Fallback w/o Dask: open each file and combine
        dsets = [xr.open_dataset(p, engine="netcdf4", drop_variables=["number", "expver"])
                 for p in paths]
        ds = xr.combine_by_coords(dsets, combine_attrs="override")
        return ds

    def add_co2_column(self, ds_era5: xr.Dataset, ds_co2: xr.Dataset) -> xr.Dataset:
        """Prepare CO2 dataset for merging."""

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

        ds_era5_coord_reajusted = self._assign_closest_lat_lon(ds_era5, ds_co2_monthly_cut, "latitude", "longitude")
        ds_era5_sortby = ds_era5_coord_reajusted.sortby(['lat', 'lon'], ascending=[False, False])

        co2_selected = ds_co2_sortby["xco2"].sel(
            year_month=ds_era5_sortby["year_month"],
            latitude=ds_era5_sortby["lat"],
            longitude=ds_era5_sortby["lon"]
        )

        ds_era5_sortby["xco2"] = (("valid_time", "latitude", "longitude"), co2_selected.data)

        return ds_era5_sortby

    def add_wtd_column(self, ds_era5: xr.Dataset, ds_wtd: xr.Dataset) -> xr.Dataset:
        """Add WTD column to ERA5 dataset."""

        # Remove unwanted columns
        ds_wtd = ds_wtd.drop_vars("spatial_ref")

        # Rename WTD indexes so it matches ERA5 indexes
        ds_wtd_renamed = ds_wtd.rename({"time": "valid_time", "y": "latitude", "x": "longitude"})

        # Add column year_month to both dataset
        ds_wtd_renamed = self._add_year_month(ds_wtd_renamed, "valid_time")
        ds_era5 = self._add_year_month(ds_era5, "valid_time")
        
        ds_wtd_monthly = ds_wtd_renamed.groupby('year_month').mean(dim='valid_time')
        ds_wtd_sortby = ds_wtd_monthly.sortby(['latitude', 'longitude'], ascending=[False, False])

        """print(ds_era5.isel(valid_time=slice(0, 5)).to_dataframe())"""

        ds_wtd_coord_reajusted = self._assign_closest_lat_lon(ds_wtd_sortby, ds_era5, "lat", "lon")

        # Reconstructing the good index
        ds_wtd_coord_reajusted = ds_wtd_coord_reajusted.set_index({
            'year_month': 'year_month',
            'latitude': 'lat',
            'longitude': 'lon'
        })

        """print(ds_wtd_coord_reajusted.to_dataframe())"""

        # Manipulate the index
        df = ds_wtd_coord_reajusted.to_dataframe().reset_index()

        # Delete duplicates
        df = df.drop_duplicates(subset=["year_month", "latitude", "longitude"])
        ds_wtd_coord_reajusted_clean = df.set_index(["year_month", "latitude", "longitude"]).to_xarray()

        wtd_selected = ds_wtd_coord_reajusted_clean["wtd"].sel(
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
        year_month_periods = pd.to_datetime(ds[time_coord].values).to_period('M')
        ds['year_month'] = (time_coord, year_month_periods)
        return ds

    def _assign_closest_lat_lon(
            self, 
            ds_projected_on: xr.Dataset, 
            ds_projecting: xr.Dataset,
            lat_name: str,
            lon_name: str
    ) -> xr.Dataset:
        """Assign closest lat/lon coordinates."""
        b_lats = np.unique(ds_projecting[lat_name].values)
        b_lons = np.unique(ds_projecting[lon_name].values)
        
        return ds_projected_on.assign_coords(
            lat=("latitude", self._match_to_closest(ds_projected_on["latitude"].values, b_lats)),
            lon=("longitude", self._match_to_closest(ds_projected_on["longitude"].values, b_lons)),
        )

    def load_and_clean_co2_dataset(self) -> xr.Dataset:
        """Load and clean CO2 dataset."""
        co2_files = glob.glob(os.path.join(self.config.UNZIP_DIR, CO2_FOLDERNAME, "*.nc"))
        if not co2_files:
            return None
        ds = xr.open_dataset(co2_files[0])
        ds["xco2"] = ds["xco2"].where(ds["xco2"] < 1e10, np.nan)

        return ds[["xco2"]]

    def load_and_clean_wtd_dataset(self, start: str, end: str) -> Union[xr.Dataset, None]:
        """Load and clean WTD dataset."""
        wtd_dir_name = "_".join(["WTD", pd.to_datetime(start).strftime("%Y-%m"), pd.to_datetime(end).strftime("%Y-%m")])
        wtd_full_path = os.path.join(self.config.UNZIP_DIR, wtd_dir_name)

        wtd_files = glob.glob(os.path.join(wtd_full_path, "*.tif"))
        if not wtd_files:
            return None
        
        datasets = []
        for fn in wtd_files:
            da = rxr.open_rasterio(fn, masked=True).squeeze("band", drop=True)
            nb_to_agg = int(np.ceil(self.config.ERA5_RES/self.config.WTD_RES))
            da_coarse = da.coarsen(x=nb_to_agg, y=nb_to_agg, boundary="trim").mean()
            ds = da_coarse.to_dataset(name="wtd")
            # Extract date from filename and set as time coordinate
            date_str = os.path.basename(fn).split('-')[2].split('.')[0]
            time_val = pd.to_datetime(date_str, format='%Y%m%d')
            ds = ds.expand_dims(time=[time_val])
            datasets.append(ds)
        
        return xr.concat(datasets, dim="time") if datasets else None

    def filter_coordinates(self, ds: xr.Dataset, regions: dict[str | int, list[float]]) -> list[xr.Dataset]:
        ds_copy = ds.copy()

        ds_lats = ds_copy.coords["latitude"].values
        ds_lons = ds_copy.coords["longitude"].values

        all_regions_to_retain = []
        for region_id, (lat_max, lon_min, lat_min, lon_max) in regions.items():
            lat_max_era5 = self._nearest_point(lat_max, ds_lats)
            lon_max_era5 = self._nearest_point(lon_max, ds_lons)
            lat_min_era5 = self._nearest_point(lat_min, ds_lats, prev=lat_max_era5)
            lon_min_era5 = self._nearest_point(lon_min, ds_lons, prev=lon_max_era5)

            # 2. Sélectionner les données avec les coordonnées ERA5
            lats = list({lat_max_era5, lat_min_era5})
            lons = list({lon_max_era5, lon_min_era5})
            rows_to_retain_for_corner = ds_copy.sel(latitude=lats, longitude=lons)

            corresponding_df: pd.DataFrame = (
                rows_to_retain_for_corner
                .to_dataframe()
                .reset_index()
            )

            # 3. Réassigner les coordonnées avec les vraies valeurs de la région
            # Créer un mapping des coordonnées ERA5 vers les vraies coordonnées
            coord_mapping = {
                lat_max_era5: lat_max,
                lat_min_era5: lat_min,
                lon_max_era5: lon_max,
                lon_min_era5: lon_min
            }

            # Appliquer le mapping
            corresponding_df["latitude"] = corresponding_df["latitude"].map(
                lambda x: coord_mapping.get(x, x)
            )
            corresponding_df["longitude"] = corresponding_df["longitude"].map(
                lambda x: coord_mapping.get(x, x)
            )

            corresponding_df["region_id"] = region_id
            corresponding_df = (
                corresponding_df
                .set_index(["region_id", "latitude", "longitude", "valid_time"])
                .sort_index()
            )

            corresponding_ds = corresponding_df.to_xarray()
            all_regions_to_retain.append(corresponding_ds)

        return all_regions_to_retain

    @staticmethod
    def _nearest_point(point: float | int, ds_points: np.ndarray, prev=None):
        if prev is not None:
            filtered = ds_points[ds_points != prev]
            if filtered.size > 0:
                ds_points = filtered
        return ds_points[np.argmin(np.abs(ds_points - point))]

    def _match_to_closest(self, values, reference_points):
        """Match values to closest reference points."""
        reference_points = np.asarray(reference_points)
        return np.array([reference_points[np.abs(reference_points - v).argmin()] for v in values])

    def apply_column_rename(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply column renaming to dataset."""
        rename_dict = {
            k: v 
            for k, v in SHORTNAME_TO_FULLNAME.items() 
            if k in ds.data_vars
        }
        return ds.rename(rename_dict)

    @staticmethod
    def build_multiindex_dataframe(df: pd.DataFrame, preds: list[str]) -> pd.DataFrame:
        """
        Restructures a DataFrame to have a MultiIndex for AmeriFlux vs ERA5 data.
        """
        ameriflux_cols = {p: f"AMF, {p}" for p in preds if p in df.columns}
        renamed_df = df.rename(columns=ameriflux_cols)
        
        for p in preds:
            era5_col_name = f"ERA5, {p}"
            if era5_col_name not in renamed_df.columns:
                 renamed_df[era5_col_name] = np.nan

        tuples = []
        for col in renamed_df.columns:
            if ", " in col:
                src, var = col.split(", ", 1)
            else:
                src, var = "AMF", col
            tuples.append((var, src))

        renamed_df.columns = pd.MultiIndex.from_tuples(tuples, names=["variable", "source"])
        return renamed_df.sort_index(axis=1, level="variable")

    def write_chunks(self, all_dss: list[xr.Dataset], preds: list[str], index: list) -> list[str]:
        """
        Write dataset in chunks.
        """
        tmp_parent_dir = "./outputs_tmp"
        tmp_dirs = []
        shutil.rmtree(tmp_parent_dir, ignore_errors=True)
        os.makedirs(tmp_parent_dir, exist_ok=True)
        processor = DataProcessor(self.config)

        # Process in larger time chunks for efficiency
        for ds in all_dss:
            rid = f"region_{ds.coords['region_id'].values[0]}"
            tmp_dir = os.path.join(tmp_parent_dir, rid)
            os.makedirs(tmp_dir, exist_ok=True)

            df = ds.to_dataframe().reset_index().set_index(index)
            lookup = {p: processor.convert_ameriflux_to_era5(df, p) for p in preds}

            path = os.path.join(tmp_dir, f"{rid}.nc")
            ds = pd.DataFrame(lookup, index=df.index).to_xarray()
            ds.to_netcdf(path, mode="w", format="NETCDF4", engine="netcdf4")

            tmp_dirs.append(tmp_dir)
        
        return tmp_dirs

    @staticmethod
    def concat_chunks(tmp_dirs: list[str]) -> dict[str, xr.Dataset]:
        region_dsets = {}
        for d in tmp_dirs:
            paths = sorted(glob.glob(os.path.join(d, "*.nc")))
            if not paths:
                print(f"No chunks found in {d}, skipping.")
                continue

            dsets = [xr.open_dataset(p, engine="netcdf4") for p in paths]
            ds = xr.combine_by_coords(dsets, combine_attrs="override").load()

            region_id = os.path.basename(d)
            region_dsets[region_id] = ds

        return region_dsets

    @staticmethod
    def save_csv(df: pd.DataFrame, out_name: str) -> None:
        """Save output in specified format."""
        path = f"{out_name}.csv"
        df.to_csv(path)
        print(f"✅ File saved to {path}")

    def aggregate_dataset(
        self,
        region_dsets: dict[str, xr.Dataset],
        resample_methods: dict[str, str],
        aggregation_type: str,
        output_name: str,
        delete_source: bool
    ):
        for rid, ds in region_dsets.items():
            variables = list(ds.data_vars.keys())
            filtered_agg_schema = {key: AGG_SCHEMA[key] for key in variables if key in AGG_SCHEMA}

            agg_ds = xr.Dataset({
                name: getattr(
                    ds[pred].resample(valid_time=resample_methods[aggregation_type]),
                    func
                )()
                for pred, agg_types in filtered_agg_schema.items()
                for agg_dict in [agg_types.get(aggregation_type.lower(), {})]
                if agg_dict != "DROP"
                for name, func in agg_dict.items()
            })

            if aggregation_type == "MONTHLY":
                agg_ds["valid_time"] = agg_ds["valid_time"].to_index().to_period("M")

            print(f"✅ Aggregation done for region {rid}")

            save_path = self.save_netcdf(
                agg_ds=agg_ds,
                output_name=f"{output_name}_{rid}",
                aggregation_type=aggregation_type,
                delete_source=delete_source
            )

            print(f"✅ Aggregation saved to {save_path}")

    def save_netcdf(
        self,
        agg_ds: xr.Dataset,
        output_name: str,
        aggregation_type: str | None = None,
        delete_source: bool | None = None,
    ) -> Path:
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

        if "valid_time" in agg_ds.coords:
            agg_ds = agg_ds.assign_coords(
                valid_time=("valid_time", np.array(agg_ds["valid_time"].values, dtype="datetime64[ns]"))
            )

        encoding = {}
        for v in agg_ds.data_vars:
            enc = {"zlib": True, "complevel": 4}
            # If float64 isn't required, store as float32 to cut size in half
            if str(agg_ds[v].dtype).startswith("float64"):
                enc["dtype"] = np.float32
            encoding[v] = enc

        agg_ds.to_netcdf(path, encoding=encoding, engine="netcdf4")

        if delete_source:
            src = Path(self.config.OUTPUT_PROCESSED_DIR) / f"{output_name}.nc"
            try:
                src.unlink()
            except FileNotFoundError:
                pass

        return path