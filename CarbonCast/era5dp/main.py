"""CLI entry point and orchestration for the carbon pipeline."""
# carbonpipeline/cli.py
import asyncio
import calendar
import json
import os
import pandas as pd
from datetime import datetime
from enum import Enum
from pathlib import Path

from .argparser import ArgumentParserManager
from .Geometry.geometry_processor import GeometryProcessor
from .Geometry.geometry import Geometry, GeometryType
from .Processing.constants import *
from .core import CarbonPipeline
from .download_registry import DownloadRegistry


class CommandExecutorError(Exception):
    """Raised for configuration or execution errors in the CLI layer."""
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class SpecialPredictors:
    """Tracks optional datasets (CO2/WTD) required by selected predictors."""
    def __init__(self, predictors: list[str]):
        self.requires_wtd_data = "WTD" in predictors
        self.requires_co2_data = "CO2" in predictors
        self.download_registry = DownloadRegistry()

    async def download_required_data(self, pipeline, start, end):
        """Kick off CO2/WTD downloads if required and missing on disk."""
        tasks = []

        if self.requires_co2_data:
            co2_root = Path(pipeline.config.CO2_DIR)
            co2_present = self.download_registry.has_data(co2_root, kind="nc")
            if not co2_present:
                print("⬇️ Downloading CO2 data...")
                tasks.append(asyncio.create_task(pipeline.downloader.download_co2_data(co2_root)))
            else:
                print("CO2 data has already been downloaded.")

        if self.requires_wtd_data:
            start_str = pd.to_datetime(start).strftime("%Y-%m")
            end_str = pd.to_datetime(end).strftime("%Y-%m")
            time_window = "_".join([start_str, end_str])

            wtd_path = Path(os.path.join(pipeline.config.WTD_DIR, time_window))
            wtd_present = self.download_registry.has_data(wtd_path, kind="tif")
            if not wtd_present:
                print("⬇️ Download WTD data...")
                tasks.append(asyncio.create_task(pipeline.downloader.download_wtd_data(start, end, wtd_path)))
            else:
                print(f"WTD data for {time_window} has already been downloaded.")

        return tasks


class ProcessingType(Enum):
    AREA = "Area"
    BOXES = "Boxes"
    SITE = "Site"


class CommandExecutor:
    """Orchestrates download/process stages based on the YAML config."""
    def __init__(self, config_dict: dict):
        self.pipeline = CarbonPipeline()

        self.manifest_path = config_dict.get("manifest")

        self.action = config_dict.get("action")
        self.output_suffix = config_dict.get("output-filename")
        self.ameriflux_csv = config_dict.get("data-file")
        self.location = config_dict.get("location-coordinates")
        self.area = config_dict.get("bbox-coordinates")
        self.geometries_dir = self.validate_geometries_dir(config_dict.get("geometries-directory"))
        self.start = config_dict.get("start-date")
        self.end = config_dict.get("end-date")
        self.predictors = config_dict.get("ameriflux-predictors")
        self.aggregation_type = config_dict.get("aggregation-type").upper()
        self.id_field = config_dict.get("id-field")

        self.all_geometries: dict[str | int, Geometry] = {}
        self.bounding_boxes_geometry: dict[Geometry, dict[str | int, list[float]]] = {}
        self.special_preds: SpecialPredictors | None = None # SpecialPredictors object
        self.era5_vars: list[str] | None = None # List of variables to download from ERA5
        self.geometry_mode: ProcessingType

    @staticmethod
    def validate_geometries_dir(coords_file: str | None) -> str | None:
        """
        Validate that coords_file is a directory if provided.
        Returns the same path if valid, or None if no path given.
        """
        if coords_file is None:
            return None

        path = Path(coords_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"coords_file path does not exist: {coords_file}")
        if not os.path.isdir(path):
            raise NotADirectoryError(f"coords_file is not a directory: {coords_file}")

        return coords_file

    @property
    def number_requests_per_region(self):
        """
        Compute number of CDS requests per region dynamically.
        Uses grouping logic instead of raw hour-difference.
        """
        start = self._parse_datetime(self.start)
        end = self._parse_datetime(self.end)

        # Ask processor to build request groups
        groups = self.pipeline.processor.build_request_groups(start, end, self.aggregation_type == "MONTHLY")

        return len(groups)

    # Only callable function
    async def run(self):
        """Dispatch to download or process based on config action."""
        match self.action:
            case "download":
                self._prepare_download_inputs()
                ArgumentParserManager.pretty_print_inputs(
                    "Downloading Data Step",
                    StartDate=self.start, 
                    EndDate=self.end, 
                    AMFPredictors=self.predictors
                )
                await self._downloading_step()
            case "process":
                content = self.pipeline.load_features_from_manifest(self.manifest_path)
                self.ameriflux_csv = content.get("data_file")
                ArgumentParserManager.pretty_print_inputs(
                    "Processing Data Step", 
                    OutputDirectory=self.pipeline.config.OUTPUT_PROCESSED_DIR,
                    DataFile=self.ameriflux_csv
                )
                self._processing_step()
            case _:
                raise ValueError(f"Unknown action: {self.action}")

    async def _downloading_step(self):
        """Execute download stage (CO2/WTD first, then ERA5 by region)."""
        self.pipeline.setup_manifest_and_dirs(
            self.manifest_path, 
            self.pipeline.config.ZIP_DIR
        )

        # download WTD/CO2 data ONCE at the beginning (global datasets)
        global_tasks = await self.special_preds.download_required_data(
            self.pipeline, 
            self.start, 
            self.end,
        )

        from .downloader import DataDownloaderError
        if global_tasks:
            results = await asyncio.gather(*global_tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, DataDownloaderError):
                    if not self._maybe_skip_wtd(res):
                        raise res
                elif isinstance(res, Exception):
                    raise res

        # download ERA5 data sequentially for each region (to avoid CDS conflicts)
        if self.geometry_mode == ProcessingType.SITE:
            gapfilling = await self._ask_gapfill()
            for i, geometry_idx in enumerate(self.all_geometries):
                geometry = self.all_geometries[geometry_idx]
                region = geometry.bbox
                region_id = CommandExecutor._generate_region_id(region, i)
                await self._download_for_stations(geometry, region, region_id, gapfilling)

        elif self.geometry_mode == ProcessingType.BOXES:
            for geometry_idx, geometry in enumerate(self.bounding_boxes_geometry):
                region = geometry.bbox
                region_id = CommandExecutor._generate_region_id(region, geometry_idx)
                await self._download_for_region(geometry, region, region_id, self.bounding_boxes_geometry[geometry])

        elif self.geometry_mode == ProcessingType.AREA:
            for geometry_idx, gid in enumerate(self.all_geometries):
                geometry = self.all_geometries[gid]
                region = geometry.bbox
                region_id = CommandExecutor._generate_region_id(region, geometry_idx)
                await self._download_for_region(geometry, region, region_id, {gid: region})

    def _maybe_skip_wtd(self, err: Exception) -> bool:
        """Skip WTD only for the 'no files for date range' error."""
        message = str(err).strip()
        if "No WTD files found for the specified date range" not in message:
            return False

        suffix = f" Details: {message}" if message else ""
        print(f"WTD data unavailable for the requested date range. Skipping WTD.\n{suffix}")

        if self.predictors and "WTD" in self.predictors:
            self.predictors.remove("WTD")
        if self.era5_vars and "wtd" in self.era5_vars:
            self.era5_vars.remove("wtd")
        if self.special_preds:
            self.special_preds.requires_wtd_data = False
        return True

    @staticmethod
    async def _ask_gapfill() -> bool:
        """Prompt user whether to gap-fill EC station data."""
        while True:
            ans = (await asyncio.to_thread(
                input, "\nDo you want to gap-fill the dataset in input? (Y/n): "
            )).strip()
            if ans.upper() == "Y":
                return True
            if ans.lower() == "n":
                return False
            print("Invalid input: please enter 'Y' to gap-fill, or 'n' if not.")

    def _processing_step(self):
        """Execute processing stage from manifest.json."""
        content = self.pipeline.load_features_from_manifest(self.manifest_path)
        self.geometry_mode = content.get("processing_type")
        self.aggregation_type = content.get("aggregation_type")
        gapfilling = content.get("gapfilling")
        features = content.get("features")
        for i in range(len(features)):
            region_id = features[i].get("region_id")
            predictors = features[i].get("preds")
            start = features[i].get("start_date")
            end = features[i].get("end_date")
            geometry = features[i].get("geometry")
            rect_regions = features[i].get("rect_regions")
            unzip_dirs = features[i].get("unzip_sub_folders")
            ds = self.pipeline.dataset_manager.merge_unzipped_netcdfs(unzip_dirs)

            if not self.output_suffix:
                self.output_suffix = "output"
            output_name = "_".join([self.output_suffix , region_id])

            match geometry:
                case GeometryType.POINT.value:
                    if gapfilling:
                        self.pipeline.run_point_process(self.ameriflux_csv, ds, predictors, start, end,
                                                        region_id, gapfilling, output_name)
                    else:
                        # fallback if the client doesn't want gap-filling to the given dataset
                        self.pipeline.run_area_process(ds, predictors, start, end, rect_regions,
                                                       output_name, self.geometry_mode, self.aggregation_type)
                case _:
                    self.pipeline.run_area_process(ds, predictors, start, end, rect_regions,
                                                   output_name, self.geometry_mode, self.aggregation_type)

    @staticmethod
    def _generate_region_id(region: list[float], geometry_idx: int) -> str:
        """
        Generate a unique identifier for each region.
        """
        lat_range = f"{region[2]:.1f}to{region[0]:.1f}"
        lon_range = f"{region[1]:.1f}to{region[3]:.1f}"
        return f"r{geometry_idx}_{lat_range}_{lon_range}"
            
    def _prepare_download_inputs(self):
        """
        Cleaning, parsing, filling all variables passed through the config file.
        """
        # Disallow aggregation when requesting global/world data (coords_dir=None)
        aggregation = getattr(self, "aggregation_type", None)
        if self.geometries_dir is None and (aggregation is not None and str(aggregation).strip().lower() not in {"", "none"}):
            raise ValueError(
                "Aggregation is not supported when `coords_dir` is None (global/world extraction). "
                "Provide `coords_dir` with geometries or set `aggregation_type` to None."
            )

        # Check the dates
        if self._validate_date_range():
            # Make a list out of the predictors
            requested_predictors = list(self.predictors or [])

            # If the requested date are out of bounds for the CO2 dataset
            co2_start_year = 2002
            co2_end_year = 2023
            if "CO2" in requested_predictors:
                start = self._parse_datetime(self.start)
                end = self._parse_datetime(self.end)

                if start.year < co2_start_year or end.year > co2_end_year:
                    print("Removing the CO2 predictors from the list because it is out of bounds for"
                          f"the requested start and end date (before {co2_start_year} or after {co2_end_year}).", flush=True)
                    requested_predictors.remove("CO2")

            # Check if any is not supported
            invalid = [p for p in requested_predictors if p not in VARIABLES_FOR_PREDICTOR]
            if invalid:
                raise ValueError(f"Invalid predictors: {invalid}.\nConsult the README for the available predictors.")

            if not requested_predictors: # Case if no predictors has been specified in the config file
                self.era5_vars = ERA5_VARIABLES
                self.predictors = list(VARIABLES_FOR_PREDICTOR)
            else:
                self.era5_vars = list({var for pred in requested_predictors for var in VARIABLES_FOR_PREDICTOR[pred]})
                self.predictors = requested_predictors

            self.special_preds = SpecialPredictors(predictors=self.predictors)

            if "xco2" in self.era5_vars:
                self.era5_vars.remove("xco2") # If we don't do that, the ERA5 request will fail because xco2 doesn't exist within this particular dataset
            if "wtd" in self.era5_vars:
                self.era5_vars.remove("wtd") # Same here

            # Verify if location was provided and if CO2 and WTD are not available for EC Stations data query
            if self.ameriflux_csv:
                if not self.location:
                    raise CommandExecutorError("No location has been provided for the EC tower. Please modify the "
                                               "config file such that [latitude, longitude] is in the location"
                                               "argument.")

                removed = []
                for special_pred in ["CO2", "WTD"]:
                    if special_pred in self.predictors:
                        if special_pred == "CO2":
                            self.special_preds.requires_co2_data = False
                        else:  # must be "WTD"
                            self.special_preds.requires_wtd_data = False
                        self.predictors.remove(special_pred)
                        removed.append(special_pred)

                if removed:
                    print(f"The AmeriFlux predictor/s {removed} are not available for point query.")
                    print("Removing it/them.")

            # If CSV file is given
            if self.geometries_dir is None and self.ameriflux_csv is not None:
                geometry = Geometry(data=self.location)
                geometry.validate_coordinates()
                geometry.bbox = GeometryProcessor.process_geometry(geometry)
           
                self.all_geometries = {0: geometry}
                self.geometry_mode = ProcessingType.SITE

            # Default
            elif self.geometries_dir is None and self.ameriflux_csv is None:
                if self.area is None:
                    raise CommandExecutorError(
                        "No area provided. When geometries-directory and data-file are both None, "
                        "you must set `bbox-coordinate: [N, W, S, E]` in the config."
                    )
                geometry = Geometry()
                geometry.bbox = self.area
                self.all_geometries = {0: geometry}
                self.geometry_mode = ProcessingType.AREA

            # If directory with GeoJSONS is given
            elif self.geometries_dir is not None and self.ameriflux_csv is None:
                self.all_geometries = self._parse_geojsons()

                for _, geometry in self.all_geometries.items():
                    geometry.bbox = GeometryProcessor.process_geometry(geometry)

                geometry = Geometry()
                geometry.bbox = self.compute_covering_bbox(list(self.all_geometries.values()))
                self.bounding_boxes_geometry[geometry] = {
                    id_geo: geo.bbox
                    for id_geo, geo in self.all_geometries.items()
                }
                self.geometry_mode = ProcessingType.BOXES  # let know the pipeline to download in the manifest


    @staticmethod
    def _parse_datetime(value):
        """Parse datetime from config into a datetime object."""
        if isinstance(value, str):
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        elif isinstance(value, datetime):
            return value
        else:
            raise TypeError(f"Unsupported type for datetime parsing: {type(value)}")

    def _validate_date_range(self):
        """Validate that start/end are ordered and match aggregation rules."""
        # Clean the dates
        start = self._parse_datetime(self.start)
        end = self._parse_datetime(self.end)

        if end <= start:
            raise ValueError(
                "End datetime must be after start datetime.\n" +
                f"Current start time: {start}, and end time: {end}"
            )

        # Check aggregation type and if it fits with self.start and self.end
        if self.aggregation_type == "DAILY":
            # Must align on whole days
            if not (start.hour == start.minute == start.second == 0):
                raise ValueError(
                    "Start datetime must be at midnight for DAILY aggregation.\n" +
                    f"Current start time: {start}"
                )
            if not (end.hour == 23 and end.minute == end.second == 0):
                raise ValueError(
                    "Start datetime must be at 23:00 for DAILY aggregation.\n" +
                    f"Current end time: {end}"
                )

            # Duration check: must be whole number of days
            delta_days = (end.date() - start.date()).days + 1
            if delta_days <= 0:
                raise ValueError(
                    "Time range must cover at least one full day.\n" +
                    f"Current start time: {start}, and end time: {end}"
                )
        elif self.aggregation_type == "MONTHLY":

            # Start must be first day of a month at 00:00
            if not (start.day == 1 and start.hour == start.minute == start.second == 0):
                raise ValueError(
                    "Start must be the first day of the month at 00:00:00 for MONTHLY aggregation.\n" +
                    f"Current start time: {start}"
                )

            # End must be last day of a month at 23:00
            last_day = calendar.monthrange(end.year, end.month)[1]
            if not (end.day == last_day and end.hour == 23 and end.minute == end.second == 0):
                raise ValueError(
                    "End must be the last day of the month at 23:00:00 for MONTHLY aggregation.\n" +
                    f"Current end time: {end}"
                )

            months_diff = (end.year - start.year) * 12 + (end.month - start.month) + 1
            if months_diff <= 0:
                raise ValueError(
                    "End date must be after start date.\n"
                    f"Current start date: {start.year}-{start.month:02d}, "
                    f"end date: {end.year}-{end.month:02d}"
                )
        elif self.aggregation_type in (None, "", "NONE"):
            pass
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")

        self.start = start.strftime("%Y-%m-%d %H:%M:%S")
        self.end = end.strftime("%Y-%m-%d %H:%M:%S")

        return True

    def _parse_geojsons(self) -> dict:
        """
        Parse the coordinate input provided by the user.
        """
        path = Path(self.geometries_dir)

        # If the provided path is a directory
        geometries_per_file: dict[str | int, Geometry] = {}
        if path.is_dir():
            missing_counter = 1
            for file_path in sorted(path.iterdir()):
                # Only consider files with recognised extensions
                if file_path.suffix not in (".geojson", ".json"):
                    continue
                with open(file_path, "r") as f:
                    json_dict = json.load(f)
                    if not json_dict.get("features"):
                        raise ValueError(f"No features found in GeoJSON file: {file_path}")
                    features = json_dict["features"]

                    for feature in features:
                        # Assigning an ID to the region
                        props = feature.get("properties", {})
                        if self.id_field in props:
                            id_geo = props[self.id_field]
                        else:
                            id_geo = missing_counter
                            missing_counter += 1

                        coordinates = feature["geometry"]["coordinates"]
                        geometry = Geometry(data=coordinates)
                        geometry.validate_coordinates()
                        geometries_per_file[id_geo] = geometry

            if not geometries_per_file:
                raise ValueError(f"No valid GeoJSON files found in directory: {path}")
        return geometries_per_file

    def compute_covering_bbox(self, geometries: list[Geometry]) -> list[float]:
        """Build a single bounding box covering all given geometries."""
        rects = self._all_rect_regions(geometries)

        if not rects:
            raise ValueError("No rect regions available to build a global covering region.")

        N = max(r[0] for r in rects)
        W = min(r[1] for r in rects)
        S = min(r[2] for r in rects)
        E = max(r[3] for r in rects)
        return [N, W, S, E]

    @staticmethod
    def _all_rect_regions(geometries: list[Geometry]):
        """Return rect regions for all geometries."""
        return [
            g.bbox
            for g in geometries
        ]

    async def _download_for_stations(self, geometry, region, region_id, gapfilling):
        """Download ERA5 data for a EC station. Runs sequentially to avoid CDS conflicts."""
        print(f"⬇️ Downloading ERA5 data for {region_id}...")
        await self.pipeline.run_download_point(
            download_bbox=region,
            region_id=region_id,
            geometry=geometry,
            start=self.start,
            end=self.end,
            predictors=self.predictors,
            era5_vars=self.era5_vars,
            gapfilling=gapfilling,
            ameriflux_csv=self.ameriflux_csv,
            manifest_path=self.manifest_path
        )

    async def _download_for_region(self, geometry, region, region_id, region_bboxes: dict[str | int, list[float]]):
        """Download ERA5 data for a single region. Runs sequentially to avoid CDS conflicts."""
        print(f"⬇️ Downloading ERA5 data for {region_id}...")
        await self.pipeline.run_download_area(
            download_bbox=region,
            region_id=region_id,
            geometry=geometry,
            start=self.start,
            end=self.end,
            predictors=self.predictors,
            era5_vars=self.era5_vars,
            region_bboxes=region_bboxes,
            geometry_mode=self.geometry_mode.value,
            aggregation_type=self.aggregation_type,
            manifest_path=self.manifest_path,
        )
                

async def main():
    args = ArgumentParserManager.build_parser().parse_args()
    config = ArgumentParserManager.load_yaml_config(args.config)
    
    ce = CommandExecutor(config_dict=config)

    if ce.action == "process":
        unzip_dir = ce.pipeline.config.UNZIP_DIR
        if not os.path.exists(unzip_dir):
            raise CommandExecutorError(f"Unzip directory does not exist: {unzip_dir}. Please download data first.")
        if not os.listdir(unzip_dir):
            raise CommandExecutorError("No downloads found in the unzip directory. Please download data first.")

    await ce.run()


def run():
    """Synchronous entry point that runs the main async function."""
    try:
        asyncio.run(main())
    except (ValueError, FileNotFoundError) as e:
        print(f"An error occurred: {e}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")


if __name__ == "__main__":
    run()
