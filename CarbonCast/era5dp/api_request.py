# carbonpipeline/api_request.py
import os
import cdsapi

from .config import CarbonPipelineConfig



class APIRequest:
    """
    Represents a request to the ERA5 dataset for a specific date, location and set of variables.

    Parameters
    ----------
    year : str 
        Year of the request.
    months : str | list[str]
        Month of the request.
    days : list[str]
        Day of the request.
    times : list[str]
        Time in "HH:MM" format.
    coords : list[float]
        Coordinates of the request.
    era5_vars : list[str] | None
        Optional list of high-level variables under ERA5 long version naming.
    """

    def __init__(
        self,
        year: str,
        months: str | list[str],
        days: str | list[str],
        times: str | list[str],
        coords: list[float],
        era5_vars: list[str],
        monthly: bool = False
    ):
        self.year = year
        self.months = months if isinstance(months, list) else [months]
        self.days = days if isinstance(days, list) else [days]
        self.times = times if isinstance(times, list) else [times]
        self.coords = coords
        self.era5_vars = era5_vars
        self.area = None
        self.monthly = monthly

    def query(self, zip_dir) -> str:
        result = self.retrieve_result()
        return self.download_result(result, zip_dir)

    def retrieve_result(self):
        """Submit an ERA5 request and wait until its result is ready."""
        if self.monthly:
            return self._retrieve_era5_monthly_result()
        return self._retrieve_era5_hourly_result()

    def download_result(self, result, zip_dir: str) -> str:
        """Download a completed ERA5 result to its deterministic ZIP filename."""
        filename = self._filename_logic()
        os.makedirs(zip_dir, exist_ok=True)
        target = os.path.join(zip_dir, filename)

        print(f"\nStarting download for {filename} -> {target}")
        result.download(target)
        print(f"\nFinished download for {filename}")

        return filename

    def expected_filename(self) -> str:
        """Return deterministic ERA5 zip filename for this request."""
        return self._filename_logic()

    def _retrieve_era5_monthly_result(self):
        """
        Submit a monthly ERA5 request and wait until the result is ready.
        """
        if len(self.coords) == 2:
            self.area = [self.coords[0], self.coords[1], self.coords[0], self.coords[1]]
        elif len(self.coords) == 4:
            self.area = self.coords

        dataset = "reanalysis-era5-single-levels-monthly-means"
        request = {
            "product_type": ["monthly_averaged_reanalysis_by_hour_of_day"],
            "variable": self.era5_vars,
            "year": [self.year],
            "month": self.months,
            "time": self.times,
            "area": self.area,
            "data_format": "netcdf",
            "download_format": "zip"
        }

        client = cdsapi.Client(wait_until_complete=True, delete=False)
        return client.retrieve(dataset, request)

    def _retrieve_era5_hourly_result(self):
        """
        Submit an hourly ERA5 request and wait until the result is ready.
        """
        if len(self.coords) == 2:
            self.area = [self.coords[0], self.coords[1], self.coords[0], self.coords[1]]
        elif len(self.coords) == 4:
            self.area = self.coords

        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type":    ["reanalysis"],
            "variable":        self.era5_vars,
            "year":            [self.year],
            "month":           self.months,
            "day":             self.days,
            "time":            self.times,
            "area":            self.area,
            "data_format":     "netcdf",
            "download_format": "zip"
        }

        client = cdsapi.Client(wait_until_complete=True, delete=False)
        return client.retrieve(dataset, request)

    @classmethod
    def query_co2(self, zip_dir: str) -> None:
        dataset = "satellite-carbon-dioxide"
        request = {
            "processing_level": ["level_3"],
            "variable": "xco2",
            "sensor_and_algorithm": "merged_obs4mips",
            "version": ["4_5"]
        }

        client = cdsapi.Client()
        result = client.retrieve(dataset, request)

        filename = f"{CarbonPipelineConfig.CO2_FOLDERNAME}.zip"
        os.makedirs(zip_dir, exist_ok=True)
        target = os.path.join(zip_dir, filename)

        result.download(target)

    def _filename_logic(self) -> str:
        # Always normalize into lists
        years = self.year if isinstance(self.year, list) else [self.year]
        months = self.months if isinstance(self.months, list) else [self.months]

        # Sort to avoid weird order
        years, months, days, times = map(sorted, (years, months, self.days, self.times))
        full_hours = [f"{h:02d}:00" for h in range(24)]

        # Case 1: full-years (rare, but keep)
        if (
            len(months) == 12
            and months == [f"{i:02d}" for i in range(1, 13)]
            and days == [f"{d:02d}" for d in range(1, 32)]
            and times == full_hours
        ):
            return (
                f"ERA5_{years[0]}_full-year.zip"
                if len(years) == 1
                else f"ERA5_{years[0]}to{years[-1]}_full-years.zip"
            )

        # Case 2: full-month
        if len(days) >= 28 and times == full_hours:  # 28 ≈ min days in Feb
            return f"ERA5_{years[0]}-{months[0]}_full-month.zip"

        # Case 3: full-day
        if len(times) == 24:
            return f"ERA5_{years[0]}-{months[0]}-{days[0]}_full-day.zip"

        # Case 4: multi-day partial month
        if len(days) > 1:
            return f"ERA5_{years[0]}-{months[0]}_days{days[0]}to{days[-1]}.zip"

        # Case 5: multi-hour partial day
        if len(times) > 1:
            return f"ERA5_{years[0]}-{months[0]}-{days[0]}T{times[0]}to{times[-1]}.zip"

        # Case 6: single hour
        return f"ERA5_{years[0]}-{months[0]}-{days[0]}T{times[0]}.zip"
