# carbonpipeline/downloader.py
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
import zipfile
from bs4 import BeautifulSoup
import pandas as pd
import requests
from tqdm import tqdm

from .api_request import CO2_FOLDERNAME, APIRequest
from .config import CarbonPipelineConfig


class DataDownloaderError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class DataDownloader:
    """Handles downloading operations for various data sources."""
    
    def __init__(self, config: CarbonPipelineConfig):
        self.config = config
    
    async def download_co2_data(self) -> None:
        """Download CO2 data asynchronously."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._download_co2_sync)
    
    def _download_co2_sync(self) -> None:
        """Synchronous CO2 download helper."""
        APIRequest.query_co2(self.config.ZIP_DIR)
        zip_fp = os.path.join(self.config.ZIP_DIR, f"{CO2_FOLDERNAME}.zip")
        unzip_fp = os.path.join(self.config.UNZIP_DIR, CO2_FOLDERNAME)
        self._extract_zip(zip_fp, unzip_fp)
        print("\nCO2 data downloaded and extracted.")

    async def download_wtd_data(self, start_date: str, end_date: str, dir_: str) -> None:
        """Web scraping for WTD data asynchronously."""
        loop = asyncio.get_running_loop()
        print("Starting WTD web scraping...")
        await loop.run_in_executor(None, self._web_scraping_wtd_sync, start_date, end_date, dir_)
        print("WTD data download complete.")
    
    def _web_scraping_wtd_sync(self, start_date: str, end_date: str, dir_: str) -> None:
        """Synchronous WTD web scraping helper."""
        response = requests.get(self.config.WTD_URL)
        response.raise_for_status()
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")

        links = soup.find_all('a')

        date_to_filename = {}
        for link in links:
            href = link.get("href")
            if href and ".tif" in href and "-bot-" not in href:
                try:
                    fn, _ = href.split(".")
                    _, _, date_str = fn.split("-")
                    datetime_object = datetime.strptime(date_str, "%Y%m%d")
                    date_to_filename[pd.to_datetime(datetime_object, format="%Y%m")] = href
                except (ValueError, IndexError):
                    continue

        os.makedirs(dir_)

        hrs = pd.date_range(start=start_date, end=end_date, freq="h")
        month_ends = {hr.to_period("M").to_timestamp(how="end").normalize() for hr in hrs}

        fns_to_download = {date_to_filename[d] for d in month_ends if d in date_to_filename}
        list_of_url_filename_pairs = [(self.config.WTD_URL + fn, os.path.join(dir_, fn)) for fn in fns_to_download]
        
        if not list_of_url_filename_pairs:
            raise DataDownloaderError("No WTD files found for the specified date range. Please remove "
                                      "this predictor from the config file or visit the available dates here: "
                                      "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/output/version_1.0/transient_1958-2015/")

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(
                tqdm(
                    executor.map(
                        self._download_tif_with_progress, 
                        list_of_url_filename_pairs
                    ), 
                    total=len(list_of_url_filename_pairs),
                    desc="Downloading WTD files"
                ),
            )

    def _download_tif_with_progress(self, url_filename) -> None:
        """Download TIF file with progress bar."""
        url, filename = url_filename
        
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            #total_size = int(r.headers.get("content-length", 0))

            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    if chunk:
                        f.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")

    async def download_groups_async(
        self,
        groups: list[tuple],
        vars_: list[str],
        coords: list[float],
        monthly: bool,
        region_id: str = None
    ) -> list[str]:
        """Asynchronous wrapper for download_groups using a background thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._download_groups, groups, vars_, coords, monthly, region_id)

    def _download_groups(
        self,
        groups: list[tuple],
        vars_: list[str],
        coords: list[float],
        monthly: bool,
        region_id: str = None
    ) -> list[str]:
        """Download data for multiple groups."""
        fldrs = [] 
        for group in tqdm(groups, desc="Downloading hourly data", unit="group", colour="green"):
            fname = self._prepare_group_request(group, self.config.ZIP_DIR, coords, vars_, monthly)
            if fname:
                zip_fp = os.path.join(self.config.ZIP_DIR, fname)

                # Create region-specific unzip directory
                if region_id:
                    base_unzip_dir = os.path.join(self.config.UNZIP_DIR, region_id)
                    os.makedirs(base_unzip_dir, exist_ok=True)
                    unzip_fp = os.path.join(base_unzip_dir, fname.split('.')[0])
                else:
                    unzip_fp = os.path.join(self.config.UNZIP_DIR, fname.split('.')[0])

                fldrs.append(unzip_fp)
                self._extract_zip(zip_fp, unzip_fp)
        return fldrs

    @staticmethod
    def _prepare_group_request(
        group: tuple,
        dir_: str,
        coords: list[float],
        vars_: list[str],
        monthly: bool
    ) -> str:
        """
        Queries data for a specific date range and location, then downloads the results.
        Group is in the form (year, months, days, hours).
        """
        Y, M, days, hours = group

        request = APIRequest(
            year=Y,
            months=M,        # can be a list
            days=days,       # now a list
            times=hours,     # now a list
            coords=coords,
            vars_=vars_,
            monthly=monthly
        )

        return request.query(dir_)

    @staticmethod
    def _extract_zip(zip_fp: str, unzip_fp: str) -> None:
        """
        Extracts all files from a ZIP archive to a specified directory.
        """
        if not os.path.exists(zip_fp):
            print(f"Warning: ZIP file not found {zip_fp}, skipping extraction.")
            return
        os.makedirs(unzip_fp, exist_ok=True)
        with zipfile.ZipFile(zip_fp, "r") as zp:
            try: 
                zp.extractall(unzip_fp)
                os.remove(zip_fp)
            except zipfile.error as e: 
                print(f"Failed to extract {zip_fp}: {e}")