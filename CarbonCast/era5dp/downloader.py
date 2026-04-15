"""Download utilities for ERA5, CO2, and WTD datasets."""
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
from pathlib import Path
import time
import zipfile
import pandas as pd
import requests
from tqdm import tqdm

from .api_request import APIRequest
from .config import CarbonPipelineConfig
from .download_registry import DownloadRegistry

ERA5_TRANSFER_READY_SENTINEL = ".transfer_ready"


class DataDownloaderError(Exception):
    """Raised when a download-related workflow cannot proceed."""
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class DataDownloader:
    """Handles downloading operations for various data sources."""

    ERA5_READY_RESULT_POOL_SIZE = 4
    ERA5_TRANSFER_POLL_SECONDS = 5
    
    def __init__(self, config: CarbonPipelineConfig):
        self.config = config
        self.download_registry = DownloadRegistry()
    
    async def download_co2_data(self, path: str | Path) -> None:
        """Download CO2 data asynchronously."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._download_co2_sync, path)

    async def download_wtd_data(self, start_date: str, end_date: str, path: str | Path) -> None:
        """Download WTD data asynchronously (via web scraping)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._web_scraping_wtd_sync, start_date, end_date, path)

    def _download_co2_sync(self, unzip_path: str | Path) -> None:
        """Synchronous CO2 download helper."""
        APIRequest.query_co2(self.config.ZIP_DIR)
        zip_path = Path(f"{self.config.ZIP_DIR}/{self.config.CO2_FOLDERNAME}.zip")
        self._extract_zip(zip_path, unzip_path)

    def _web_scraping_wtd_sync(self, start_date: str, end_date: str, unzip_path: str | Path) -> None:
        """Synchronous WTD web scraping helper."""
        from bs4 import BeautifulSoup
        index_response = requests.get(self.config.WTD_URL)
        index_response.raise_for_status()
        html_text = index_response.text
        soup = BeautifulSoup(html_text, "html.parser")

        anchors = soup.find_all("a")

        date_to_filename = {}
        for link in anchors:
            href = link.get("href")
            if href and ".tif" in href and "-bot-" not in href:
                try:
                    fn, _ = href.split(".")
                    _, _, date_str = fn.split("-")
                    datetime_object = datetime.strptime(date_str, "%Y%m%d")
                    date_to_filename[pd.to_datetime(datetime_object, format="%Y%m")] = href
                except (ValueError, IndexError):
                    continue

        hrs = pd.date_range(start=start_date, end=end_date, freq="h")
        month_ends = {hr.to_period("M").to_timestamp(how="end").normalize() for hr in hrs}

        filenames_to_download = {date_to_filename[d] for d in month_ends if d in date_to_filename}
        list_of_url_filename_pairs = [
            (self.config.WTD_URL + filename, os.path.join(unzip_path, filename))
            for filename in filenames_to_download
        ]

        if not list_of_url_filename_pairs:
            raise DataDownloaderError(
                "No WTD files found for the specified date range. Please remove "
                "this predictor from the config file or visit the available dates here: "
                "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/output/version_1.0/transient_1958-2015/"
            )

        os.makedirs(unzip_path, exist_ok=True)

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
        """Download a single TIF file."""
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

    async def download_request_groups_async(
        self,
        groups: list[tuple],
        era5_vars: list[str],
        coords: list[float],
        monthly: bool,
        region_id: str = None
    ) -> list[str]:
        """Asynchronous wrapper for request-group downloads."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._download_request_groups,
            groups,
            era5_vars,
            coords,
            monthly,
            region_id,
        )

    def _download_request_groups(
        self,
        groups: list[tuple],
        era5_vars: list[str],
        coords: list[float],
        monthly: bool,
        region_id: str = None
    ) -> list[str]:
        """Request ERA5 results continuously, but download/extract one chunk at a time."""
        unzip_dirs = []
        pending_results: deque[tuple[APIRequest, object, str]] = deque()

        for group in tqdm(groups, desc="Requesting ERA5 data", unit="group", colour="green"):
            request = self._build_group_request(group, coords, era5_vars, monthly)
            zip_name = request.expected_filename()
            zip_fp = os.path.join(self.config.ZIP_DIR, zip_name)

            # Create region-specific unzip directory
            if region_id:
                base_unzip_dir = os.path.join(self.config.ERA5_DIR, region_id)
                os.makedirs(base_unzip_dir, exist_ok=True)
                unzip_fp = os.path.join(base_unzip_dir, zip_name.split(".")[0])
            else:
                os.makedirs(self.config.ERA5_DIR, exist_ok=True)
                unzip_fp = os.path.join(self.config.ERA5_DIR, zip_name.split(".")[0])

            unzip_dirs.append(unzip_fp)

            # Skip if this chunk already exists locally or is known available from registry state.
            if self.download_registry.has_data(unzip_fp):
                print(f"Skipping existing ERA5 chunk: {unzip_fp}")
                continue

            if os.path.exists(zip_fp):
                if self._extract_zip(zip_fp, unzip_fp):
                    self.download_registry.mark_available(unzip_fp, kind="nc")
                    self._mark_era5_chunk_ready(unzip_fp)
                continue

            if len(pending_results) >= self.ERA5_READY_RESULT_POOL_SIZE:
                self._download_next_ready_result(pending_results)

            result = request.retrieve_result()
            pending_results.append((request, result, unzip_fp))

            if self._era5_transfer_slot_available():
                self._download_next_ready_result(pending_results, wait_for_transfer_slot=False)

        while pending_results:
            self._download_next_ready_result(pending_results)
        return unzip_dirs

    @staticmethod
    def _build_group_request(
        group: tuple,
        coords: list[float],
        era5_vars: list[str],
        monthly: bool
    ) -> APIRequest:
        """
        Build an ERA5 API request for a specific date range and location.
        Group is in the form (year, months, days, hours).
        """
        Y, M, days, hours = group

        return APIRequest(
            year=Y,
            months=M,        # can be a list
            days=days,       # now a list
            times=hours,     # now a list
            coords=coords,
            era5_vars=era5_vars,
            monthly=monthly
        )

    @staticmethod
    def _extract_zip(zip_fp: str, unzip_fp: str) -> bool:
        """
        Extracts all files from a ZIP archive to a specified directory.
        """
        if not os.path.exists(zip_fp):
            print(f"Warning: ZIP file not found {zip_fp}, skipping extraction.")
            return False
        os.makedirs(unzip_fp, exist_ok=True)
        with zipfile.ZipFile(zip_fp, "r") as zp:
            try: 
                zp.extractall(unzip_fp)
                os.remove(zip_fp)
                return True
            except zipfile.error as e: 
                print(f"Failed to extract {zip_fp}: {e}")
                return False

    def _download_next_ready_result(
        self,
        pending_results: deque[tuple[APIRequest, object, str]],
        wait_for_transfer_slot: bool = True,
    ) -> None:
        """Download/extract the next ready ERA5 result once the transfer slot is free."""
        if not pending_results:
            return

        if wait_for_transfer_slot:
            self._wait_for_era5_transfer_slot()
        elif not self._era5_transfer_slot_available():
            return

        request, result, unzip_fp = pending_results.popleft()
        downloaded_zip_name = request.download_result(result, self.config.ZIP_DIR)
        downloaded_zip_fp = os.path.join(self.config.ZIP_DIR, downloaded_zip_name)
        if self._extract_zip(downloaded_zip_fp, unzip_fp):
            self.download_registry.mark_available(unzip_fp, kind="nc")
            self._mark_era5_chunk_ready(unzip_fp)

    def _wait_for_era5_transfer_slot(self) -> None:
        """Block until there is no ready ERA5 chunk waiting to be synced away."""
        while not self._era5_transfer_slot_available():
            print("Waiting for ERA5 sync to finish before downloading the next result...")
            time.sleep(self.ERA5_TRANSFER_POLL_SECONDS)

    def _era5_transfer_slot_available(self) -> bool:
        """Only allow one locally materialized ERA5 chunk awaiting sync at a time."""
        era5_root = Path(self.config.ERA5_DIR)
        if not era5_root.exists():
            return True
        return not any(era5_root.rglob(ERA5_TRANSFER_READY_SENTINEL))

    @staticmethod
    def _mark_era5_chunk_ready(unzip_fp: str) -> None:
        """Signal that an ERA5 chunk is fully extracted and ready to sync."""
        os.makedirs(unzip_fp, exist_ok=True)
        Path(unzip_fp, ERA5_TRANSFER_READY_SENTINEL).touch()
