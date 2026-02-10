"""Configuration constants for the carbon pipeline."""
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class CarbonPipelineConfig:
    """Configuration constants for the carbon pipeline."""
    ZIP_DIR = "./datasets/zip"
    UNZIP_DIR = "./datasets/unzip"
    ERA5_DIR = os.path.join(UNZIP_DIR, "ERA5")
    CO2_DIR = os.path.join(UNZIP_DIR, "CO2")
    WTD_DIR = os.path.join(UNZIP_DIR, "WTD")
    DATETIME_FMT = r"%Y-%m-%d %H:%M:%S" # Corrected format to match strptime
    OUTPUT_PROCESSED_DIR = "./outputs"
    CO2_FOLDERNAME = "co2_CDS_2003-2022"
    WTD_URL = "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/output/version_1.0/transient_1958-2015/"
    WTD_RES = 30/3600  # arcsec to degree
    ERA5_RES = 0.25  # degree
