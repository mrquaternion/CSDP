"""Preflight check for existing downloads over a given time range."""

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .argparser import ArgumentParserManager
from .config import CarbonPipelineConfig


class DownloadErrorCode(Enum):
    ALL_ALREADY_DOWNLOADED = 2000
    ERA5_ALREADY_DOWNLOADED = 2001
    CO2_ALREADY_DOWNLOADED = 2002
    WTD_ALREADY_DOWNLOADED = 2003


class DataSource(Enum):
    ERA5 = "ERA5 - Climate Data Store"
    CO2 = "Carbon Dioxide - Climate Data Store"
    WTD = "Water Table Depth - GLOBGM, Utrecht University"

    def __str__(self):
        return self.value


class DownloadPresenceError(RuntimeError):
    def __init__(self, code: DownloadErrorCode, message: str, status=None):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class SourceStatus:
    data_source: DataSource
    error_code: DownloadErrorCode
    is_downloaded: bool
    absolute_path: str


class DownloadPresenceChecker:
    """Verify if data has already been downloaded for the given predictors and time range."""

    def __init__(self, config_path: str | None = None):
        self.status: list[SourceStatus] = []
        self.config_path = config_path

    def validate(self) -> list[SourceStatus]:
        self._validate_era5()
        self._validate_co2()
        self._validate_wtd()
        return self.status

    def _validate_era5(self):
        era5_dir = Path(CarbonPipelineConfig.ERA5_DIR).resolve()
        has_data = era5_dir.exists() and any(era5_dir.rglob("*.nc"))
        self.status.append(
            SourceStatus(
                data_source=DataSource.ERA5,
                error_code=DownloadErrorCode.ERA5_ALREADY_DOWNLOADED,
                is_downloaded=has_data,
                absolute_path=str(era5_dir),
            )
        )

    def _validate_co2(self):
        co2_dir = Path(CarbonPipelineConfig.CO2_DIR).resolve()
        has_data = co2_dir.exists() and any(co2_dir.rglob("*"))
        self.status.append(
            SourceStatus(
                data_source=DataSource.CO2,
                error_code=DownloadErrorCode.CO2_ALREADY_DOWNLOADED,
                is_downloaded=has_data,
                absolute_path=str(co2_dir),
            )
        )

    def _validate_wtd(self):
        wtd_dir = Path(CarbonPipelineConfig.WTD_DIR).resolve()
        has_data = wtd_dir.exists() and any(wtd_dir.rglob("*.tif"))
        self.status.append(
            SourceStatus(
                data_source=DataSource.WTD,
                error_code=DownloadErrorCode.WTD_ALREADY_DOWNLOADED,
                is_downloaded=has_data,
                absolute_path=str(wtd_dir),
            )
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carbonpipeline-preflight",
        description="Validate a carbonpipeline config without running downloads.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = ArgumentParserManager.load_yaml_config(args.config)
    del config

    validator = DownloadPresenceChecker(config_path=args.config)

    status = validator.validate()

    already_downloaded = [s for s in status if s.is_downloaded]

    # everything is already there
    if len(already_downloaded) == len(status):
        raise DownloadPresenceError(
            DownloadErrorCode.ALL_ALREADY_DOWNLOADED,
            "All data sources are already downloaded. Nothing to do.",
            status=status,
        )

    # some data already exists
    for stat in already_downloaded:
        raise DownloadPresenceError(
            stat.error_code,
            f"{stat.data_source} data already exists at {stat.absolute_path}.",
            status=stat,
        )

    print("No existing data found. Safe to start fresh download.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DownloadPresenceError as exc:
        print(f"ERROR ({exc.code.value}): {exc}")
        raise SystemExit(exc.code.value)
