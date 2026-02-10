"""Preflight check for existing downloads over a given time range."""

import argparse
from dataclasses import dataclass
from enum import Enum

from .argparser import ArgumentParserManager


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

    def __init__(self):
        self.status: list[SourceStatus] = []

    def validate(self) -> list[SourceStatus]:
        self._validate_era5()
        self._validate_co2()
        self._validate_wtd()
        return self.status

    def _validate_era5(self):
        pass

    def _validate_co2(self):
        pass

    def _validate_wtd(self):
        pass


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
    
    validator = DownloadPresenceChecker()

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