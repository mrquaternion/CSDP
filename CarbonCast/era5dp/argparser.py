import argparse
import yaml

from pathlib import Path



class ArgumentParserManager: 
    """Handles command line argument parsing."""
    
    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        """Build the main argument parser."""
        parser = argparse.ArgumentParser(
            prog="carbonpipeline",
            description="Pipeline to retrieve and compute AmeriFlux variables based on ERA5 variables.",
            epilog="More information available on GitHub."
        )
        subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
        ArgumentParserManager._add_download_subparser(subparsers)
        ArgumentParserManager._add_process_subparser(subparsers)
        return parser
    
    @staticmethod
    def _add_download_subparser(subparsers):
        p = subparsers.add_parser("download", help="Downloads the required data depending on the configuration.")
        p.add_argument("--config", required=True, type=str, help="The configuration file containing the arguments.")

    @staticmethod
    def _add_process_subparser(subparsers):
        p = subparsers.add_parser("process", help="Process the downloaded data depending on the coordinates geometry.")
        p.add_argument("--config", required=True, type=str, help="The configuration file containing the arguments.")
    
    @staticmethod
    def load_yaml_config(pathstr: str) -> dict:
        path = Path(pathstr)
        with open(path, "r") as f:
            if path.suffix == ".yaml" or path.suffix == ".yml":
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Only .yaml or .yml config files are supported. Not {path.suffix}.")
    
    @staticmethod
    def pretty_print_inputs(title: str, **fields):
        print(f"\n------------------- {title.upper()} -------------------")
        for k, v in fields.items():
            print(f"- {k:<15}: {v}")
        print("----------------------------------------------------------\n")