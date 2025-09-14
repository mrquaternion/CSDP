# CS-Pipeline overview

`CS-Pipeline` is a command‑line workflow that enriches eddy-covariance (EC) station data with reanalysis variables from ERA5, and optionally gap‑fills AmeriFlux predictors. It also helps getting data to feed a neural network (previously used to analyze fires conditions across Canada). It operates in two main stages driven by YAML configuration files. It starts by querying data from the [Copernicus Data Store](https://cds.climate.copernicus.eu).

> **Note:** The most recent development is happening on the [`non-interactive`](https://github.com/mrquaternion/CS-Pipeline/tree/non-interactive) branch.  
> This version removes runtime prompts and focuses on fully automated workflows.  
> If you’re looking for the latest features and improvements, please check out that branch.

## Install
The `carbonpipeline` package can be installed in the following way:

1. Clone the repository on your computer
2. Navigate to the root of the project
3. Create a virtual environment and activate it
4. Make sure your CDS API credentials are set up in `~/.cdsapirc`, otherwise follow this instruction https://arc.net/l/quote/ilmawrkf
5. Finally, enter the command pip install -e . in your CLI
6. Use carbonpipeline --help for more info!

## Core workflow
**1. Prepare configuration**
- Create a YAML file (please use the same structure as in the repo, `download_config.yaml`) describing [the date range, target predictors, geographic footprint, aggregation level, and an optional field name to label features](#opts).
- When gap-filling a specific station, supply a CSV file _only_ in the `process_config.yaml`.

**2. Download ERA5 (and optional datasets)**
- Run `carbonpipeline download --config download_config.yaml`
- For each region, the pipeline:
    - Builds grouped CDS API requests (hourly, daily or monthly, depending on the date range).
    - Asynchronously fetches ERA5 NetCDF files, storing them under `datasets/`.
    - Automatically retrieves CO<sub>2</sub> or WTD products when those predictors are requested.
- A `manifest.json` records geometry, predictors, and file locations for the subsequent processing step.

**3. Process and convert**
- Run `carbonpipeline process --config process_config.yaml`
- Using the manifest, the pipeline:
    - Merges NetCDF files, renames ERA5 shortnames, and appends CO<sub>2</sub>/WTD layers when available.
    - Clips data to each region's bounding box and converts ERA5 variables into AmeriFlux predictors with the `carbonepipeline/Processing/processing_utils.py` script.
    - Writes one NetCDF per region in `outputs/`, with optional daily or monthly aggregation.
    - If a CSV was supplied, then add a new column for each requested predictors with the corresponding data.

## <a name="opts"></a> Configuration options 

| Option            | Description                                                                 | Status                                                     |
|-------------------|-----------------------------------------------------------------------------|------------------------------------------------------------|
| `action`          | Stage to run (`download` or `process`)                                      | Fully supported                                            |
| `start`, `end`    | ISO datetime range; must respect chosen aggregation granularity             | Fully supported                                            |
| `preds`           | List of AmeriFlux predictors (TA, PA, etc.)                                 | Mature                       |
| `coords-dir`      | Directory of GeoJSON features; omit for global coverage                     | Fully supported                                            |
| `aggregation-type`| `DAILY`, `MONTHLY`, or omitted for hourly                                   | Works well regionally; global runs not supported            |
| `id-field`        | Property name used to label each GeoJSON feature                            | Supported                                                   |
| `data-file`       | CSV file for gap‑filling missing values                                     | Experimental & non‑interactive in v0.1.0                   |
| `output-filename` | Prefix for processed NetCDF files                                           | Supported                                                   |

## Known limitations
- CSV gap-filling (`data-file` option) is still experimental and lacks full validation in the latest release.
- v1.0.0 runs non‑interactively, i.e. no prompts or checkpoints during execution.
- Saving a separate file for each region (polygon) is not always optimal, especially when dealing with more than a thousand regions. Depending on the size of each region, the storage requirements can become very large. For this reason, it is recommended to run the pipeline on **(1) an external hard drive** or **(2) a computing cluster with sufficient storage**.

## Contributing

Contributions are very welcome!  
If you encounter issues, have ideas for new features, or would like to improve the documentation, feel free to open an issue or submit a pull request.  

I recognize that the code may not yet be perfectly structured and that documentation is still sparse. Any help in improving clarity, structure, and maintainability is especially appreciated.  

