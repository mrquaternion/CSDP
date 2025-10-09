# CarbonSense Data Pipeline (CSDP)

`CS-Pipeline` is a command‑line workflow that enriches eddy-covariance (EC) station data with reanalysis variables from ERA5, and optionally gap‑fills AmeriFlux predictors. It also helps getting data to feed a neural network (previously used to analyze fires conditions across Canada). It operates in two main stages driven by YAML configuration files. It starts by querying data from the [Copernicus Data Store](https://cds.climate.copernicus.eu).

> **Note:** There is 2 possible use cases of the pipeline. Both download ERA5 data but use it differently.
> 
> The *first* is asking for a directory containing GeoJSON files. The goal of this use case is if you have a large area that contains polygons defining a
set of spatial regions (e.g., administrative boundaries, fire perimeters, or ecological zones), and you want to aggregate the ERA5 data over each polygon rather than at individual points.
> 
> The *second* is designed for point-based extraction. Instead of polygons, you provide a latitude/longitude coordinate (e.g. weather station, eddy covariance tower site, or random sampling location), 
> and the pipeline will download the ERA5 variables directly at this exact point without performing any spatial aggregation.

## Installation
First, `git clone` the project to your desired local directory and go to `pipeline/`. Once this is done, please run the following commands:
```commandline
conda env create -f environment.yaml
conda activate carbon
```
Easy as that! You are now all set up!

## Core workflow
**1. Prepare configuration**
- Create a YAML file (please use the same structure as in the repo, `download_config.yaml`) describing [the date range, target predictors, geographic footprint, aggregation level, and an optional field name to label features](#opts).
- Nothing as to be put in the `process_config.yaml` file *except* the desired prefix of your output file(s).

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

## Reproducible examples
### Regions bound by a polygon
Simply throw the command `carbonpipeline download --config examples/download_config_geojsons.yaml`. Once the download is finished,
you can now do the processing. Thus, running `carbonpipeline process --config examples/download_config_process.yaml`.

### Gap-filling eddy covariance site's data
Same as the previous processing type. Just change `download_config_geojsons` for `download_config_site` and `process_config_geojsons`
for `process_config_geojsons`.

## <a name="opts"></a> Configuration options

| Option             | Description                                                     | Processing type          | Pipeline step |
|--------------------|-----------------------------------------------------------------|--------------------------|---------------|
| `action`           | Stage to run (`download` or `process`)                          | N/A                      | N/A           |
| `output-filename`  | Prefix for processed NetCDF files                               | Polygons & site location | Process       |
| `start`, `end`     | ISO datetime range; must respect chosen aggregation granularity | Polygons & site location | Download      | 
| `preds`            | List of AmeriFlux predictors (TA, PA, etc.)                     | Polygons & site location | Download      |
| `coords-dir`       | Directory of GeoJSON features; omit for global coverage         | Polygons                 | Download      |
| `aggregation-type` | `DAILY`, `MONTHLY`, or omitted for hourly                       | Polygons                 | Download      |
| `id-field`         | Property name used to label each GeoJSON feature                | Polygons                 | Download      |
| `data-file`        | CSV file for gap‑filling missing values                         | Site location            | Download      |
| `location`         | Coordinates of the site location                                | Site location            | Download      |

## Known limitations
Saving a separate file for each region (polygon) is not always optimal, especially when dealing with more than a thousand regions. Depending on the size of each region, the storage requirements can become very large. For this reason, it is recommended to run the pipeline on **(1) an external hard drive** or **(2) a computing cluster with sufficient storage**.

Right now, the processing type "Site Location" cannot process multiple files at the same time. However, this could be changed in the future.

## Contributing
Contributions is welcome. I recognize that the code may not yet be perfectly structured and that documentation is still sparse. Any help in improving clarity, structure, and maintainability is especially appreciated.  

