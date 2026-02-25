# CarbonCast

`CarbonCast` is a command‑line workflow that enriches eddy-covariance (EC) station data with reanalysis variables from ERA5, and optionally gap‑fills AmeriFlux predictors. It also helps getting data to feed a neural network (previously used to analyze fires conditions across Canada). It operates in two main stages driven by YAML configuration files. It starts by querying data from the [Copernicus Data Store](https://cds.climate.copernicus.eu).

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
conda activate ccenv
```
Easy as that! You are now all set up!

## CarbonCast Setup (CLI/Web)

#### Step 1: SSH connection requirements
1. Know the name of the machine you want to connect to.
2. Know your username.
3. Know your password or have an SSH key (see Step 2).
4. Be registered for MFA (Duo Mobile is the recommended method).

#### Step 2: Generate SSH key (skip if password auth)
Reference: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
1. Generate a new SSH key if you do not already have one.
2. Run `ssh-keygen -t ed25519 -C "your_email@example.com"` (replace with your email).
3. Press Enter to accept the default location.
4. Set a secure passphrase you can remember.
5. Start the agent: `eval "$(ssh-agent -s)"`
6. Add your key: `ssh-add ~/.ssh/id_ed25519`
7. Enter your passphrase once.
8. Continue to Step 3.

#### Step 3: Add SSH key to DRAC / Alliance (skip if password auth)
Reference: https://docs.alliancecan.ca/wiki/SSH_Keys
1. Print and copy your public key: `cat ~/.ssh/id_ed25519.pub`
2. Sign in to https://ccdb.alliancecan.ca/security/login
3. Go to `My Account` -> `SSH Keys`.
4. Paste the key, add a description, and confirm.
5. Wait a few minutes for propagation.

#### Step 4: Setup CDS API key
Reference: https://cds.climate.copernicus.eu/how-to-api
1. Configure your CDS API key (no package install required beyond the normal app environment).

#### Step 5: Recommended SSH config for Alliance clusters
If passphrase prompts keep appearing in the web workflow, ensure keychain settings apply to your cluster host (not only GitHub):

```sshconfig
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519

Host alliance-cluster
  HostName <cluster_host>   # e.g. narval.alliancecan.ca 
  User <your_alliance_username>
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
  ControlMaster auto
  ControlPersist 10m
  ControlPath /tmp/csdp-ssh-%C

Host *
  ServerAliveInterval 60
  ServerAliveCountMax 5
```

Then run once:

```commandline
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
ssh alliance-cluster
```

## CLI workflow (locally)
Use this when you want to run everything locally from terminal commands and config files.

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

## Web workflow (cluster-connected)
Use this when you want to drive the pipeline from the Flask UI and sync data/jobs with a cluster (for example, with a valid Compute Canada account).

1. Start the web app:
```commandline
flask --app web.app run
```
2. Fill the forms in the UI (query type, area type, configuration, credentials).
3. Launch the remote monitoring workflow:
- download step (with continuous sync to cluster storage),
- post-processing step (submitted and monitored on the cluster),
- output sync back to local `outputs/`.

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