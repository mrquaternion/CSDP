# The code below is intended for use with EcoPerceiver. The purpose of creating this database is to facilitate the
# inference process by maintaining consistency with the input format used in EcoPerceiver’s training phase.

import os
import glob
import sqlite3
import rioxarray as rxr
import xarray as xr
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


# ======================== Database Constants ========================

DATA_PATH = 'data/'
DB_SCHEMA_PATH = 'carbonpipeline_db_struct.sql'
DB_NAME = 'data/era5.db'
DATA_TABLE = 'ec_data'
COORD_TABLE = 'coord_data'
IGBP_PATH = 'data/igbp.tiff'

EC_PREDICTORS = (
    'DOY', 'TOD', 'TA', 'P', 'RH', 'VPD', 'PA', 'CO2', 'SW_IN', 'SW_IN_POT',
    'SW_OUT', 'LW_IN', 'LW_OUT', 'NETRAD', 'PPFD_IN', 'PPFD_OUT',
    'WS', 'WD', 'USTAR', 'WTD', 'G', 'H', 'LE',
    'SWC_1', 'SWC_2', 'SWC_3', 'SWC_4', 'SWC_5',
    'TS_1', 'TS_2', 'TS_3', 'TS_4', 'TS_5',
)

IGBP_ACRONYMS = {
    0: 'WAT', 1: 'ENF', 2: 'EBF', 3: 'DNF', 4: 'DBF', 5: 'MF', 6: 'CSH',
    7: 'OSH', 8: 'WSA', 9: 'SAV', 10: 'GRA', 11: 'WET', 12: 'CRO',
    13: None, 14: 'CVM', 15: 'SNO', 16: None,
}



# ======================== Data Transfer ========================
def launch_sqlite():
    with open(DB_SCHEMA_PATH, 'r') as f:
        schema = f.read()

    sql = schema.format(
        table_name=DATA_TABLE,
        vars=', '.join([f'{c} REAL' for c in EC_PREDICTORS])
    )

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.executescript(sql)
    conn.commit()
    conn.close()

    print('~ The SQLite database has been initialized ~')


def port_netcdf_to_database():
    for path in glob.glob(os.path.join(DATA_PATH, '*.nc')):
        ds = xr.open_dataset(path)
        df = (ds.to_dataframe()
              .reset_index()
              .drop(columns=['region_id'])
              .rename(columns={
                'latitude': 'lat',
                'longitude': 'lon',
                'valid_time': 'timestamp',
                'LOCATION_ELEV': 'elev',
              })
            )
        
        df = add_doy_and_tod(df)
        df = add_igbp(df)
        df['timestamp'] = df['timestamp'].apply(format_datetime)

        coords = df[['lat', 'lon', 'elev', 'igbp']].drop_duplicates().reset_index(drop=True)
        coords['coord_id'] = coords.index + 1

        df = df.merge(coords, on=['lat', 'lon', 'elev', 'igbp'], how='left')

        with sqlite3.connect(DB_NAME) as conn:
            coords.to_sql(COORD_TABLE, conn, if_exists='replace', index=False)
            df[[ 'coord_id', 'timestamp', *EC_PREDICTORS ]].to_sql(DATA_TABLE, conn, if_exists='append', index=False)
            conn.commit()

        print(f'Inserted {len(df)} rows from {os.path.basename(path)}')



# ======================== Helpers ========================
def add_doy_and_tod(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['DOY'] = df['timestamp'].apply(lambda dt: dt.dayofyear)
    df_copy['TOD'] = df['timestamp'].apply(lambda dt: dt.hour + 1)
    return df_copy


def add_igbp(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    da = rxr.open_rasterio(IGBP_PATH).isel(band=0)
    df_igbp = pd.DataFrame({
        'x': np.repeat(da.x.values, len(da.y)),
        'y': np.tile(da.y.values, len(da.x)),
        'val': da.values.flatten()
    })

    tree = cKDTree(df_igbp[['x', 'y']])

    _, idx = tree.query(df_copy[['lon', 'lat']], k=1)
    igbp_vals = df_igbp.iloc[idx]['val'].values

    df_copy['igbp'] = [IGBP_ACRONYMS.get(v) for v in igbp_vals]
    return df_copy


def format_datetime(dt: pd.Timestamp) -> int:
    return int(dt.strftime('%Y%m%d%H%M%S'))


# ======================== Entry Point ========================
if __name__ == '__main__':
    launch_sqlite()
    port_netcdf_to_database()
