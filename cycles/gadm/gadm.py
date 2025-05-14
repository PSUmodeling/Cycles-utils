import geopandas as gpd
import os
import pandas as pd

pt = os.path.dirname(os.path.realpath(__file__))

GADM = lambda path, country, level: f'{path}/gadm41_{country}_{level}.shp'
GADM_LEVELS = {
    'country': 0,
    'state': 1,
    'county': 2,
}
STATE_CSV = os.path.join(pt, '../data/us_states.csv')


def read_gadm(path: str, country: str, level: str, conus: bool=True) -> gpd.GeoDataFrame:
    level = GADM_LEVELS[level.lower()]
    gdf = gpd.read_file(GADM(path, country, level))
    gdf.rename(columns={f'GID_{level}': 'GID'}, inplace=True)
    gdf.set_index('GID', inplace=True)

    # Generate a CONUS GeoDataFrame by removing Alaska and Hawaii
    return gdf[~gdf['NAME_1'].isin(['Alaska', 'Hawaii'])] if country == 'USA' and conus else gdf


def _read_state_csv(index_col: str) -> pd.DataFrame:
    return pd.read_csv(
        STATE_CSV,
        dtype={'state': str, 'gid': str, 'abbreviation': str, 'fips': int},
        index_col=index_col,
    )


def state_gid(state: str=None, abbreviation: str=None, fips: int=None) -> str:
    for name, value in locals().items():
        if value is None: continue

        df = _read_state_csv(name)
        try:
            return df.loc[value, 'gid']
        except:
            raise KeyError(f'GID for {name} {value} cannot be found.')


def state_abbreviation(gid: str=None, state: str=None, fips: int=None) -> str:
    for name, value in locals().items():
        if value is None: continue

        df = _read_state_csv(name)
        try:
            return df.loc[value, 'abbreviation']
        except:
            raise KeyError(f'State abbreviation for {name} {value} cannot be found.')


def state_fips(gid: str=None, state: str=None, abbreviation: str=None) -> int:
    for name, value in locals().items():
        if value is None: continue

        df = _read_state_csv(name)
        try:
            return df.loc[value, 'fips']
        except:
            raise KeyError(f'FIPS for {name} {value} cannot be found.')


def state_name(gid: str=None, abbreviation: str=None, fips: int=None) -> str:
    for name, value in locals().items():
        if value is None: continue

        df = _read_state_csv(name)
        try:
            return df.loc[value, 'state']
        except:
            raise KeyError(f'State name for {name} {value} cannot be found.')
