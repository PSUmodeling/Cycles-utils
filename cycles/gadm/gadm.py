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
COUNTY_CSV = os.path.join(pt, '../data/fips_gid_conversion.csv')

def read_gadm(path: str, country: str, level_str: str, *, conus: bool=True) -> gpd.GeoDataFrame:
    level = GADM_LEVELS[level_str.lower()]
    gdf = gpd.read_file(GADM(path, country, level))
    gdf.rename(columns={f'GID_{level}': 'GID'}, inplace=True)
    gdf.set_index('GID', inplace=True)

    return gdf[~gdf['NAME_1'].isin(['Alaska', 'Hawaii'])] if country == 'USA' and conus else gdf


def _read_state_csv(index_col: str) -> pd.DataFrame:
    return pd.read_csv(
        STATE_CSV,
        dtype={'state': str, 'gid': str, 'abbreviation': str, 'fips': int},
        index_col=index_col,
    )


def _find_state_representation(representation: str, **kwargs) -> str | int:
    for name, value in kwargs.items():
        if value is None: continue

        df = _read_state_csv(name)
        try:
            return df.loc[value, representation]    # type: ignore
        except KeyError:
            pass

    raise KeyError(f'{representation.capitalize()} for {name} {value} cannot be found.')


def state_gid(*, state: str | None=None, abbreviation: str | None=None, fips: int | None=None) -> str:
    return _find_state_representation('gid', state=state, abbreviation=abbreviation, fips=fips) # type: ignore


def state_abbreviation(*, state: str | None=None, gid: str | None=None, fips: int | None=None) -> str:
    return _find_state_representation('abbreviation', state=state, gid=gid, fips=fips)  # type: ignore


def state_fips(*, state: str | None=None, abbreviation: str | None=None, gid: str | None=None) -> int:
    return _find_state_representation('fips', state=state, abbreviation=abbreviation, gid=gid)  # type: ignore


def state_name(*, abbreviation: str | None=None, gid: str | None=None, fips: int | None=None) -> str:
    return _find_state_representation('state', abbreviation=abbreviation, gid=gid, fips=fips)   # type: ignore


def _read_county_csv(index_col: str) -> pd.DataFrame:
    return pd.read_csv(
        COUNTY_CSV,
        dtype={'fips': int},
        index_col=index_col,
    )


def _find_county_representation(representation: str, **kwargs) -> str | int:
    for name, value in kwargs.items():
        if value is None: continue

        df = _read_county_csv(name)

        if representation == 'name':
            try:
                return f'{df.loc[value, "name_2"]}, {df.loc[value, "name_1"]}'
            except KeyError:
                pass

        try:
            return df.loc[value, representation]    # type: ignore
        except KeyError:
            pass

    raise KeyError(f'{representation.capitalize()} for {name} {value} cannot be found.')


def county_gid(*, fips: int) -> str:
    return _find_county_representation('gid', fips=fips)    # type: ignore


def county_fips(*, gid: str) -> int:
    return _find_county_representation('fips', gid=gid) # type: ignore

def county_name(*, gid: str | None=None, fips: int | None=None) -> str:
    return _find_county_representation('name', gid=gid, fips=fips)  # type: ignore
