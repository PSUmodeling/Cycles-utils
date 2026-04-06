from __future__ import annotations
import geopandas as gpd
import pandas as pd
from pathlib import Path

_HERE = Path(__file__).parent.resolve()

STATE_CSV: Path = _HERE / '../data/us_states.csv'
COUNTY_CSV: Path = _HERE / '../data/fips_gid_conversion.csv'

GADM_LEVELS: dict[str, int] = {
    'country': 0,
    'state': 1,
    'county': 2,
}

STATE_DTYPES: dict[str, type] = {'state': str, 'gid': str, 'abbreviation': str, 'fips': int}
COUNTY_DTYPES: dict[str, type] = {'fips': int}


def _gadm_path(path: Path, country: str, level: int) -> Path:
    return path / f'gadm41_{country}_{level}.shp'


def _read_csv(fn: Path, dtypes: dict, index_col: str) -> pd.DataFrame:
    return pd.read_csv(fn, dtype=dtypes, index_col=index_col)


def _find_representation(csv: Path, dtypes: dict, representation: str, **kwargs) -> str | int:
    """Look up a representation value by trying each provided keyword argument."""
    for col, value in kwargs.items():
        if value is None:
            continue
        df = _read_csv(csv, dtypes, index_col=col)
        try:
            return df.loc[value, representation]
        except KeyError:
            continue
    raise KeyError(f'{representation.capitalize()} not found for: ' + ', '.join(f'{k}={v}' for k, v in kwargs.items() if v is not None))


def _find_county_name(csv: Path, dtypes: dict, **kwargs) -> str:
    """County name is a special case — composed from name_2 and name_1."""
    for col, value in kwargs.items():
        if value is None:
            continue
        df = _read_csv(csv, dtypes, index_col=col)
        try:
            return f'{df.loc[value, "name_2"]}, {df.loc[value, "name_1"]}'
        except KeyError:
            continue
    raise KeyError(
        'County name not found for: '
        + ', '.join(f'{k}={v}' for k, v in kwargs.items() if v is not None)
    )


def read_gadm(path: str | Path, country: str, level_str: str, *, conus: bool = True) -> gpd.GeoDataFrame:
    level = GADM_LEVELS[level_str.lower()]
    gdf   = gpd.read_file(_gadm_path(Path(path), country, level))

    if country != 'global':
        gdf.rename(columns={f'GID_{level}': 'GID'}, inplace=True)
    gdf.set_index('GID', inplace=True)

    if country == 'USA' and conus:
        gdf = gdf[~gdf['NAME_1'].isin(['Alaska', 'Hawaii'])]

    return gdf


def state_gid(*, state: str | None = None, abbreviation: str | None = None, fips: int | None = None) -> str:
    return str(_find_representation(STATE_CSV, STATE_DTYPES, 'gid', state=state, abbreviation=abbreviation, fips=fips))


def state_abbreviation(*, state: str | None = None, gid: str | None = None, fips: int | None = None) -> str:
    return str(_find_representation(STATE_CSV, STATE_DTYPES, 'abbreviation', state=state, gid=gid, fips=fips))


def state_fips(*, state: str | None = None, abbreviation: str | None = None, gid: str | None = None) -> int:
    return int(_find_representation(STATE_CSV, STATE_DTYPES, 'fips', state=state, abbreviation=abbreviation, gid=gid))


def state_name(*, abbreviation: str | None = None, gid: str | None = None, fips: int | None = None) -> str:
    return str(_find_representation(STATE_CSV, STATE_DTYPES, 'state', abbreviation=abbreviation, gid=gid, fips=fips))


def county_gid(*, fips: int) -> str:
    return str(_find_representation(COUNTY_CSV, COUNTY_DTYPES, 'gid', fips=fips))


def county_fips(*, gid: str) -> int:
    return int(_find_representation(COUNTY_CSV, COUNTY_DTYPES, 'fips', gid=gid))


def county_name(*, gid: str | None = None, fips: int | None = None) -> str:
    return str(_find_county_name(COUNTY_CSV, COUNTY_DTYPES, gid=gid, fips=fips))
