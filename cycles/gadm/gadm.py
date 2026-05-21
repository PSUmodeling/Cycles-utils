from __future__ import annotations
import geopandas as gpd
import pandas as pd
from enum import Enum
from pathlib import Path

_HERE = Path(__file__).parent.resolve()

STATE_CSV: Path = _HERE / '../data/us_states.csv'
COUNTY_CSV: Path = _HERE / '../data/fips_gid_conversion.csv'

class GADMLevel(Enum):
    COUNTRY = 0
    STATE = 1
    COUNTY = 2

STATE_DTYPES: dict[str, type] = {'state': str, 'gid': str, 'abbreviation': str, 'fips': int}
COUNTY_DTYPES: dict[str, type] = {'fips': int}

def _gadm_path(path: Path, country: str, level: GADMLevel) -> Path:
    return path / f'gadm41_{country}_{level.value}.shp'


def _read_csv(fn: Path, dtypes: dict, index_col: str) -> pd.DataFrame:
    return pd.read_csv(fn, dtype=dtypes, index_col=index_col)


def _find_representation(csv: Path, dtypes: dict, representation: str, **kwargs) -> str | int:
    for col, value in kwargs.items():
        if value is None:
            continue
        df = _read_csv(csv, dtypes, index_col=col)
        try:
            return df.loc[value, representation]    # type: ignore
        except KeyError:
            continue
    raise KeyError(f'{representation.capitalize()} not found for: ' + ', '.join(f'{k}={v}' for k, v in kwargs.items() if v is not None))


def _find_county_name(csv: Path, dtypes: dict, **kwargs) -> str:
    # County name is a special case — composed from name_2 and name_1
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
    """Read a GADM layer and normalize its index.

    Args:
        path: Directory containing GADM shapefiles.
        country: Country code used in GADM file names.
        level_str: Administrative level name (country, state, county).
        conus: For USA state/county layers, exclude Alaska and Hawaii.

    Returns:
        GeoDataFrame indexed by GID.
    """
    level = GADMLevel[level_str.upper()]
    gdf = gpd.read_file(_gadm_path(Path(path), country, level))

    if country != 'global':
        gdf.rename(columns={f'GID_{level.value}': 'GID'}, inplace=True)
    gdf.set_index('GID', inplace=True)

    if country == 'USA' and conus:
        gdf = gdf[~gdf['NAME_1'].isin(['Alaska', 'Hawaii'])]

    return gdf


def state_gid(*, state: str | None = None, abbreviation: str | None = None, fips: int | None = None) -> str:
    """Look up state GID by name, abbreviation, or FIPS code.

    Args:
        state: Full state name.
        abbreviation: Two-letter state abbreviation.
        fips: Numeric state FIPS code.

    Returns:
        State GID string.

    Raises:
        KeyError: If no matching state record is found.
    """
    return str(_find_representation(STATE_CSV, STATE_DTYPES, 'gid', state=state, abbreviation=abbreviation, fips=fips))


def state_abbreviation(*, state: str | None = None, gid: str | None = None, fips: int | None = None) -> str:
    """Look up state abbreviation by name, GID, or FIPS code.

    Args:
        state: Full state name.
        gid: State GID string.
        fips: Numeric state FIPS code.

    Returns:
        Two-letter state abbreviation.

    Raises:
        KeyError: If no matching state record is found.
    """
    return str(_find_representation(STATE_CSV, STATE_DTYPES, 'abbreviation', state=state, gid=gid, fips=fips))


def state_fips(*, state: str | None = None, abbreviation: str | None = None, gid: str | None = None) -> int:
    """Look up state FIPS code by name, abbreviation, or GID.

    Args:
        state: Full state name.
        abbreviation: Two-letter state abbreviation.
        gid: State GID string.

    Returns:
        Numeric state FIPS code.

    Raises:
        KeyError: If no matching state record is found.
    """
    return int(_find_representation(STATE_CSV, STATE_DTYPES, 'fips', state=state, abbreviation=abbreviation, gid=gid))


def state_name(*, abbreviation: str | None = None, gid: str | None = None, fips: int | None = None) -> str:
    """Look up state name by abbreviation, GID, or FIPS code.

    Args:
        abbreviation: Two-letter state abbreviation.
        gid: State GID string.
        fips: Numeric state FIPS code.

    Returns:
        Full state name.

    Raises:
        KeyError: If no matching state record is found.
    """
    return str(_find_representation(STATE_CSV, STATE_DTYPES, 'state', abbreviation=abbreviation, gid=gid, fips=fips))


def county_gid(*, fips: int) -> str:
    """Look up county GID by county FIPS code.

    Args:
        fips: Numeric county FIPS code.

    Returns:
        County GID string.

    Raises:
        KeyError: If no matching county record is found.
    """
    return str(_find_representation(COUNTY_CSV, COUNTY_DTYPES, 'gid', fips=fips))


def county_fips(*, gid: str) -> int:
    """Look up county FIPS code by county GID.

    Args:
        gid: County GID string.

    Returns:
        Numeric county FIPS code.

    Raises:
        KeyError: If no matching county record is found.
    """
    return int(_find_representation(COUNTY_CSV, COUNTY_DTYPES, 'fips', gid=gid))


def county_name(*, gid: str | None = None, fips: int | None = None) -> str:
    """Look up county display name by GID or FIPS code.

    Args:
        gid: County GID string.
        fips: Numeric county FIPS code.

    Returns:
        County display name formatted as "County, State".

    Raises:
        KeyError: If no matching county record is found.
    """
    return str(_find_county_name(COUNTY_CSV, COUNTY_DTYPES, gid=gid, fips=fips))
