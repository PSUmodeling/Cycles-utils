from __future__ import annotations
import geopandas as gpd
import pandas as pd
import re
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


def _read_csv(fn: Path, dtypes: dict, index_col: str | None) -> pd.DataFrame:
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


# Matches a trailing county-equivalent designation (e.g. "Centre County" -> "Centre") so lookups accept both the bare
# name and the full GADM/Census "place name" style, case-insensitively.
_COUNTY_SUFFIX_RE = re.compile(r'\s+(county|parish|borough|census area|municipality)$', re.IGNORECASE)


def _resolve_state_name(state: str) -> str:
    """Normalize a state token to its full name, accepting a 2-letter abbreviation too."""
    state = state.strip()
    if len(state) == 2 and state.isalpha():
        return state_name(abbreviation=state.upper())
    return state


def _split_county_state(name: str) -> tuple[str, str]:
    try:
        county, state = name.split(',', 1)
    except ValueError:
        raise ValueError(f"Expected a 'County, State' formatted name, got: {name!r}")
    return county.strip(), _resolve_state_name(state.strip())


def _find_county_by_name(csv: Path, dtypes: dict, name: str) -> pd.Series:
    """Look up a county row by a 'County, State' formatted name.

    Both the county and state tokens are matched case-insensitively.
    The county token may optionally include a trailing designation
    ("Centre" or "Centre County" both match), and the state token may be
    either the full state name or its 2-letter abbreviation (e.g. both
    'Centre, PA' and 'Centre, Pennsylvania' resolve to the same county).

    Args:
        csv: Path to the county FIPS/GID conversion CSV.
        dtypes: Column dtypes to apply when reading the CSV.
        name: County name formatted as 'County, State'.

    Returns:
        The matching row as a pandas Series.

    Raises:
        ValueError: If `name` isn't in 'County, State' format.
        KeyError: If no county, or more than one county, matches.
    """
    county, state = _split_county_state(name)
    county = _COUNTY_SUFFIX_RE.sub('', county).strip()

    df = _read_csv(csv, dtypes, index_col=None)
    match = df[(df['name_2'].str.casefold() == county.casefold()) & (df['name_1'].str.casefold() == state.casefold())]

    if match.empty:
        raise KeyError(f'County not found for name={name!r}')
    if len(match) > 1:
        raise KeyError(f'Multiple counties matched for name={name!r}')
    return match.iloc[0]


def read_gadm(path: str | Path, country: str, level_str: str, *, conus: bool=True) -> gpd.GeoDataFrame:
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


def state_gid(*, state: str | None=None, abbreviation: str | None=None, fips: int | None=None) -> str:
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


def state_abbreviation(*, state: str | None=None, gid: str | None=None, fips: int | None=None) -> str:
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


def state_fips(*, state: str | None=None, abbreviation: str | None=None, gid: str | None=None) -> int:
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


def state_name(*, abbreviation: str | None=None, gid: str | None=None, fips: int | None=None) -> str:
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


def county_gid(*, fips: int | None=None, name: str | None=None) -> str:
    """Look up county GID by county FIPS code or by name.

    Args:
        fips: Numeric county FIPS code.
        name: County name formatted as 'County, State', e.g. 'Centre, PA' or 'Centre, Pennsylvania' (county suffix and
            state form are both optional/interchangeable; matching is case-insensitive).

    Returns:
        County GID string.

    Raises:
        ValueError: If `name` isn't in 'County, State' format.
        KeyError: If no matching county record is found.
    """
    if name is not None:
        return str(_find_county_by_name(COUNTY_CSV, COUNTY_DTYPES, name)['gid'])
    return str(_find_representation(COUNTY_CSV, COUNTY_DTYPES, 'gid', fips=fips))


def county_fips(*, gid: str | None=None, name: str | None=None) -> int:
    """Look up county FIPS code by county GID or by name.

    Args:
        gid: County GID string.
        name: County name formatted as 'County, State', e.g. 'Centre, PA' or 'Centre, Pennsylvania' (county suffix and
            state form are both optional/interchangeable; matching is case-insensitive).

    Returns:
        Numeric county FIPS code.

    Raises:
        ValueError: If `name` isn't in 'County, State' format.
        KeyError: If no matching county record is found.
    """
    if name is not None:
        return int(_find_county_by_name(COUNTY_CSV, COUNTY_DTYPES, name)['fips'])
    return int(_find_representation(COUNTY_CSV, COUNTY_DTYPES, 'fips', gid=gid))


def county_name(*, gid: str | None=None, fips: int | None=None) -> str:
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
