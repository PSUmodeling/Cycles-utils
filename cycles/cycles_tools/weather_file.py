import pandas as pd
from pathlib import Path

DAILY_COLUMNS: dict[str, type] = {
    'YEAR':  int,
    'DOY':   int,
    'PP':    float,
    'TX':    float,
    'TN':    float,
    'SOLAR': float,
    'RHX':   float,
    'RHN':   float,
    'WIND':  float,
}

SUBDAILY_COLUMNS: dict[str, type] = {
    'YEAR':  int,
    'DOY':   int,
    'HOUR':  int,
    'PP':    float,
    'TMP':   float,
    'SOLAR': float,
    'RH':    float,
    'WIND':  float,
}

WEATHER_HEADER_LINES: int = 4

def _build_date_index(df: pd.DataFrame, *, subdaily: bool) -> pd.DatetimeIndex:
    base = df['YEAR'].astype(str) + '-' + df['DOY'].astype(str)
    if subdaily:
        return pd.to_datetime(base + ' ' + df['HOUR'].astype(str), format='%Y-%j %H')
    return pd.to_datetime(base, format='%Y-%j')


def read_weather_file(fn: str | Path, *, start_year: int = 0, end_year: int = 9999, subdaily: bool = False) -> pd.DataFrame:
    columns = SUBDAILY_COLUMNS if subdaily else DAILY_COLUMNS
    df = pd.read_csv(
        Path(fn),
        usecols=list(range(len(columns))),
        names=list(columns.keys()),
        comment='#',
        sep=r'\s+',
        na_values='-999',
        skiprows=WEATHER_HEADER_LINES,
    )
    df = df.astype(columns)
    df.index = _build_date_index(df, subdaily=subdaily)
    df.index.name = 'date'
    return df[(df['YEAR'] >= start_year) & (df['YEAR'] <= end_year)]
