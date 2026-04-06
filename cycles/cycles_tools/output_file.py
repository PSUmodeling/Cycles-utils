import pandas as pd
from pathlib import Path

DATE_COLUMNS: frozenset[str] = frozenset({'date', 'plant_date'})

def _parse_units(lines: list[str], columns) -> dict[str, str]:
    """Find the unit comment line and map column names to units."""
    unit_line = next(
        line.strip().lstrip('#')
        for line in lines
        if line.strip().startswith('#') and ',' in line
    )
    return dict(zip(columns, [u.strip() for u in unit_line.split(',')]))


def read_output(path: str | Path, output_type: str) -> tuple[pd.DataFrame, dict[str, str]]:
    fn = Path(path) / f'{output_type}.csv'
    lines = fn.read_text().splitlines()

    df = pd.read_csv(fn, comment='#')
    for col in DATE_COLUMNS & set(df.columns):
        df[col] = pd.to_datetime(df[col])

    units = _parse_units(lines, df.columns)

    return df, units
