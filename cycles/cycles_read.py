import pandas as pd

HARVEST_TOOLS = [
    'grain_harvest',
    'harvest_grain',
    'grainharvest',
    'harvestgrain',
    'forage_harvest',
    'harvest_forage',
    'forageharvest',
    'harvestforage',
]


def read_output(cycles_path: str, simulation: str, output: str) -> tuple[pd.DataFrame, dict]:
    '''Read harvest output file for harvested crops, harvest , plan dates, and yield
    '''
    df = pd.read_csv(
        f'{cycles_path}/output/{simulation}/{output}.csv',
        comment='#',
    )

    for col in ['date', 'plant_date']:
        if col in df.columns: df[col] = pd.to_datetime(df[col])

    with open(f'{cycles_path}/output/{simulation}/{output}.csv') as f:
        lines = f.readlines()

    units = {col: lines[1].strip()[1:].split(',')[ind] for ind, col in enumerate(df.columns)}

    return df, units


def _read_operation_parameter(type: type, line_no: int, lines: list[str]) -> str:
    return type(lines[line_no].split()[1])


def read_operations(cycles_path: str, operation: str) -> pd.DataFrame:
    with open(f'{cycles_path}/input/{operation}.operation') as f:
        lines = f.read().splitlines()

    lines = [line for line in lines if (not line.strip().startswith('#')) and len(line.strip()) > 0]

    operations = []
    k = 0
    while k < len(lines):
        match lines[k]:
            case 'FIXED_FERTILIZATION':
                operations.append({
                    'type': 'fertilization',
                    'year': _read_operation_parameter(int, k + 1, lines),
                    'doy': _read_operation_parameter(int, k + 2, lines),
                    'source': _read_operation_parameter(str, k + 3, lines),
                    'mass': _read_operation_parameter(float, k + 4, lines),
                })
                k += 5
            case 'TILLAGE':
                tool = _read_operation_parameter(str, k + 3, lines)
                year = _read_operation_parameter(int, k + 1, lines)
                doy = _read_operation_parameter(int, k + 2, lines)
                crop = _read_operation_parameter(str, k + 7, lines)

                if tool.strip().lower() in HARVEST_TOOLS:
                    operations.append({
                        'type': 'harvest',
                        'year': year,
                        'doy': doy,
                        'crop': crop,
                    })
                elif tool.strip().lower() == 'kill_crop':
                    operations.append({
                        'type': 'kill',
                        'year': year,
                        'doy': doy,
                        'crop': crop,
                    })
                else:
                    operations.append({
                        'type': 'tillage',
                        'year': year,
                        'doy': doy,
                        'tool': tool,
                    })
                k += 8
            case 'PLANTING':
                operations.append({
                    'type': 'planting',
                    'year': _read_operation_parameter(int, k + 1, lines),
                    'doy': _read_operation_parameter(int, k + 2, lines),
                    'crop': _read_operation_parameter(str, k + 8, lines),
                })
                k += 9
            case _:
                k += 1

    df = pd.DataFrame(operations)

    return df


def read_weather(cycles_path: str, weather: str, *, start_year: int=0, end_year: int=9999) -> pd.DataFrame:
    NUM_HEADER_LINES = 4
    columns = {
        'YEAR': int,
        'DOY': int,
        'PP': float,
        'TX': float,
        'TN': float,
        'SOLAR': float,
        'RHX': float,
        'RHN': float,
        'WIND': float,
    }
    df = pd.read_csv(
        f'{cycles_path}/input/{weather}',
        usecols=list(range(len(columns))),
        names=list(columns.keys()),
        comment='#',
        sep=r'\s+',
        na_values='-999',
    )
    df = df.iloc[NUM_HEADER_LINES:, :]
    df = df.astype(columns)
    df['date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['DOY'].astype(str), format='%Y-%j')
    df.set_index('date', inplace=True)

    return df[(df['YEAR'] <= end_year) & (df['YEAR'] >= start_year)]
