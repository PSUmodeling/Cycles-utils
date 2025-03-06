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


def read_season(cycles_path, simulation):
    '''Read season output file for harvested crop, harvest time, plant time, and yield
    '''
    df = pd.read_csv(
        f'{cycles_path}/output/{simulation}/harvest.txt',
        sep='\t',
        header=0,
        skiprows=[1],
        skipinitialspace=True,
    )
    df = df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))
    df['crop'] = df['crop'].str.strip()

    for col in ['date', 'plant_date']: df[col] = pd.to_datetime(df[col])

    return df


def read_operations(cycles_path, operation):
    with open(f'{cycles_path}/input/{operation}.operation') as f:
        lines = f.read().splitlines()

    lines = [line for line in lines if (not line.strip().startswith('#')) and len(line.strip()) > 0]

    operations = []
    k = 0
    while k < len(lines):
        if lines[k] == 'FIXED_FERTILIZATION':
            operations.append(
                {
                    'type': 'fertilization',
                    'year': int(lines[k + 1].split()[1]),
                    'doy': int(lines[k + 2].split()[1]),
                    'source': lines[k + 3].split()[1],
                    'mass': lines[k + 4].split()[1],
                }
            )
            k += 5
        elif lines[k] == 'TILLAGE':
            if lines[k + 3].split()[1].strip().lower() in HARVEST_TOOLS:
                operations.append(
                    {
                        'type': 'harvest',
                        'year': int(lines[k + 1].split()[1]),
                        'doy': int(lines[k + 2].split()[1]),
                        'crop': lines[k + 7].split()[1],
                    }
                )
            elif lines[k + 3].split()[1].strip().lower() == 'kill_crop':
                operations.append(
                    {
                        'type': 'kill',
                        'year': int(lines[k + 1].split()[1]),
                        'doy': int(lines[k + 2].split()[1]),
                        'crop': lines[k + 7].split()[1],
                    }
                )
            else:
                operations.append(
                    {
                        'type': 'tillage',
                        'year': int(lines[k + 1].split()[1]),
                        'doy': int(lines[k + 2].split()[1]),
                        'tool': lines[k + 3].split()[1],
                    }
                )
            k += 8
        elif lines[k] == 'PLANTING':
            operations.append(
                {
                    'type': 'planting',
                    'year': int(lines[k + 1].split()[1]),
                    'doy': int(lines[k + 2].split()[1]),
                    'crop': lines[k + 8].split()[1],
                }
            )
            k += 9
        else:
            k += 1

    df = pd.DataFrame(operations)

    return df


def read_weather(cycles_path, weather, start_year=0, end_year=9999):
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
        f,
        usecols=list(range(len(columns))),
        names=columns.keys(),
        comment='#',
        sep='\s+',
        na_values=[-999],
    )
    df = df.iloc[NUM_HEADER_LINES:, :]
    df = df.astype(columns)

    return df[(df['YEAR'] <= end_year) & (df['YEAR'] >= start_year)]
