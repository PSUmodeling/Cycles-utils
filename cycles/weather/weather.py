import geopandas as gpd
import math
import numpy as np
import os
import pandas as pd
import subprocess
from datetime import datetime, timedelta
from netCDF4 import Dataset

pt = os.path.dirname(os.path.realpath(__file__))

URLS = {
    'GLDAS': 'https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_3H.2.1',
    'gridMET': 'http://www.northwestknowledge.net/metdata/data/',
    'NLDAS': 'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0',
}
NETCDF_EXTENSIONS = {
    'GLDAS': 'nc4',
    'NLDAS': 'nc',
}
NETCDF_PREFIXES = {
    'GLDAS': 'GLDAS_NOAH025_3H.A',
    'NLDAS': 'NLDAS_FORA0125_H.A',
}
NETCDF_SUFFIXES = {
    'GLDAS': '021.nc4',
    'NLDAS': '020.nc',
}
NETCDF_SHAPES = {
    'GLDAS': (600, 1440),
    'gridMET': (585, 1386),
    'NLDAS': (224, 464),
}
DATA_INTERVALS = {
    # Data interval in hours
    'GLDAS': 3,
    'NLDAS': 1,
}
LAND_MASKS = {
    'GLDAS': {'file': os.path.join(pt, '../data/GLDASp5_landmask_025d.nc4'), 'variable': 'GLDAS_mask'},
    'gridMET': {'file': os.path.join(pt, '../data/gridMET_elevation_mask.nc'), 'variable': 'elevation'},    # For gridMET, mask and elevation are the same file
    'NLDAS': {'file': os.path.join(pt, '../data/NLDAS_masks-veg-soil.nc4'), 'variable': 'CONUS_mask'},
}
ELEVATIONS = {
    'GLDAS': {'file': os.path.join(pt, '../data/GLDASp5_elevation_025d.nc4'), 'variable': 'GLDAS_elevation'},
    'gridMET': {'file': os.path.join(pt, '../data/gridMET_elevation_mask.nc'), 'variable': 'elevation'},
    'NLDAS': {'file': os.path.join(pt, '../data/NLDAS_elevation.nc4'), 'variable': 'NLDAS_elev'},
}
NETCDF_VARIABLES = {
    'precipitation': {
        'GLDAS': 'Rainf_f_tavg',
        'NLDAS': 'Rainf',
    },
    'air_temperature': {
        'GLDAS': 'Tair_f_inst',
        'NLDAS': 'Tair',
    },
    'specific_humidity': {
        'GLDAS': 'Qair_f_inst',
        'NLDAS': 'Qair',
    },
    'wind_u': {
        'GLDAS': 'Wind_f_inst',
        'NLDAS': 'Wind_E',
    },
    'wind_v': {
        'GLDAS': 'Wind_f_inst',
        'NLDAS': 'Wind_N',
    },
    'solar': {
        'GLDAS': 'SWdown_f_tavg',
        'NLDAS': 'SWdown',
    },
    #'longwave': {
    #    'GLDAS': 'LWdown_f_tavg',
    #    'NLDAS': 'LWdown',
    #},
    'air_pressure': {
        'GLDAS': 'Psurf_f_inst',
        'NLDAS': 'PSurf',
    },
}
START_DATES = {
    'GLDAS': datetime.strptime('2000-01-01', '%Y-%m-%d'),
    'gridMET': datetime.strptime('1979-01-01', '%Y-%m-%d'),
    'NLDAS': datetime.strptime('1979-01-01', '%Y-%m-%d'),
}
START_HOURS = {
    'GLDAS': 3,
    'NLDAS': 13,
}
LA1 = {
    'GLDAS': -59.875,
    'gridMET': 49.4,
    'NLDAS': 25.0625,
}
LO1 = {
    'GLDAS': -179.875,
    'gridMET': -124.76667,
    'NLDAS': -124.9375,
}
DI = {
    'GLDAS': 0.25,
    'gridMET': 1.0 / 24.0,
    'NLDAS': 0.125,
}
DJ = {
    'GLDAS': 0.25,
    'gridMET': -1.0 / 24.0,
    'NLDAS': 0.125,
}
IND_J = lambda reanalysis, lat: int(round((lat - LA1[reanalysis]) / DJ[reanalysis]))
IND_I = lambda reanalysis, lon: int(round((lon - LO1[reanalysis]) / DI[reanalysis]))
WEATHER_FILE_VARIABLES = {
    # variable is the name of the variable in the NETCDF_VARIABLES dictionary
    # func is the function that converts the raw data to corresponding weather file variables
    # format is the output format in weather files
    'PP': {
        'XLDAS':{
            'variable': 'precipitation',
            'func': lambda x: x.resample('D').mean() * 86400,
        },
        'gridMET':{
            'variable': ('pr', 'precipitation_amount'),
            'func': lambda x: x,
        },
        'format': lambda x: "%-#.5g" % x if x >= 1.0 else "%-.4f" % x,
    },
    'TX': {
        'XLDAS': {
            'variable': 'air_temperature',
            'func': lambda x: x.resample('D').max() - 273.15,
        },
        'gridMET':{
            'variable': ('tmmx', 'air_temperature'),
            'func': lambda x: x - 273.15,
        },
        'format': lambda x: '%-7.2f' % x,
    },
    'TN': {
        'XLDAS': {
            'variable': 'air_temperature',
            'func': lambda x: x.resample('D').min() - 273.15,
        },
        'gridMET':{
            'variable': ('tmmn', 'air_temperature'),
            'func': lambda x: x - 273.15,
        },
        'format': lambda x: '%-7.2f' % x,
    },
    'SOLAR': {
        'XLDAS': {
            'variable': 'solar',
            'func': lambda x: x.resample('D').mean() * 86400.0 * 1.0E-6,
        },
        'gridMET':{
            'variable': ('srad', 'surface_downwelling_shortwave_flux_in_air'),
            'func': lambda x: x * 86400.0 * 1.0E-6,
        },
        'format': lambda x: '%-7.3f' % x,
    },
    'RHX': {
        'XLDAS': {
            'variable': 'relative_humidity',
            'func': lambda x: x.resample('D').max() * 100.0,
        },
        'gridMET':{
            'variable': ('rmax', 'relative_humidity'),
            'func': lambda x: x,
        },
        'format': lambda x: '%-7.2f' % x,
    },
    'RHN': {
        'XLDAS': {
            'variable': 'relative_humidity',
            'func': lambda x: x.resample('D').min() * 100.0,
        },
        'gridMET':{
            'variable': ('rmin', 'relative_humidity'),
            'func': lambda x: x,
        },
        'format': lambda x: '%-7.2f' % x,
    },
    'WIND': {
        'XLDAS': {
            'variable': 'wind',
            'func': lambda x: x.resample('D').mean(),
        },
        'gridMET':{
            'variable': ('vs', 'wind_speed'),
            'func': lambda x: x,
        },
        'format': lambda x: '%-.2f' % x,
    },
}

COOKIE_FILE = './.urs_cookies'


def _download_daily_xldas(path, xldas, day):
    cmd = [
        'wget',
        '--load-cookies',
        COOKIE_FILE,
        '--save-cookies',
        COOKIE_FILE,
        '--keep-session-cookies',
        '--no-check-certificate',
        '-r',
        '-c',
        '-N',
        '-nH',
        '-nd',
        '-np',
        '-A',
        NETCDF_EXTENSIONS[xldas],
        f'{URLS[xldas]}/{day.strftime("%Y/%j")}/',
        '-P',
        f'{path}/{day.strftime("%Y/%j")}',
    ]
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def download_xldas(data_path, xldas, date_start, date_end):
    os.makedirs(f'{data_path}/', exist_ok=True)

    d = date_start
    while d <= date_end:
        _download_daily_xldas(data_path, xldas, d)
        d += timedelta(days=1)


def download_gridmet(data_path, year):
    """Download gridMET forcing files
    """
    os.makedirs(f'{data_path}/', exist_ok=True)

    for var in WEATHER_FILE_VARIABLES:
        cmd = [
            'wget',
            '-c',
            '-N',
            '-nd',
            f'{URLS["gridMET"]}/{WEATHER_FILE_VARIABLES[var]["gridMET"]["variable"][0]}_{year}.nc',
            '-P',
            f'{data_path}/',
        ]
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def read_land_mask(reanalysis):
    with Dataset(LAND_MASKS[reanalysis]['file']) as nc:
        if reanalysis == 'gridMET':
            mask = nc[LAND_MASKS[reanalysis]['variable']][:, :]
        else:
            mask = nc[LAND_MASKS[reanalysis]['variable']][0]
        lats, lons = np.meshgrid(nc['lat'][:], nc['lon'][:], indexing='ij')

    with Dataset(ELEVATIONS[reanalysis]['file']) as nc:
        if reanalysis == 'gridMET':
            elevations = nc[ELEVATIONS[reanalysis]['variable']][:, :]
        else:
            elevations = nc[ELEVATIONS[reanalysis]['variable']][0][:, :]

    grid_df = pd.DataFrame({
        'latitude': lats.flatten(),
        'longitude': lons.flatten(),
        'mask': mask.flatten(),
        'elevation': elevations.flatten(),
    })

    if reanalysis == 'gridMET':
        grid_df.loc[~grid_df['mask'].isna(), 'mask'] = 1
        grid_df.loc[grid_df['mask'].isna(), 'mask'] = 0

    grid_df['mask'] = grid_df['mask'].astype(int)

    return grid_df


def _find_grid(reanalysis, grid_ind, mask_df, model, rcp):
    grid_lat, grid_lon = mask_df.loc[grid_ind, ['latitude', 'longitude']]

    grid_str = '%.3f%sx%.3f%s' % (
            abs(grid_lat), 'S' if grid_lat < 0.0 else 'N', abs(grid_lon), 'W' if grid_lon < 0.0 else 'E'
    )

    if reanalysis == 'MACA':
        fn = f'macav2metdata_{model}_rcp{rcp}_{grid_str}.weather'
    else:
        fn = f'{reanalysis}_{grid_str}.weather'

    return grid_lat, fn, mask_df.loc[grid_ind, 'elevation']


def find_grids(reanalysis, locations=None, model=None, rcp=None):
    mask_df = read_land_mask(reanalysis)

    if locations is None:
        indices = [ind for ind, row in mask_df.iterrows() if row['mask'] > 0]
        df = pd.DataFrame({'grid_index': indices})
    else:
        indices = []

        for (lat, lon) in locations:
            ind = np.ravel_multi_index((IND_J(reanalysis, lat), IND_I(reanalysis, lon)), NETCDF_SHAPES[reanalysis])

            if mask_df.loc[ind]['mask'] == 0:
                mask_df['distance'] = mask_df.apply(
                    lambda x: math.sqrt((x['latitude'] - lat) ** 2 + (x['longitude'] - lon) ** 2),
                    axis=1,
                )
                mask_df.loc[mask_df['mask'] == 0, 'distance'] = 1E6
                ind = mask_df['distance'].idxmin()

            indices.append(ind)
        df = pd.DataFrame({'grid_index': indices, 'input_coordinate': locations})

    df[['grid_latitude', 'weather_file', 'elevation']] = df.apply(
        lambda x: _find_grid(reanalysis, x['grid_index'], mask_df, model, rcp),
        axis=1,
        result_type='expand',
    )

    if locations is not None:
        if any(df.duplicated(subset=['grid_index'])):
            print(f"The following input coordinates share {reanalysis} grids:")
            indices = df['grid_index']
            print(df[indices.isin(indices[indices.duplicated()])].sort_values('grid_index')[['input_coordinate', 'weather_file']].to_string(index=False))
            print()

        print(f"{reanalysis} weather files:")
        df = df.drop_duplicates(subset=['grid_index'], keep='first')
        print(df[['input_coordinate', 'weather_file']].to_string(index=False))
        print()
    
    df.set_index('grid_index', inplace=True)

    return df


def _write_header(weather_path, fn, latitude, elevation, screening_height=10.0):
    with open(f'{weather_path}/{fn}', 'w') as f:
        # Open meteorological file and write header lines
        f.write('%-23s\t%.2f\n' % ('LATITUDE', latitude))
        f.write('%-23s\t%.2f\n' % ('ALTITUDE', elevation))
        f.write('%-23s\t%.1f\n' % ('SCREENING_HEIGHT', screening_height))
        f.write('%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%s\n' %
            ('YEAR', 'DOY', 'PP', 'TX', 'TN', 'SOLAR', 'RHX', 'RHN', 'WIND'))
        f.write('%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%s\n' %
            ('####', '###', 'mm', 'degC', 'degC', 'MJ/m2', '%', '%', 'm/s'))


def write_headers(weather_path, grid_df):
    grid_df.apply(lambda x: _write_header(weather_path, x['weather_file'], x['grid_latitude'], x['elevation']), axis=1)


def relative_humidity(air_temperature, air_pressure, specific_humidity):
    es = 611.2 * np.exp(17.67 * (air_temperature - 273.15) / (air_temperature - 273.15 + 243.5))
    ws = 0.622 * es / (air_pressure - es)
    w = specific_humidity / (1.0 - specific_humidity)
    rh = w / ws
    rh = np.minimum(rh, np.full(rh.shape, 1.0))
    rh = np.maximum(rh, np.full(rh.shape, 0.01))

    return rh


def _read_var(t, xldas, nc, indices, df):
    """Read meteorological variables of an array of desired grids from netCDF

    The netCDF variable arrays are flattened to make reading faster
    """
    values = {}
    for key in NETCDF_VARIABLES:
        values[key] = nc[NETCDF_VARIABLES[key][xldas]][0].flatten()[indices]

    if xldas == 'NLDAS':     # NLDAS precipitation unit is kg m-2. Convert to kg m-2 s-1 to be consistent with GLDAS
        values['precipitation'] /= DATA_INTERVALS[xldas] * 3600.0

    values['wind'] = np.sqrt(values['wind_u'] **2 + values['wind_v'] **2)

    ## Calculate relative humidity from specific humidity
    values['relative_humidity'] = relative_humidity(
        values['air_temperature'], values['air_pressure'], values['specific_humidity']
    )

    for var in ['precipitation', 'air_temperature', 'solar', 'relative_humidity', 'wind']:
        df.loc[t, df.columns.get_level_values(1) == var] = values[var]


def _write_weather_files(weather_path, daily_df, grid_df):
    daily_df['YEAR'] = daily_df.index.year.map(lambda x: "%-7d" % x)
    daily_df['DOY'] = daily_df.index.map(lambda x: "%-7d" % x.timetuple().tm_yday)

    for grid in grid_df.index:
        output_df = daily_df.loc[:, pd.IndexSlice[grid, :]].copy()
        output_df.columns = output_df.columns.droplevel()
        output_df = daily_df[['YEAR', 'DOY']].droplevel('variables', axis=1).join(output_df)

        for v in WEATHER_FILE_VARIABLES:
            output_df[v] = output_df[v].map(WEATHER_FILE_VARIABLES[v]['format'])

        with open(f'{weather_path}/{grid_df.loc[grid, "weather_file"]}', 'a') as f:
            output_df.to_csv(
                f,
                sep='\t',
                header=False,
                index=False,
        )


def _initialize_weather_files(weather_path, reanalysis, locations, header):
    os.makedirs(f'{weather_path}/', exist_ok=True)

    grid_df = find_grids(reanalysis, locations)

    if header == True: write_headers(weather_path,  grid_df)

    return grid_df


def process_xldas(data_path, weather_path, xldas, date_start, date_end, locations=None, header=True):
    """Process daily XLDAS data and write them to meteorological files
    """
    grid_df = _initialize_weather_files(weather_path, xldas, locations, header)

    ## Arrays to store daily values
    variables = ['precipitation', 'air_temperature', 'solar', 'relative_humidity', 'wind']
    columns = pd.MultiIndex.from_product([grid_df.index, variables], names=('grids', 'variables'))
    df = pd.DataFrame(columns=columns)

    t = date_start
    while t < date_end + timedelta(days=1):
        if t < START_DATES[xldas] + timedelta(hours=START_HOURS[xldas]): continue

        # netCDF file name
        fn = f'{t.strftime("%Y/%j")}/{NETCDF_PREFIXES[xldas]}{t.strftime("%Y%m%d.%H%M")}.{NETCDF_SUFFIXES[xldas]}'

        # Read one netCDF file
        with Dataset(f'{data_path}/{fn}') as nc:
            _read_var(t, xldas, nc, np.array(grid_df.index), df)

        t += timedelta(hours=DATA_INTERVALS[xldas])

    daily_df = pd.DataFrame()

    for key in WEATHER_FILE_VARIABLES:
        variable = WEATHER_FILE_VARIABLES[key]['XLDAS']['variable']
        func = WEATHER_FILE_VARIABLES[key]['XLDAS']['func']
        daily_df = pd.concat(
            [
                daily_df,
                func(df.loc[:, df.columns.get_level_values(1) == variable]).rename(columns={variable: key}, level=1)
            ],
            axis=1,
        )

    _write_weather_files(weather_path, daily_df, grid_df)


def process_gridmet(data_path, weather_path, date_start, date_end, locations=None, header=True):
    """Process annual gridMET data and write them to weather files
    """
    grid_df = _initialize_weather_files(weather_path, 'gridMET', locations, header)

    year = -9999
    variables = list(WEATHER_FILE_VARIABLES.keys())
    columns = pd.MultiIndex.from_product([grid_df.index, variables], names=('grids', 'variables'))
    df = pd.DataFrame(columns=columns)

    t = date_start
    while t < date_end + timedelta(days=1):
        if t.year != year:
            # Close netCDF files that are open
            if year != -9999:
                for key in WEATHER_FILE_VARIABLES: ncs[key].close()

            year = t.year
            ncs = {
                key: Dataset(f'{data_path}/{value["gridMET"]["variable"][0]}_{year}.nc')
                for key, value in WEATHER_FILE_VARIABLES.items()
            }

        for key in WEATHER_FILE_VARIABLES:
            variable = WEATHER_FILE_VARIABLES[key]['gridMET']['variable'][1]
            func = WEATHER_FILE_VARIABLES[key]['gridMET']['func']
            df.loc[t, df.columns.get_level_values(1) == key] = func(
                ncs[key][variable][t.timetuple().tm_yday - 1].flatten()[np.array(grid_df.index)]
            )

        t += timedelta(days=1)

    for key in WEATHER_FILE_VARIABLES: ncs[key].close()

    _write_weather_files(weather_path, df, grid_df)
