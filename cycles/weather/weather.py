import geopandas as gpd
import math
import numpy as np
import os
import pandas as pd
import subprocess
from datetime import datetime, timedelta
from netCDF4 import Dataset
from tqdm import tqdm

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
        'GLDAS': '',
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

SUBDAILY_WEATHER_FILE_VARIABLES = {
    # variable is the name of the variable in the NETCDF_VARIABLES dictionary
    # func is the function that converts the raw data to corresponding weather file variables
    # format is the output format in weather files
    'PP': {
        'variable': 'precipitation',
        'func': lambda x, h: x * h * 3600.0,
        'format': lambda x: "%-#.5g" % x if x >= 1.0 else "%-.4f" % x,
    },
    'TMP': {
        'variable': 'air_temperature',
        'func': lambda x, h: x - 273.15,
        'format': lambda x: '%-7.2f' % x,
    },
    'SOLAR': {
        'variable': 'solar',
        'func': lambda x, h: x * h * 3600.0 * 1.0E-6,
        'format': lambda x: '%-7.3f' % x,
    },
    'RH': {
        'variable': 'relative_humidity',
        'func': lambda x, h: x * 100.0,
        'format': lambda x: '%-7.2f' % x,
    },
    'WIND': {
        'variable': 'wind',
        'func': lambda x, h: x,
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


def download_xldas(data_path: str, xldas: str, date_start: datetime, date_end: datetime) -> None:
    os.makedirs(f'{data_path}/', exist_ok=True)

    d = date_start
    with tqdm(total=(date_end - date_start).days + 1, desc=f'Download {xldas} files', unit=' days') as progress_bar:
        while d <= date_end:
            _download_daily_xldas(data_path, xldas, d)
            d += timedelta(days=1)
            progress_bar.update(1)


def download_gridmet(data_path: str, year_start: int, year_end: int) -> None:
    """Download gridMET forcing files
    """
    os.makedirs(f'{data_path}/', exist_ok=True)

    for year in range(year_start, year_end + 1):
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


def _read_land_mask(reanalysis):
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

    grid_str = '%.3f%sx%.3f%s' % (abs(grid_lat), 'S' if grid_lat < 0.0 else 'N', abs(grid_lon), 'W' if grid_lon < 0.0 else 'E')

    if reanalysis == 'MACA':
        fn = f'macav2metdata_{model}_rcp{rcp}_{grid_str}'
    else:
        fn = f'{reanalysis}_{grid_str}'

    return grid_lat, fn, mask_df.loc[grid_ind, 'elevation']


def find_grids(reanalysis: str, *, locations: dict[str, tuple[float, float]] | list[tuple[float, float]]=None, model: str=None, rcp:str=None, screen_output=True) -> pd.DataFrame:
    mask_df = _read_land_mask(reanalysis)

    if locations is None:
        indices = [ind for ind, row in mask_df.iterrows() if row['mask'] > 0]
        df = pd.DataFrame({'grid_index': indices})
    else:
        indices = []
        sites = []

        for loc in locations:
            if isinstance(locations, list):
                (lat, lon) = loc
            elif isinstance(locations, dict):
                (lat, lon) = locations[loc]
            else:
                raise TypeError('Location input must be a dict or list of coordinates.')

            sites.append(loc)

            ind = np.ravel_multi_index((IND_J(reanalysis, lat), IND_I(reanalysis, lon)), NETCDF_SHAPES[reanalysis])

            if mask_df.loc[ind]['mask'] == 0:
                mask_df['distance'] = mask_df.apply(
                    lambda x: math.sqrt((x['latitude'] - lat) ** 2 + (x['longitude'] - lon) ** 2),
                    axis=1,
                )
                mask_df.loc[mask_df['mask'] == 0, 'distance'] = 1E6
                ind = mask_df['distance'].idxmin()

            indices.append(ind)

        df = pd.DataFrame({
            'grid_index': indices,
            'input_coordinate': locations if isinstance(locations, list) else locations.values(),
        })

        if sites: df['site'] = sites

    df[['grid_latitude', 'weather_file', 'elevation']] = df.apply(
        lambda x: _find_grid(reanalysis, x['grid_index'], mask_df, model, rcp),
        axis=1,
        result_type='expand',
    )

    if locations is not None:
        if any(df.duplicated(subset=['grid_index'])):
            indices = df['grid_index']
            if screen_output is True:
                print(f"The following input coordinates share {reanalysis} grids:")
                print(df[indices.isin(indices[indices.duplicated()])].sort_values('grid_index')[['input_coordinate', 'weather_file']].to_string(index=False))
                print()

        if screen_output is True:
            print(f"{reanalysis} weather files:")
            if not sites:
                print(df[['input_coordinate', 'weather_file']].to_string(index=False))
            else:
                print(df[['site', 'input_coordinate', 'weather_file']].to_string(index=False))
            print()

    df = df.drop_duplicates(subset=['grid_index'], keep='first')
    df.set_index('grid_index', inplace=True)

    return df


def _write_header(weather_path, fn, latitude, elevation, *, screening_height=10.0, hourly=False):
    with open(f'{weather_path}/{fn}.{"hourly.weather" if hourly else "weather"}', 'w') as f:
        # Open meteorological file and write header lines
        f.write('%-23s\t%.2f\n' % ('LATITUDE', latitude))
        f.write('%-23s\t%.2f\n' % ('ALTITUDE', elevation))
        f.write('%-23s\t%.1f\n' % ('SCREENING_HEIGHT', screening_height))
        if hourly:
            f.write('%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%s\n' % ('YEAR', 'DOY', 'HOUR', 'PP', 'TMP', 'SOLAR', 'RH', 'WIND'))
            f.write('%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%s\n' % ('####', '###', '###', 'mm', 'degC', 'MJ/m2', '%', 'm/s'))
        else:
            f.write('%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%s\n' % ('YEAR', 'DOY', 'PP', 'TX', 'TN', 'SOLAR', 'RHX', 'RHN', 'WIND'))
            f.write('%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%s\n' % ('####', '###', 'mm', 'degC', 'degC', 'MJ/m2', '%', '%', 'm/s'))


def _write_weather_headers(weather_path, grid_df, hourly=False):
    grid_df.apply(lambda x: _write_header(weather_path, x['weather_file'], x['grid_latitude'], x['elevation'], hourly=hourly), axis=1)


def _relative_humidity(air_temperature: np.array, air_pressure: np.array, specific_humidity: np.array) -> np.array:
    es = 611.2 * np.exp(17.67 * (air_temperature - 273.15) / (air_temperature - 273.15 + 243.5))
    ws = 0.622 * es / (air_pressure - es)
    w = specific_humidity / (1.0 - specific_humidity)
    rh = w / ws
    rh = np.minimum(rh, np.full(rh.shape, 1.0))
    rh = np.maximum(rh, np.full(rh.shape, 0.01))

    return rh


def _read_xldas_netcdf(t, xldas, nc, indices, df):
    """Read meteorological variables of an array of desired grids from netCDF

    The netCDF variable arrays are flattened to make reading faster
    """
    values = {key: 0.0 if not NETCDF_VARIABLES[key][xldas] else nc[NETCDF_VARIABLES[key][xldas]][0].flatten()[indices]
        for key in NETCDF_VARIABLES}

    # NLDAS precipitation unit is kg m-2. Convert to kg m-2 s-1 to be consistent with GLDAS
    if xldas == 'NLDAS': values['precipitation'] /= DATA_INTERVALS[xldas] * 3600.0

    values['wind'] = np.sqrt(values['wind_u'] **2 + values['wind_v'] **2)

    ## Calculate relative humidity from specific humidity
    values['relative_humidity'] = _relative_humidity(values['air_temperature'], values['air_pressure'], values['specific_humidity'])

    for var in ['precipitation', 'air_temperature', 'solar', 'relative_humidity', 'wind']:
        df.loc[t, df.columns.get_level_values(1) == var] = values[var]


def _write_weather_files(weather_path, weather_df, grid_df, *, hourly=False):
    weather_df['YEAR'] = weather_df.index.year.map(lambda x: "%-7d" % x)
    weather_df['DOY'] = weather_df.index.map(lambda x: "%-7.3d" % x.timetuple().tm_yday)
    if hourly:
        weather_df['HOUR'] = weather_df.index.hour.map(lambda x: "%-7.2d" % x)

    for grid in grid_df.index:
        output_df = weather_df.loc[:, pd.IndexSlice[grid, :]].copy()
        output_df.columns = output_df.columns.droplevel()
        if hourly:
            output_df = weather_df[['YEAR', 'DOY', 'HOUR']].droplevel('variables', axis=1).join(output_df)
        else:
            output_df = weather_df[['YEAR', 'DOY']].droplevel('variables', axis=1).join(output_df)

        if hourly:
            for v in SUBDAILY_WEATHER_FILE_VARIABLES:
                output_df[v] = output_df[v].map(SUBDAILY_WEATHER_FILE_VARIABLES[v]['format'])
        else:
            for v in WEATHER_FILE_VARIABLES:
                output_df[v] = output_df[v].map(WEATHER_FILE_VARIABLES[v]['format'])

        with open(f'{weather_path}/{grid_df.loc[grid, "weather_file"]}.{"hourly.weather" if hourly else "weather"}', 'a') as f:
            output_df.to_csv(
                f,
                sep='\t',
                header=False,
                index=False,
        )


def _initialize_weather_files(weather_path, reanalysis, locations, *, header=False, hourly=False):
    os.makedirs(f'{weather_path}/', exist_ok=True)

    grid_df = find_grids(reanalysis, locations=locations)

    if header == True: _write_weather_headers(weather_path,  grid_df, hourly=hourly)

    return grid_df


def process_xldas(data_path: str, weather_path: str, xldas: str, date_start: datetime, date_end: datetime, *, hourly: bool=False, locations: dict[str, tuple[float, float]] | list[tuple[float, float]]=None, header: bool=True) -> None:
    grid_df = _initialize_weather_files(weather_path, xldas, locations, header=header, hourly=hourly)

    # Arrays to store daily values
    variables = ['precipitation', 'air_temperature', 'solar', 'relative_humidity', 'wind']
    columns = pd.MultiIndex.from_product([grid_df.index, variables], names=('grids', 'variables'))
    df = pd.DataFrame(columns=columns)

    t = date_start
    with tqdm(total=(date_end - date_start).days + 1, desc=f'Process {xldas} files', unit=' days') as progress_bar:
        while t < date_end + timedelta(days=1):
            if t >= START_DATES[xldas] + timedelta(hours=START_HOURS[xldas]):
                # netCDF file name
                fn = f'{t.strftime("%Y/%j")}/{NETCDF_PREFIXES[xldas]}{t.strftime("%Y%m%d.%H%M")}.{NETCDF_SUFFIXES[xldas]}'

                # Read one netCDF file
                with Dataset(f'{data_path}/{fn}') as nc:
                    _read_xldas_netcdf(t, xldas, nc, np.array(grid_df.index), df)

            t += timedelta(hours=DATA_INTERVALS[xldas])
            if (t - date_start).total_seconds() % 86400 == 0: progress_bar.update(1)

    output_df = pd.DataFrame()

    if hourly:
        for key in SUBDAILY_WEATHER_FILE_VARIABLES:
            variable = SUBDAILY_WEATHER_FILE_VARIABLES[key]['variable']
            func = SUBDAILY_WEATHER_FILE_VARIABLES[key]['func']
            output_df = pd.concat(
                [output_df, func(df.loc[:, df.columns.get_level_values(1) == variable], DATA_INTERVALS[xldas]).rename(columns={variable: key}, level=1)],
                axis=1,
            )
        # If data interval is not hourly, interpolate to hourly
        if DATA_INTERVALS[xldas] != 1:
            output_df = output_df.astype(float).resample('H').mean().interpolate(method='linear')
            output_df.loc[:, (slice(None), 'PP')] /= DATA_INTERVALS[xldas]
    else:
        for key in WEATHER_FILE_VARIABLES:
            variable = WEATHER_FILE_VARIABLES[key]['XLDAS']['variable']
            func = WEATHER_FILE_VARIABLES[key]['XLDAS']['func']
            output_df = pd.concat(
                [output_df, func(df.loc[:, df.columns.get_level_values(1) == variable]).rename(columns={variable: key}, level=1)],
                axis=1,
            )

    _write_weather_files(weather_path, output_df, grid_df, hourly=hourly)


def process_gridmet(data_path: str, weather_path: str, date_start: datetime, date_end: datetime, *, locations: dict[str, tuple[float, float]] | list[tuple[float, float]]=None, header: bool=True) -> None:
    """Process annual gridMET data and write them to weather files
    """
    grid_df = _initialize_weather_files(weather_path, 'gridMET', locations, header=header)

    year = -9999
    variables = list(WEATHER_FILE_VARIABLES.keys())
    columns = pd.MultiIndex.from_product([grid_df.index, variables], names=('grids', 'variables'))
    df = pd.DataFrame(columns=columns)

    t = date_start
    with tqdm(total=(date_end - date_start).days + 1, desc=f'Process gridMET files', unit=' days') as progress_bar:
        while t < date_end + timedelta(days=1):
            if t.year != year:
                # Close netCDF files that are open
                if year != -9999:
                    for key in WEATHER_FILE_VARIABLES: ncs[key].close()

                year = t.year
                ncs = {key: Dataset(f'{data_path}/{value["gridMET"]["variable"][0]}_{year}.nc') for key, value in WEATHER_FILE_VARIABLES.items()}

            for key in WEATHER_FILE_VARIABLES:
                variable = WEATHER_FILE_VARIABLES[key]['gridMET']['variable'][1]
                func = WEATHER_FILE_VARIABLES[key]['gridMET']['func']
                df.loc[t, df.columns.get_level_values(1) == key] = func(ncs[key][variable][t.timetuple().tm_yday - 1].flatten()[np.array(grid_df.index)])

            t += timedelta(days=1)
            progress_bar.update(1)

    for key in WEATHER_FILE_VARIABLES: ncs[key].close()

    _write_weather_files(weather_path, df, grid_df)
