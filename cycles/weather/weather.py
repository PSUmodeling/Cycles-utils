import geopandas as gpd
import math
import numpy as np
import os
import pandas as pd
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from netCDF4 import Dataset
from pathlib import Path
from tqdm import tqdm
from typing import Callable

pt = os.path.dirname(os.path.realpath(__file__))

@dataclass
class Reanalysis:
    name: str
    url: str
    netcdf_extension: str | None
    netcdf_prefix: str | None
    netcdf_suffix: str | None
    netcdf_shape: tuple[int, int]
    data_interval: int
    land_mask_file: Path | str
    land_mask: Callable
    elevation_file: Path | str
    elevation: Callable
    start_date: datetime
    start_hour: int
    la1: float
    lo1: float
    di: float
    dj: float
    ind_j: Callable
    ind_i: Callable
    netcdf_variables: dict[str, str]
    weather_file_variables: Callable

REANALYSES = {
    'GLDAS': Reanalysis(
        name='GLDAS',
        url='https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_3H.2.1',
        netcdf_extension='nc4',
        netcdf_prefix='GLDAS_NOAH025_3H.A',
        netcdf_suffix='021.nc4',
        netcdf_shape=(600, 1440),
        data_interval=3,
        land_mask_file=os.path.join(pt, '../data/GLDASp5_landmask_025d.nc4'),
        land_mask=lambda nc: nc['GLDAS_mask'][0],
        elevation_file=os.path.join(pt, '../data/GLDASp5_elevation_025d.nc4'),
        elevation=lambda nc: nc['GLDAS_elevation'][0],
        start_date=datetime.strptime('2000-01-01', '%Y-%m-%d'),
        start_hour=3,
        la1=-59.875,
        lo1=-179.875,
        di=0.25,
        dj=0.25,
        ind_j=lambda lat: int(round((lat - (-59.875)) / 0.25)),
        ind_i=lambda lon: int(round((lon - (-179.875)) / 0.25)),
        netcdf_variables={
            'precipitation': 'Rainf_f_tavg',
            'air temperature': 'Tair_f_inst',
            'specific humidity': 'Qair_f_inst',
            'wind': 'Wind_f_inst',
            'solar': 'SWdown_f_tavg',
            'air pressure': 'Psurf_f_inst',
        },
        weather_file_variables={
            'PP': lambda dfs, hourly: _interpolate_to_hourly(dfs['precipitation']) * 3600.0 if hourly else dfs['precipitation'].resample('D').mean() * 86400.0,
            'TMP': lambda dfs, _: _interpolate_to_hourly(dfs['air temperature']) - 273.15,
            'TX': lambda dfs, _: dfs['air temperature'].resample('D').max() - 273.15,
            'TN': lambda dfs, _: dfs['air temperature'].resample('D').min() - 273.15,
            'SOLAR': lambda dfs, hourly: _interpolate_to_hourly(dfs['solar']) * 3600.0 * 1.0E-6 if hourly else dfs['solar'].resample('D').mean() * 86400.0 * 1.0E-6,
            'RH': lambda dfs, _: _interpolate_to_hourly(_relative_humidity(dfs)) * 100.0,
            'RHX': lambda dfs, _: _relative_humidity(dfs).resample('D').max() * 100.0,
            'RHN': lambda dfs, _: _relative_humidity(dfs).resample('D').min() * 100.0,
            'WIND': lambda dfs, hourly: _interpolate_to_hourly(dfs['wind']) if hourly else dfs['wind'].resample('D').mean(),
        }
    ),
    'gridMET': Reanalysis(
        name='gridMET',
        url='http://www.northwestknowledge.net/metdata/data/',
        netcdf_extension=None,
        netcdf_prefix=None,
        netcdf_suffix=None,
        netcdf_shape=(585, 1386),
        data_interval=None,
        # For gridMET, land mask and elevation are the same file
        land_mask_file=os.path.join(pt, '../data/gridMET_elevation_mask.nc'),
        land_mask=lambda nc: nc['elevation'][:, :],
        elevation_file=os.path.join(pt, '../data/gridMET_elevation_mask.nc'),
        elevation=lambda nc: nc['elevation'][:, :],
        start_date=datetime.strptime('1979-01-01', '%Y-%m-%d'),
        start_hour=None,
        la1=49.4,
        lo1=-124.76667,
        di=1.0 / 24.0,
        dj=-1.0 / 24.0,
        ind_j=lambda lat: int(round((lat - 49.4) / (-1.0 / 24.0))),
        ind_i=lambda lon: int(round((lon - (-124.76667)) / (1.0 / 24.0))),
        netcdf_variables={
            'pr': 'precipitation_amount',
            'tmmx': 'air_temperature',
            'tmmn': 'air_temperature',
            'srad': 'surface_downwelling_shortwave_flux_in_air',
            'rmax': 'relative_humidity',
            'rmin': 'relative_humidity',
            'vs': 'wind_speed',
        },
        weather_file_variables={
            'PP': lambda dfs, _: dfs['pr'],
            'TX': lambda dfs, _: dfs['tmmx'] - 273.15,
            'TN': lambda dfs, _: dfs['tmmn'] - 273.15,
            'SOLAR': lambda dfs, _: dfs['srad'] * 86400.0 * 1.0E-6,
            'RHX': lambda dfs, _: dfs['rmax'],
            'RHN': lambda dfs, _: dfs['rmin'],
            'WIND': lambda dfs, _: dfs['vs'],
        }
    ),
    'NLDAS': Reanalysis(
        name='NLDAS',
        url='https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0',
        netcdf_extension='nc',
        netcdf_prefix='NLDAS_FORA0125_H.A',
        netcdf_suffix='020.nc',
        netcdf_shape=(224, 464),   # NLDAS grid is 0.125 degree, but only covers CONUS (25.0625N-49.9375N, 124.9375W-66.0625W)
        data_interval=1,
        land_mask_file=os.path.join(pt, '../data/NLDAS_masks-veg-soil.nc4'),
        land_mask=lambda nc: nc['CONUS_mask'][0],
        elevation_file=os.path.join(pt, '../data/NLDAS_elevation.nc4'),
        elevation=lambda nc: nc['NLDAS_elev'][0],
        start_date=datetime.strptime('1979-01-01', '%Y-%m-%d'),
        start_hour=13,
        la1=25.0625,
        lo1=-124.9375,
        di=0.125,
        dj=0.125,
        ind_j=lambda lat: int(round((lat - 25.0625) / 0.125)),
        ind_i=lambda lon: int(round((lon - (-124.9375)) / 0.125)),
        netcdf_variables={
            'precipitation': 'Rainf',
            'air temperature': 'Tair',
            'specific humidity': 'Qair',
            'wind_u': 'Wind_E',
            'wind_v': 'Wind_N',
            'solar': 'SWdown',
            'air pressure': 'PSurf',
        },
        weather_file_variables={
            'PP': lambda dfs, hourly: dfs['precipitation'] if hourly else dfs['precipitation'].resample('D').sum(),
            'TMP': lambda dfs, _: dfs['air temperature'] - 273.15,
            'TX': lambda dfs, _: dfs['air temperature'].resample('D').max() - 273.15,
            'TN': lambda dfs, _: dfs['air temperature'].resample('D').min() - 273.15,
            'SOLAR': lambda dfs, hourly: dfs['solar'] * 3600.0 * 1.0E-6 if hourly else dfs['solar'].resample('D').mean() * 86400.0 * 1.0E-6,
            'RH': lambda dfs, _: _relative_humidity(dfs) * 100.0,
            'RHX': lambda dfs, _: _relative_humidity(dfs).resample('D').max() * 100.0,
            'RHN': lambda dfs, _: _relative_humidity(dfs).resample('D').min() * 100.0,
            'WIND': lambda dfs, hourly: _wind_speed(dfs) if hourly else _wind_speed(dfs).resample('D').mean(),
        }
    ),
}

@dataclass
class WeatherFileVariable:
    fmt: Callable
    unit: str
    daily: bool
    hourly: bool

WEATHER_FILE_VARIABLES = {
    'YEAR': WeatherFileVariable(fmt=lambda x: '%-7.4d' % x, unit='####', daily=True, hourly=True),
    'DOY': WeatherFileVariable(fmt=lambda x: '%-7.3d' % x, unit='###', daily=True, hourly=True),
    'HOUR': WeatherFileVariable(fmt=lambda x: '%-7.2d' % x, unit='####', daily=False, hourly=True),
    'PP': WeatherFileVariable(fmt=lambda x: "%-#.5g" % x if x >= 1.0 else "%-.4f" % x, unit='mm', daily=True, hourly=True),
    'TMP': WeatherFileVariable(fmt=lambda x: '%-7.2f' % x, unit='degC', daily=False, hourly=True),
    'TX': WeatherFileVariable(fmt=lambda x: '%-7.2f' % x, unit='degC', daily=True, hourly=False),
    'TN': WeatherFileVariable(fmt=lambda x: '%-7.2f' % x, unit='degC', daily=True, hourly=False),
    'SOLAR': WeatherFileVariable(fmt=lambda x: '%-7.3f' % x, unit='MJ/m2', daily=True, hourly=True),
    'RH': WeatherFileVariable(fmt=lambda x: '%-7.2f' % x, unit='%', daily=False, hourly=True),
    'RHX': WeatherFileVariable(fmt=lambda x: '%-7.2f' % x, unit='%', daily=True, hourly=False),
    'RHN': WeatherFileVariable(fmt=lambda x: '%-7.2f' % x, unit='%', daily=True, hourly=False),
    'WIND': WeatherFileVariable(fmt=lambda x: '%-.2f' % x, unit='m/s', daily=True, hourly=True),
}
COOKIE_FILE = './.urs_cookies'


def _interpolate_to_hourly(s: pd.Series) -> pd.Series:
    return s.astype(float).resample('h').mean().interpolate(method='linear')


def _download_daily_xldas(path: Path, forcing: str, day: datetime):
    reanalysis = REANALYSES[forcing]
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
        reanalysis.netcdf_extension,
        f'{reanalysis.url}/{day.strftime("%Y/%j")}/',
        '-P',
        path/f'{day.strftime("%Y/%j")}',
    ]
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def download_xldas(data_path: str | Path, forcing: str, date_start: datetime, date_end: datetime) -> None:
    # Create data directory if it doesn't exist
    Path(data_path).mkdir(parents=True, exist_ok=True)

    d = date_start
    with tqdm(total=(date_end - date_start).days + 1, desc=f'Download {forcing} files', unit=' days') as progress_bar:
        while d <= date_end:
            _download_daily_xldas(Path(data_path), forcing, d)
            d += timedelta(days=1)
            progress_bar.update(1)


def download_gridmet(data_path: Path | str, year_start: int, year_end: int) -> None:
    """Download gridMET forcing files
    """
    # Create data directory if it doesn't exist
    Path(data_path).mkdir(parents=True, exist_ok=True)
    gridmet = REANALYSES['gridMET']

    with tqdm(total=(year_end - year_start + 1) * len(gridmet.netcdf_variables), desc='Download gridMET files', unit=' files') as progress_bar:
        for year in range(year_start, year_end + 1):
            for var in gridmet.netcdf_variables:
                cmd = [
                    'wget',
                    '-c',
                    '-N',
                    '-nd',
                    f'{gridmet.url}/{var}_{year}.nc',
                    '-P',
                    f'{data_path}/',
                ]
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                progress_bar.update(1)


def _read_land_mask(forcing: str) -> pd.DataFrame:
    reanalysis = REANALYSES[forcing]
    # Land mask
    with Dataset(reanalysis.land_mask_file) as nc:
        mask = reanalysis.land_mask(nc)
        lats, lons = np.meshgrid(nc['lat'][:], nc['lon'][:], indexing='ij')

    # Elevation
    with Dataset(reanalysis.elevation_file) as nc:
        elevations = reanalysis.elevation(nc)

    grid_df = pd.DataFrame({
        'latitude': lats.flatten(),
        'longitude': lons.flatten(),
        'mask': mask.flatten(),
        'elevation': elevations.flatten(),
    })

    if forcing == 'gridMET':
        grid_df.loc[~grid_df['mask'].isna(), 'mask'] = 1
        grid_df.loc[grid_df['mask'].isna(), 'mask'] = 0

    grid_df['mask'] = grid_df['mask'].astype(int)

    return grid_df


def _find_grid(forcing: str, grid_ind, mask_df, model, rcp) -> tuple[float, str, float]:
    grid_lat, grid_lon = mask_df.loc[grid_ind, ['latitude', 'longitude']]
    grid_str = '%.3f%sx%.3f%s' % (abs(grid_lat), 'S' if grid_lat < 0.0 else 'N', abs(grid_lon), 'W' if grid_lon < 0.0 else 'E')

    fn = f'macav2metdata_{model}_rcp{rcp}_{grid_str}' if forcing == 'MACA' else f'{forcing}_{grid_str}'

    return grid_lat, fn, mask_df.loc[grid_ind, 'elevation']


def find_grids(forcing: str, *, locations: dict[str, tuple[float, float]] | list[tuple[float, float]] | None=None,
    model: str | None=None, rcp:str | None=None, screen_output=True) -> pd.DataFrame:
    reanalysis = REANALYSES[forcing]

    mask_df = _read_land_mask(forcing)

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
                (lat, lon) = locations[loc] # type: ignore
            else:
                raise TypeError('Location input must be a dict or list of coordinates.')

            sites.append(loc)

            ind = np.ravel_multi_index((reanalysis.ind_j(lat), reanalysis.ind_i(lon)), reanalysis.netcdf_shape)

            if mask_df.loc[ind]['mask'] == 0:   # type: ignore
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
        lambda x: _find_grid(forcing, x['grid_index'], mask_df, model, rcp),
        axis=1,
        result_type='expand',
    )

    if locations is not None:
        if any(df.duplicated(subset=['grid_index'])):
            indices = df['grid_index']
            if screen_output is True:
                print(f"The following input coordinates share {forcing} grids:")
                print(df[indices.isin(indices[indices.duplicated()])].sort_values('grid_index')[['input_coordinate', 'weather_file']].to_string(index=False))
                print()

        if screen_output is True:
            print(f"{forcing} weather files:")
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
            f.write('\t'.join(['%-7s' for var in WEATHER_FILE_VARIABLES.values() if var.hourly]) % tuple([key for key, var in WEATHER_FILE_VARIABLES.items() if var.hourly]))
            f.write('\n')
            f.write('\t'.join(['%-7s' for var in WEATHER_FILE_VARIABLES.values() if var.hourly]) % tuple([var.unit for var in WEATHER_FILE_VARIABLES.values() if var.hourly]))
            f.write('\n')
        else:
            f.write('\t'.join(['%-7s' for var in WEATHER_FILE_VARIABLES.values() if var.daily]) % tuple([key for key, var in WEATHER_FILE_VARIABLES.items() if var.daily]))
            f.write('\n')
            f.write('\t'.join(['%-7s' for var in WEATHER_FILE_VARIABLES.values() if var.daily]) % tuple([var.unit for var in WEATHER_FILE_VARIABLES.values() if var.daily]))
            f.write('\n')


def _write_weather_headers(weather_path, grid_df, hourly=False):
    grid_df.apply(lambda x: _write_header(weather_path, x['weather_file'], x['grid_latitude'], x['elevation'], hourly=hourly), axis=1)


def _relative_humidity(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    air_temperature = dfs['air temperature']
    air_pressure = dfs['air pressure']
    specific_humidity = dfs['specific humidity']

    es = 611.2 * np.exp(17.67 * (air_temperature - 273.15) / (air_temperature - 273.15 + 243.5))
    ws = 0.622 * es / (air_pressure - es)
    w = specific_humidity / (1.0 - specific_humidity)
    rh = w / ws
    rh = np.minimum(rh, pd.DataFrame(np.full(rh.shape, 1.0), index=rh.index, columns=rh.columns))
    rh = np.maximum(rh, pd.DataFrame(np.full(rh.shape, 0.01), index=rh.index, columns=rh.columns))

    return rh


def _wind_speed(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    u = dfs['wind_u']
    v = dfs['wind_v']

    wind = pd.DataFrame(np.sqrt(u ** 2 + v ** 2))

    return wind


def _read_xldas_netcdf(t, reanalysis: Reanalysis, nc, indices, dfs: dict[str, pd.DataFrame]):
    """Read meteorological variables of an array of desired grids from netCDF

    The netCDF variable arrays are flattened to make reading faster
    """
    values = {key: nc[reanalysis.netcdf_variables[key]][0].flatten()[indices] for key in reanalysis.netcdf_variables}

    for var in reanalysis.netcdf_variables:
        if dfs[var].empty:
            dfs[var] = pd.DataFrame([values[var]], columns=[f'grid_{k}' for k in indices], index=[t])
        else:
            dfs[var] = pd.concat([dfs[var], pd.DataFrame([values[var]], columns=[f'grid_{k}' for k in indices], index=[t])])


def _write_weather_files(weather_path: Path | str, weather_data: dict[str, pd.DataFrame], grid_df, *, hourly=False):
    for grid in grid_df.index:
        # Choose variables for output
        if hourly:
            output_df = pd.DataFrame({key: weather_data[key][f'grid_{grid}'] for key, var in WEATHER_FILE_VARIABLES.items() if var.hourly and not var.unit.startswith('#')})
        else:
            output_df = pd.DataFrame({key: weather_data[key][f'grid_{grid}'] for key, var in WEATHER_FILE_VARIABLES.items() if var.daily and not var.unit.startswith('#')})

        # Add time columns
        output_df.insert(0, 'YEAR', output_df.index.year)
        output_df.insert(1, 'DOY', output_df.index.map(lambda x: x.timetuple().tm_yday))
        if hourly: output_df.insert(2, 'HOUR', output_df.index.hour)

        for c in output_df.columns:
            output_df[c] = output_df[c].map(WEATHER_FILE_VARIABLES[c].fmt)

        with open(f'{weather_path}/{grid_df.loc[grid, "weather_file"]}.{"hourly.weather" if hourly else "weather"}', 'a') as f:
            output_df.to_csv(
                f,
                sep='\t',
                header=False,
                index=False,
        )


def _initialize_weather_files(weather_path, forcing: str, locations, *, header=False, hourly=False):
    os.makedirs(f'{weather_path}/', exist_ok=True)

    grid_df = find_grids(forcing, locations=locations)

    if header == True: _write_weather_headers(weather_path,  grid_df, hourly=hourly)

    return grid_df


def _process_xldas(data_path: str, reanalysis: Reanalysis, date_start: datetime, date_end: datetime, grid_df: pd.DataFrame, hourly: bool) -> dict[str, pd.DataFrame]:
    # Arrays to store daily values
    dfs = {var: pd.DataFrame() for var in reanalysis.netcdf_variables}

    t = date_start
    with tqdm(total=(date_end - date_start).days + 1, desc=f'Process {reanalysis.name} files', unit=' days') as progress_bar:
        while t < date_end + timedelta(days=1):
            if t >= reanalysis.start_date + timedelta(hours=reanalysis.start_hour):
                # netCDF file name
                fn = f'{t.strftime("%Y/%j")}/{reanalysis.netcdf_prefix}{t.strftime("%Y%m%d.%H%M")}.{reanalysis.netcdf_suffix}'

                # Read one netCDF file
                with Dataset(f'{data_path}/{fn}') as nc:
                    _read_xldas_netcdf(t, reanalysis, nc, np.array(grid_df.index), dfs)

            t += timedelta(hours=reanalysis.data_interval)
            if (t - date_start).total_seconds() % 86400 == 0: progress_bar.update(1)
    
    return dfs


def _process_gridmet(data_path: str, date_start: datetime, date_end: datetime, grid_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Process annual gridMET data and write them to weather files
    """
    gridmet = REANALYSES['gridMET']

    dfs = {var: pd.DataFrame() for var in gridmet.netcdf_variables}

    year = -9999
    t = date_start
    with tqdm(total=(date_end - date_start).days + 1, desc=f'Process gridMET files', unit=' days') as progress_bar:
        while t < date_end + timedelta(days=1):
            if t.year != year:
                # Close netCDF files that are open
                if year != -9999:
                    for key in gridmet.netcdf_variables: ncs[key].close()

                year = t.year
                ncs = {key: Dataset(f'{data_path}/{key}_{year}.nc') for key in gridmet.netcdf_variables}

            values = {key: ncs[key][var][t.timetuple().tm_yday - 1].flatten()[np.array(grid_df.index)] for key, var in gridmet.netcdf_variables.items()}

            for var in gridmet.netcdf_variables:
                if dfs[var].empty:
                    dfs[var] = pd.DataFrame([values[var]], columns=[f'grid_{k}' for k in np.array(grid_df.index)], index=[t])
                else:
                    dfs[var] = pd.concat([dfs[var], pd.DataFrame([values[var]], columns=[f'grid_{k}' for k in np.array(grid_df.index)], index=[t])])

            t += timedelta(days=1)
            progress_bar.update(1)

    for key in gridmet.netcdf_variables: ncs[key].close()

    return dfs


def generate_weather_files(data_path: str, weather_path: str, forcing: str, date_start: datetime, date_end: datetime, *,
    hourly: bool=False, locations: dict[str, tuple[float, float]] | list[tuple[float, float]] | None=None, header: bool=True) -> None:
    '''Generate weather files for the specified locations and date range. For each location, the nearest grid cell with valid data will be used.
    '''
    reanalysis = REANALYSES[forcing]
    grid_df = _initialize_weather_files(weather_path, forcing, locations, header=header, hourly=hourly)

    if forcing == 'gridMET':
        dfs = _process_gridmet(data_path, date_start, date_end, grid_df)
    elif forcing in ['GLDAS', 'NLDAS']:
        dfs = _process_xldas(data_path, reanalysis, date_start, date_end, grid_df, hourly=hourly)

    if hourly:
        weather_data = {key: func(dfs, hourly) for key, func in reanalysis.weather_file_variables.items() if WEATHER_FILE_VARIABLES[key].hourly}
    else:
        weather_data = {key: func(dfs, hourly) for key, func in reanalysis.weather_file_variables.items() if WEATHER_FILE_VARIABLES[key].daily}
    
    _write_weather_files(weather_path, weather_data, grid_df, hourly=hourly)
