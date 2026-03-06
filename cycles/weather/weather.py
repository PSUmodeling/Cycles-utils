import numpy as np
import os
import pandas as pd
import subprocess
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import astuple, dataclass
from datetime import datetime, timedelta
from enum import Enum
from netCDF4 import Dataset
from pathlib import Path
from tqdm import tqdm

pt = os.path.dirname(os.path.realpath(__file__))

Coordinate = tuple[float, float]
LocationInput = Sequence[Coordinate] | Mapping[str, Coordinate] | None

@dataclass
class ReanalysisDataMixin:
    url: str
    netcdf_extension: str | None
    netcdf_prefix: str | None
    netcdf_suffix: str | None
    netcdf_shape: tuple[int, int]
    data_interval: int | None
    land_mask_file: Path | str
    land_mask: Callable
    elevation_file: Path | str
    elevation: Callable
    start_time: datetime
    ind_j: Callable
    ind_i: Callable
    netcdf_variables: dict[str, str]
    weather_file_variables: dict[str, Callable]

class REANALYSIS(ReanalysisDataMixin, Enum):
    GLDAS = astuple(ReanalysisDataMixin(
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
        start_time=datetime.strptime('2000-01-01 03:00', '%Y-%m-%d %H:%M'),
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
            'PP': lambda ts, nc_data, hourly: _interpolate_to_hourly(ts, nc_data['precipitation']) * 3600.0 if hourly else np.nanmean(_reshape_to_daily(nc_data['precipitation'], 3), axis=1) * 86400.0,
            'TMP': lambda ts, nc_data, _: _interpolate_to_hourly(ts, nc_data['air temperature']) - 273.15,
            'TX': lambda ts, nc_data, _: np.nanmax(_reshape_to_daily(nc_data['air temperature'], 3), axis=1) - 273.15,
            'TN': lambda ts, nc_data, _: np.nanmin(_reshape_to_daily(nc_data['air temperature'], 3), axis=1) - 273.15,
            'SOLAR': lambda ts, nc_data, hourly: _interpolate_to_hourly(ts, nc_data['solar']) * 3600.0 * 1.0E-6 if hourly else np.nanmean(_reshape_to_daily(nc_data['solar'], 3), axis=1) * 86400.0 * 1.0E-6,
            'RH': lambda ts, nc_data, _: _interpolate_to_hourly(ts, _relative_humidity(nc_data)) * 100.0,
            'RHX': lambda ts, nc_data, _: np.nanmax(_reshape_to_daily(_relative_humidity(nc_data), 3), axis=1) * 100.0,
            'RHN': lambda ts, nc_data, _: np.nanmin(_reshape_to_daily(_relative_humidity(nc_data), 3), axis=1) * 100.0,
            'WIND': lambda ts, nc_data, hourly: _interpolate_to_hourly(ts, nc_data['wind']) if hourly else np.nanmean(_reshape_to_daily(nc_data['wind'], 3), axis=1),
        }
    ))
    gridMET = astuple(ReanalysisDataMixin(
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
        start_time=datetime.strptime('1979-01-01', '%Y-%m-%d'),
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
            'PP': lambda _ts, nc_data, _hourly: nc_data['pr'],
            'TX': lambda _ts, nc_data, _hourly: nc_data['tmmx'] - 273.15,
            'TN': lambda _ts, nc_data, _hourly: nc_data['tmmn'] - 273.15,
            'SOLAR': lambda _ts, nc_data, _hourly: nc_data['srad'] * 86400.0 * 1.0E-6,
            'RHX': lambda _ts, nc_data, _hourly: nc_data['rmax'],
            'RHN': lambda _ts, nc_data, _hourly: nc_data['rmin'],
            'WIND': lambda _ts, nc_data, _hourly: nc_data['vs'],
        }
    ))
    NLDAS = astuple(ReanalysisDataMixin(
        url='https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0',
        netcdf_extension='nc',
        netcdf_prefix='NLDAS_FORA0125_H.A',
        netcdf_suffix='020.nc',
        netcdf_shape=(224, 464),
        data_interval=1,
        land_mask_file=os.path.join(pt, '../data/NLDAS_masks-veg-soil.nc4'),
        land_mask=lambda nc: nc['CONUS_mask'][0],
        elevation_file=os.path.join(pt, '../data/NLDAS_elevation.nc4'),
        elevation=lambda nc: nc['NLDAS_elev'][0],
        start_time=datetime.strptime('1979-01-01 13:00', '%Y-%m-%d %H:%M'),
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
            'PP': lambda _ts, nc_data, hourly: nc_data['precipitation'] if hourly else np.nansum(_reshape_to_daily(nc_data['precipitation'], 1), axis=1),
            'TMP': lambda _ts, nc_data, _hourly: nc_data['air temperature'] - 273.15,
            'TX': lambda _ts, nc_data, _hourly: np.nanmax(_reshape_to_daily(nc_data['air temperature'], 1), axis=1) - 273.15,
            'TN': lambda _ts, nc_data, _hourly: np.nanmin(_reshape_to_daily(nc_data['air temperature'], 1), axis=1) - 273.15,
            'SOLAR': lambda _ts, nc_data, hourly: nc_data['solar'] * 3600.0 * 1.0E-6 if hourly else np.nanmean(_reshape_to_daily(nc_data['solar'], 1), axis=1) * 86400.0 * 1.0E-6,
            'RH': lambda _ts, nc_data, _hourly: _relative_humidity(nc_data) * 100.0,
            'RHX': lambda _ts, nc_data, _hourly: np.nanmax(_reshape_to_daily(_relative_humidity(nc_data), 1), axis=1) * 100.0,
            'RHN': lambda _ts, nc_data, _hourly: np.nanmin(_reshape_to_daily(_relative_humidity(nc_data), 1), axis=1) * 100.0,
            'WIND': lambda _ts, nc_data, hourly: _wind_speed(nc_data) if hourly else np.nanmean(_reshape_to_daily(_wind_speed(nc_data), 1), axis=1),
        }
    ))


def _reshape_to_daily(nc_data: np.ndarray, interval: int) -> np.ndarray:
    return nc_data.reshape(-1, 24 // interval, nc_data.shape[1])

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


def download_forcing(data_path: Path | str, forcing: str, date_start: datetime | int, date_end: datetime | int) -> None:
    # Create data directory if it doesn't exist
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    if isinstance(date_start, int) and isinstance(date_end, int):
        date_start = datetime(date_start, 1, 1)
        date_end = datetime(date_end, 12, 31)

    reanalysis = REANALYSIS[forcing]
    if reanalysis is REANALYSIS.gridMET:
        _download_gridmet(data_path, reanalysis, date_start.year, date_end.year)
    else:
        _download_xldas(data_path, reanalysis, date_start, date_end)


def find_grids(forcing: str, *, locations: LocationInput=None, screen_output=True) -> pd.DataFrame:
    """Find the nearest grid cell with valid data for each input location, and return a DataFrame with grid information and corresponding weather file names.
    """
    _find_grids(REANALYSIS[forcing], locations, screen_output)


def _find_grids(reanalysis: REANALYSIS, locations: LocationInput, screen_output: bool) -> pd.DataFrame:
    mask_df = _read_land_mask(reanalysis)

    if locations is None:
        #indices = [ind for ind, row in mask_df.iterrows() if row['mask'] > 0]
        indices = list(mask_df[mask_df['mask'] > 0].index)
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
                    lambda x: np.sqrt((x['latitude'] - lat) ** 2 + (x['longitude'] - lon) ** 2),
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
        lambda x: _get_grid_info(reanalysis, x['grid_index'], mask_df),
        axis=1,
        result_type='expand',
    )

    if locations is not None:
        if any(df.duplicated(subset=['grid_index'])):
            indices = df['grid_index']
            if screen_output is True:
                print(f"The following input coordinates share {reanalysis.name} grids:")
                print(df[indices.isin(indices[indices.duplicated()])].sort_values('grid_index')[['input_coordinate', 'weather_file']].to_string(index=False))
                print()

        if screen_output is True:
            print(f"{reanalysis.name} weather files:")
            if not sites:
                print(df[['input_coordinate', 'weather_file']].to_string(index=False))
            else:
                print(df[['site', 'input_coordinate', 'weather_file']].to_string(index=False))
            print()

    df = df.drop_duplicates(subset=['grid_index'], keep='first')
    df.set_index('grid_index', inplace=True)

    return df


def generate_weather_files(data_path: Path | str, weather_path: Path | str, forcing: str, date_start: datetime, date_end: datetime, *,
    hourly: bool=False, locations: LocationInput=None, header: bool=True) -> None:
    '''Generate weather files for the specified locations and date range. For each location, the nearest grid cell with valid data will be used.
    '''
    reanalysis = REANALYSIS[forcing]
    grid_df = _initialize_weather_files(Path(weather_path), reanalysis, locations)

    if reanalysis is REANALYSIS.gridMET:
        nc_data = _process_gridmet(Path(data_path), date_start, date_end, grid_df)
    elif reanalysis in [REANALYSIS.GLDAS, REANALYSIS.NLDAS]:
        timestamps, nc_data = _process_xldas(Path(data_path), reanalysis, date_start, date_end, grid_df, hourly)

    if hourly:
        weather_data = {key: func(timestamps, nc_data, hourly) for key, func in reanalysis.weather_file_variables.items() if WEATHER_FILE_VARIABLES[key].hourly}
    else:
        weather_data = {key: func(None, nc_data, hourly) for key, func in reanalysis.weather_file_variables.items() if WEATHER_FILE_VARIABLES[key].daily}

    time_index = pd.date_range(start=date_start, end=date_end + timedelta(days=1), freq='1h' if hourly else '1d', inclusive='left')

    _write_weather_files(Path(weather_path), time_index, weather_data, grid_df, header, hourly)


def _download_xldas(data_path: Path, reanalysis: REANALYSIS, date_start: datetime, date_end: datetime) -> None:
    d = date_start
    with tqdm(total=(date_end - date_start).days + 1, desc=f'Download {reanalysis.name} files', unit=' days') as progress_bar:
        while d <= date_end:
            _download_daily_xldas(data_path, reanalysis, d)
            d += timedelta(days=1)
            progress_bar.update(1)


def _download_gridmet(data_path: Path, gridmet: REANALYSIS, year_start: int, year_end: int) -> None:
    """Download gridMET forcing files
    """
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


def _interpolate_to_hourly(time_index: pd.Series, array: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(array, index=time_index).astype(float)
    return df.resample('h').mean().interpolate(method='linear').values


def _download_daily_xldas(path: Path, reanalysis: REANALYSIS, day: datetime):
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


def _read_land_mask(reanalysis: REANALYSIS) -> pd.DataFrame:
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

    if reanalysis.name == 'gridMET':
        grid_df.loc[~grid_df['mask'].isna(), 'mask'] = 1
        grid_df.loc[grid_df['mask'].isna(), 'mask'] = 0

    grid_df['mask'] = grid_df['mask'].astype(int)

    return grid_df


def _get_grid_info(reanalysis: REANALYSIS, grid_ind: int, mask_df: pd.DataFrame) -> tuple[float, str, float]:
    grid_lat, grid_lon = mask_df.loc[grid_ind, ['latitude', 'longitude']]   # type: ignore
    grid_str = '%.3f%sx%.3f%s' % (abs(grid_lat), 'S' if grid_lat < 0.0 else 'N', abs(grid_lon), 'W' if grid_lon < 0.0 else 'E')

    fn = f'{reanalysis.name}_{grid_str}'

    return grid_lat, fn, mask_df.loc[grid_ind, 'elevation']     # type: ignore


def _write_header(f: typing.IO, latitude: float, elevation: float, hourly: bool, *, screening_height: float=10.0) -> None:
        # Open meteorological file and write header lines
        f.write('%-23s\t%.2f\n' % ('LATITUDE', latitude))
        f.write('%-23s\t%.2f\n' % ('ALTITUDE', elevation))
        f.write('%-23s\t%.1f\n' % ('SCREENING_HEIGHT', screening_height))
        if hourly:
            f.write('\t'.join([f'{key:<7s}' for key, var in WEATHER_FILE_VARIABLES.items() if var.hourly]) + '\n')
            f.write('\t'.join([f'{var.unit:<7s}' for var in WEATHER_FILE_VARIABLES.values() if var.hourly]) + '\n')
        else:
            f.write('\t'.join([f'{key:<7s}' for key, var in WEATHER_FILE_VARIABLES.items() if var.daily]) + '\n')
            f.write('\t'.join([f'{var.unit:<7s}' for var in WEATHER_FILE_VARIABLES.values() if var.daily]) + '\n')


def _relative_humidity(nc_data: dict[str, np.ndarray]) -> np.ndarray:
    air_temperature = nc_data['air temperature']
    air_pressure = nc_data['air pressure']
    specific_humidity = nc_data['specific humidity']

    es = 611.2 * np.exp(17.67 * (air_temperature - 273.15) / (air_temperature - 273.15 + 243.5))
    ws = 0.622 * es / (air_pressure - es)
    w = specific_humidity / (1.0 - specific_humidity)
    rh = w / ws
    rh[rh > 1.0] = 1.0
    rh[rh < 0.01] = 0.01

    return rh


def _wind_speed(nc_data: dict[str, np.ndarray]) -> np.ndarray:
    return np.sqrt(nc_data['wind_u'] ** 2 + nc_data['wind_v'] ** 2)


def _read_xldas_netcdf(t: datetime, reanalysis: REANALYSIS, nc: Dataset, indices: np.ndarray, nc_data: dict[str, list]) -> None:
    """Read meteorological variables of an array of desired grids from netCDF

    The netCDF variable arrays are flattened to make reading faster
    """
    for var in reanalysis.netcdf_variables:
        nc_data[var].append(list(nc[reanalysis.netcdf_variables[var]][0].flatten()[indices]))


def _write_weather_files(weather_path: Path | str, time_ts, weather_data: dict[str, np.ndarray], grid_df: pd.DataFrame, header: bool, hourly: bool):
    # Add time columns
    time_df = pd.DataFrame({
        'YEAR': time_ts.year,   # type: ignore
        'DOY': time_ts.map(lambda x: x.timetuple().tm_yday),   # type: ignore
    })
    if hourly:
        time_df['HOUR'] = time_ts.hour   # type: ignore

    for ind, grid in enumerate(grid_df.index):
        # Choose variables for output
        if hourly:
            output_df = pd.DataFrame({key: weather_data[key][:, ind] for key, var in WEATHER_FILE_VARIABLES.items() if var.hourly and not var.unit.startswith('#')})
        else:
            output_df = pd.DataFrame({key: weather_data[key][:, ind] for key, var in WEATHER_FILE_VARIABLES.items() if var.daily and not var.unit.startswith('#')})

        output_df = pd.concat([time_df, output_df.reset_index(drop=True)], axis=1).dropna()

        for c in output_df.columns:
            output_df[c] = output_df[c].map(WEATHER_FILE_VARIABLES[c].fmt)

        with open(f'{weather_path}/{grid_df.loc[grid, "weather_file"]}.{"hourly.weather" if hourly else "weather"}', 'w' if header else 'a') as f:
            if header:
                _write_header(f, grid_df.loc[grid, 'grid_latitude'], grid_df.loc[grid, 'elevation'], hourly)   # type: ignore
            output_df.to_csv(
                f,
                sep='\t',
                header=False,
                index=False,
        )


def _initialize_weather_files(weather_path: Path, reanalysis: REANALYSIS, locations: LocationInput):
    weather_path.mkdir(parents=True, exist_ok=True)

    grid_df = _find_grids(reanalysis, locations, False)

    return grid_df


def _process_xldas(data_path: Path, reanalysis: REANALYSIS, date_start: datetime, date_end: datetime, grid_df: pd.DataFrame, hourly: bool) -> tuple(list(datetime), dict[str, np.ndarray]):
    # Arrays to store daily values
    nc_data = {var: [] for var in reanalysis.netcdf_variables}

    timestamps = []
    t = date_start
    with tqdm(total=(date_end - date_start).days + 1, desc=f'Process {reanalysis.name} files', unit=' days') as progress_bar:
        while t < date_end + timedelta(days=1):
            if t < reanalysis.start_time:
                for var in reanalysis.netcdf_variables:
                    nc_data[var].append(list(np.full(len(grid_df), np.nan)))
            else:
                # netCDF file name
                fn = f'{t.strftime("%Y/%j")}/{reanalysis.netcdf_prefix}{t.strftime("%Y%m%d.%H%M")}.{reanalysis.netcdf_suffix}'

                # Read one netCDF file
                with Dataset(data_path/fn) as nc:
                    _read_xldas_netcdf(t, reanalysis, nc, np.array(grid_df.index), nc_data)

            timestamps.append(t)
            t += timedelta(hours=reanalysis.data_interval)     # type: ignore
            if (t - date_start).total_seconds() % 86400 == 0: progress_bar.update(1)

    nc_data = {var: np.array(nc_data[var]) for var in nc_data}

    return timestamps, nc_data


def _process_gridmet(data_path: Path, date_start: datetime, date_end: datetime, grid_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Process annual gridMET data and write them to weather files
    """
    gridmet = REANALYSIS.gridMET

    nc_data = {var: [] for var in gridmet.netcdf_variables}

    year = -9999
    t = date_start
    with tqdm(total=(date_end - date_start).days + 1, desc=f'Process gridMET files', unit=' days') as progress_bar:
        while t < date_end + timedelta(days=1):
            if t.year != year:
                # Close netCDF files that are open
                if year != -9999:
                    for key in gridmet.netcdf_variables: ncs[key].close()

                year = t.year
                ncs = {key: Dataset(data_path/f'{key}_{year}.nc') for key in gridmet.netcdf_variables}

            for var in gridmet.netcdf_variables:
                nc_data[var].append(list(ncs[var][gridmet.netcdf_variables[var]][t.timetuple().tm_yday - 1].flatten()[np.array(grid_df.index)]))

            t += timedelta(days=1)
            progress_bar.update(1)

    for key in gridmet.netcdf_variables:
        nc_data[key] = np.array(nc_data[key])
        ncs[key].close()

    return nc_data
