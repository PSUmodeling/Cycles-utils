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
from scipy.interpolate import interp1d
from tqdm import tqdm

pt = os.path.dirname(os.path.realpath(__file__))

Coordinate = tuple[float, float]
LocationInput = Sequence[Coordinate] | Mapping[str, Coordinate] | None

class Resolution(Enum):
    DAILY = 1
    HOURLY = 2

@dataclass
class ReanalysisDataMixin:
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
    start_time: datetime
    la1: float
    lo1: float
    di: float
    dj: float
    netcdf_variables: dict[str, str]
    weather_file_variables: dict[str, Callable]

    def nearest_grid_index(self, lat, lon) -> int:
        return np.ravel_multi_index((round((lat - self.la1) / self.dj), round((lon - self.lo1) / self.di)), self.netcdf_shape)


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
        la1=-59.875,
        lo1=-179.875,
        di=0.25,
        dj=0.25,
        netcdf_variables={
            'precipitation': 'Rainf_f_tavg',
            'air temperature': 'Tair_f_inst',
            'specific humidity': 'Qair_f_inst',
            'wind': 'Wind_f_inst',
            'solar': 'SWdown_f_tavg',
            'air pressure': 'Psurf_f_inst',
        },
        weather_file_variables={
            'PP': lambda nc_data, resolution: nc_data['precipitation'] * 3600.0 if resolution is Resolution.HOURLY else nc_data['precipitation'].mean(axis=0, keepdims=True) * 86400.0,
            'TMP': lambda nc_data, _: nc_data['air temperature'] - 273.15,
            'TX': lambda nc_data, _: nc_data['air temperature'].max(axis=0, keepdims=True) - 273.15,
            'TN': lambda nc_data, _: nc_data['air temperature'].min(axis=0, keepdims=True) - 273.15,
            'SOLAR': lambda nc_data, resolution: nc_data['solar'] * 3600.0 * 1.0E-6 if resolution is Resolution.HOURLY else nc_data['solar'].mean(axis=0, keepdims=True) * 86400.0 * 1.0E-6,
            'RH': lambda nc_data, _: _relative_humidity(nc_data) * 100.0,
            'RHX': lambda nc_data, _: _relative_humidity(nc_data).max(axis=0, keepdims=True) * 100.0,
            'RHN': lambda nc_data, _: _relative_humidity(nc_data).min(axis=0, keepdims=True) * 100.0,
            'WIND': lambda nc_data, resolution: nc_data['wind'] if resolution is Resolution.HOURLY else nc_data['wind'].mean(axis=0, keepdims=True),
        }
    ))
    gridMET = astuple(ReanalysisDataMixin(
        url='http://www.northwestknowledge.net/metdata/data/',
        netcdf_extension=None,
        netcdf_prefix=None,
        netcdf_suffix=None,
        netcdf_shape=(585, 1386),
        data_interval=24,
        # For gridMET, land mask and elevation are the same file
        land_mask_file=os.path.join(pt, '../data/gridMET_elevation_mask.nc'),
        land_mask=lambda nc: nc['elevation'][:, :],
        elevation_file=os.path.join(pt, '../data/gridMET_elevation_mask.nc'),
        elevation=lambda nc: nc['elevation'][:, :],
        start_time=datetime.strptime('1979-01-01', '%Y-%m-%d'),
        la1=49.4,
        lo1=-124.76667,
        di=1.0/24.0,
        dj=-1.0/24.0,
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
            'PP': lambda nc_data, _: nc_data['pr'],
            'TX': lambda nc_data, _: nc_data['tmmx'] - 273.15,
            'TN': lambda nc_data, _: nc_data['tmmn'] - 273.15,
            'SOLAR': lambda nc_data, _: nc_data['srad'] * 86400.0 * 1.0E-6,
            'RHX': lambda nc_data, _: nc_data['rmax'],
            'RHN': lambda nc_data, _: nc_data['rmin'],
            'WIND': lambda nc_data, _: nc_data['vs'],
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
        la1=25.0625,
        lo1=-124.9375,
        di=0.125,
        dj=0.125,
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
            'PP': lambda nc_data, resolution: nc_data['precipitation'] if resolution is Resolution.HOURLY else nc_data['precipitation'].sum(axis=0, keepdims=True),
            'TMP': lambda nc_data, _: nc_data['air temperature'] - 273.15,
            'TX': lambda nc_data, _: nc_data['air temperature'].max(axis=0, keepdims=True) - 273.15,
            'TN': lambda nc_data, _: nc_data['air temperature'].min(axis=0, keepdims=True) - 273.15,
            'SOLAR': lambda nc_data, resolution: nc_data['solar'] * 3600.0 * 1.0E-6 if resolution is Resolution.HOURLY else nc_data['solar'].mean(axis=0, keepdims=True) * 86400.0 * 1.0E-6,
            'RH': lambda nc_data, _: _relative_humidity(nc_data) * 100.0,
            'RHX': lambda nc_data, _: _relative_humidity(nc_data).max(axis=0, keepdims=True) * 100.0,
            'RHN': lambda nc_data, _: _relative_humidity(nc_data).min(axis=0, keepdims=True) * 100.0,
            'WIND': lambda nc_data, resolution: _wind_speed(nc_data) if resolution is Resolution.HOURLY else _wind_speed(nc_data).mean(axis=0, keepdims=True),
        }
    ))

@dataclass
class WeatherFileDataMixin:
    fmt: Callable
    unit: str
    resolution: list[Resolution]

WEATHER_FILE_VARIABLES = {
    'YEAR': WeatherFileDataMixin(fmt=lambda x: '%-7.4d' % x, unit='####', resolution=[Resolution.DAILY, Resolution.HOURLY]),
    'DOY': WeatherFileDataMixin(fmt=lambda x: '%-7.3d' % x, unit='###', resolution=[Resolution.DAILY, Resolution.HOURLY]),
    'HOUR': WeatherFileDataMixin(fmt=lambda x: '%-7.2d' % x, unit='####', resolution=[Resolution.HOURLY]),
    'PP': WeatherFileDataMixin(fmt=lambda x: "%-#.5g" % x if x >= 1.0 else "%-.4f" % x, unit='mm', resolution=[Resolution.DAILY, Resolution.HOURLY]),
    'TMP': WeatherFileDataMixin(fmt=lambda x: '%-7.2f' % x, unit='degC', resolution=[Resolution.HOURLY]),
    'TX': WeatherFileDataMixin(fmt=lambda x: '%-7.2f' % x, unit='degC', resolution=[Resolution.DAILY]),
    'TN': WeatherFileDataMixin(fmt=lambda x: '%-7.2f' % x, unit='degC', resolution=[Resolution.DAILY]),
    'SOLAR': WeatherFileDataMixin(fmt=lambda x: '%-7.3f' % x, unit='MJ/m2', resolution=[Resolution.DAILY, Resolution.HOURLY]),
    'RH': WeatherFileDataMixin(fmt=lambda x: '%-7.2f' % x, unit='%', resolution=[Resolution.HOURLY]),
    'RHX': WeatherFileDataMixin(fmt=lambda x: '%-7.2f' % x, unit='%', resolution=[Resolution.DAILY]),
    'RHN': WeatherFileDataMixin(fmt=lambda x: '%-7.2f' % x, unit='%', resolution=[Resolution.DAILY]),
    'WIND': WeatherFileDataMixin(fmt=lambda x: '%-.2f' % x, unit='m/s', resolution=[Resolution.DAILY, Resolution.HOURLY]),
}

COOKIE_FILE = './.urs_cookies'


def download_forcing(data_path: Path | str, forcing: str, date_start: datetime | int, date_end: datetime | int) -> None:
    # Create data directory if it doesn't exist
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    if isinstance(date_start, int) and isinstance(date_end, int):
        date_start = datetime(date_start, 1, 1)
        date_end = datetime(date_end, 12, 31)

    assert isinstance(date_start, datetime) and isinstance(date_end, datetime)

    reanalysis = REANALYSIS[forcing]
    if reanalysis is REANALYSIS.gridMET:
        _download_gridmet(data_path, reanalysis, date_start.year, date_end.year)
    else:
        _download_xldas(data_path, reanalysis, date_start, date_end)


def find_grids(forcing: str, *, locations: LocationInput=None, screen_output: bool=True, remove_duplicates: bool=True) -> str | list[str]:
    """Find the nearest grid cell with valid data for each input location, and return a DataFrame with grid information
    and corresponding weather file names.
    """
    df = _find_grids(REANALYSIS[forcing], locations, screen_output=screen_output, remove_duplicates=remove_duplicates)

    return df['weather_file'].iloc[0] if len(df) == 1 else df['weather_file'].tolist()


def _find_grids(reanalysis: REANALYSIS, locations: LocationInput, screen_output: bool, remove_duplicates: bool) -> pd.DataFrame:
    mask_df = _read_land_mask(reanalysis)

    if locations is None:
        coordinates = None
        sites = None
        indices = list(mask_df[mask_df['mask'] > 0].index)
    else:
        if isinstance(locations, Mapping):
            coordinates = list(locations.values())
            sites = list(locations.keys())
        elif isinstance(locations, Sequence):
            coordinates = locations
            sites = None

        for coordinate in coordinates:
            if isinstance(coordinate, tuple) and len(coordinate) == 2:
                continue
            else:
                raise TypeError('Each location should be in the format of (latitude, longitude).')

        indices = [_calculate_grid_index(reanalysis, coordinate, mask_df) for coordinate in coordinates]

    df = pd.DataFrame({
        'grid_index': indices,
        'input_coordinate': coordinates if coordinates else [None] * len(indices),
        'site': sites if sites else [None] * len(indices),
    })

    df[['grid_latitude', 'weather_file', 'elevation']] = df.apply(
        lambda x: _get_grid_info(reanalysis, x['grid_index'], mask_df),
        axis=1,
        result_type='expand',
    )

    if locations is not None and remove_duplicates is True:
        df = _remove_duplicated_locations(reanalysis, df, screen_output)

    if screen_output is True:
        print(f"{reanalysis.name} weather files:")
        print(df[['site', 'input_coordinate', 'weather_file']].to_string(index=False))
        print()

    df.set_index('grid_index', inplace=True)

    return df


def _calculate_grid_index(reanalysis: REANALYSIS, coordinate: tuple[float, float], mask_df: pd.DataFrame) -> int:
    lat, lon = coordinate
    ind = reanalysis.nearest_grid_index(lat, lon)

    if mask_df.loc[ind]['mask'] == 0:   # type: ignore
        mask_df['distance'] = mask_df.apply(
            lambda x: np.sqrt((x['latitude'] - lat) ** 2 + (x['longitude'] - lon) ** 2),
            axis=1,
        )
        mask_df.loc[mask_df['mask'] == 0, 'distance'] = 1E6
        ind = mask_df['distance'].idxmin()

    return ind


def _remove_duplicated_locations(reanalysis: REANALYSIS, df: pd.DataFrame, screen_output: bool) -> pd.DataFrame:
    if any(df.duplicated(subset=['grid_index'])):
        indices = df['grid_index']
        if screen_output is True:
            print(f"The following input coordinates share {reanalysis.name} grids:")
            print(df[indices.isin(indices[indices.duplicated()])].sort_values('grid_index')[['input_coordinate', 'weather_file']].to_string(index=False))
            print()

    return df.drop_duplicates(subset=['grid_index'], keep='first')


def generate_weather_files(data_path: Path | str, weather_path: Path | str, forcing: str, date_start: datetime, date_end: datetime, *,
    hourly: bool=False, locations: LocationInput=None, header: bool=True) -> None:
    '''Generate weather files for the specified locations and date range. For each location, the nearest grid cell with
    valid data will be used.
    '''
    reanalysis = REANALYSIS[forcing]
    resolution = Resolution.HOURLY if hourly else Resolution.DAILY

    Path(weather_path).mkdir(parents=True, exist_ok=True)

    grid_df = _find_grids(reanalysis, locations, screen_output=False, remove_duplicates=True)

    if reanalysis is REANALYSIS.gridMET:
        weather_data = _process_gridmet(Path(data_path), date_start, date_end, grid_df)
    elif reanalysis in [REANALYSIS.GLDAS, REANALYSIS.NLDAS]:
        weather_data = _process_xldas(Path(data_path), reanalysis, date_start, date_end, grid_df, resolution)

    if resolution is Resolution.HOURLY:
        start = max(date_start, reanalysis.start_time)
        freq = '1h'
    elif resolution is Resolution.DAILY:
        start = date_start
        freq = '1d'

    time_index = pd.date_range(start=start, end=date_end + timedelta(days=1), freq=freq, inclusive='left')

    weather_data = {key: _interpolate_to_hourly(reanalysis, np.array(value)) if hourly and reanalysis.data_interval > 1 else np.array(value) for key, value in weather_data.items()}

    _write_weather_files(Path(weather_path), time_index, weather_data, grid_df, header, resolution)


def _download_xldas(data_path: Path, reanalysis: REANALYSIS, date_start: datetime, date_end: datetime) -> None:
    for d in tqdm(pd.date_range(start=date_start, end=date_end), desc=f'Download {reanalysis.name} files', unit=' days'):
        _download_daily_xldas(data_path, reanalysis, d)


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


def _write_header(f: typing.IO, latitude: float, elevation: float, resolution: Resolution, *, screening_height: float=10.0) -> None:
        # Open meteorological file and write header lines
        f.write('%-23s\t%.2f\n' % ('LATITUDE', latitude))
        f.write('%-23s\t%.2f\n' % ('ALTITUDE', elevation))
        f.write('%-23s\t%.1f\n' % ('SCREENING_HEIGHT', screening_height))
        f.write('\t'.join([f'{key:<7s}' for key, var in WEATHER_FILE_VARIABLES.items() if resolution in var.resolution]) + '\n')
        f.write('\t'.join([f'{var.unit:<7s}' for var in WEATHER_FILE_VARIABLES.values() if resolution in var.resolution]) + '\n')


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


def _write_weather_files(weather_path: Path | str, time_ts, weather_data: dict[str, np.ndarray], grid_df: pd.DataFrame, header: bool, resolution: Resolution) -> None:
    # Add time columns
    time_df = pd.DataFrame({
        'YEAR': time_ts.year,   # type: ignore
        'DOY': time_ts.map(lambda x: x.timetuple().tm_yday),   # type: ignore
    })
    if resolution is Resolution.HOURLY:
        time_df['HOUR'] = time_ts.hour   # type: ignore

    for ind, grid in enumerate(grid_df.index):
        # Choose variables for output
        output_df = pd.DataFrame({key: weather_data[key][:, ind] for key, var in WEATHER_FILE_VARIABLES.items() if resolution in var.resolution and not var.unit.startswith('#')})

        output_df = pd.concat([time_df, output_df.reset_index(drop=True)], axis=1).dropna()

        for c in output_df.columns:
            output_df[c] = output_df[c].map(WEATHER_FILE_VARIABLES[c].fmt)

        with open(f'{weather_path}/{grid_df.loc[grid, "weather_file"]}.{"hourly.weather" if resolution is Resolution.HOURLY else "weather"}', 'w' if header else 'a') as f:
            if header:
                _write_header(f, grid_df.loc[grid, 'grid_latitude'], grid_df.loc[grid, 'elevation'], resolution)   # type: ignore
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


def _process_xldas(data_path: Path, reanalysis: REANALYSIS, date_start: datetime, date_end: datetime, grid_df: pd.DataFrame, resolution: Resolution) -> dict[str, np.ndarray]:
    # Arrays to store daily values
    weather_data = {var: [] for var in reanalysis.weather_file_variables if resolution in WEATHER_FILE_VARIABLES[var].resolution}

    for d in tqdm(pd.date_range(start=date_start, end=date_end + timedelta(days=1), inclusive='left'), desc=f'Process {reanalysis.name} files', unit=' days'):
        _process_daily_xldas(data_path, reanalysis, d, grid_df, resolution, weather_data)

    weather_data = {key: np.array(value) for key, value in weather_data.items()}

    return weather_data


def _process_daily_xldas(data_path: Path, reanalysis: REANALYSIS, t: datetime, grid_df: pd.DataFrame, resolution: Resolution, weather_data: dict[str, list]) -> None:
    nc_data = {var: [] for var in reanalysis.netcdf_variables}
    for _t in pd.date_range(start=t, end=t + timedelta(days=1), freq=f'{reanalysis.data_interval}h', inclusive='left'):
        if _t < reanalysis.start_time:
            continue
        # Read one netCDF file
        with Dataset(data_path / f'{_t.strftime("%Y/%j")}/{reanalysis.netcdf_prefix}{_t.strftime("%Y%m%d.%H%M")}.{reanalysis.netcdf_suffix}') as nc:
            for nc_var in reanalysis.netcdf_variables:
                nc_data[nc_var].append(nc[reanalysis.netcdf_variables[nc_var]][0].flatten()[grid_df.index].tolist())

    nc_data = {key: np.array(value) for key, value in nc_data.items()}

    for weather_var, func in reanalysis.weather_file_variables.items():
        if resolution not in WEATHER_FILE_VARIABLES[weather_var].resolution:
            continue
        weather_data[weather_var] += func(nc_data, resolution).tolist()


def _process_gridmet(data_path: Path, date_start: datetime, date_end: datetime, grid_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Process annual gridMET data and write them to weather files
    """
    gridmet = REANALYSIS.gridMET

    nc_data = {var: [] for var in gridmet.netcdf_variables}

    year = -9999
    for d in tqdm(pd.date_range(start=date_start, end=date_end + timedelta(days=1), inclusive='left'), desc=f'Process gridMET files', unit=' days'):
        if d.year != year:
            # Close netCDF files that are open
            if year != -9999:
                for key in gridmet.netcdf_variables: ncs[key].close()

            year = d.year
            ncs = {key: Dataset(data_path/f'{key}_{year}.nc') for key in gridmet.netcdf_variables}

        for var in gridmet.netcdf_variables:
            nc_data[var].append(ncs[var][gridmet.netcdf_variables[var]][d.timetuple().tm_yday - 1].flatten()[np.array(grid_df.index)].tolist())

    nc_data = {key: np.array(value) for key, value in nc_data.items()}

    for key, values in nc_data.items():
        ncs[key].close()

    # Note that gridMET only provides daily data
    weather_data = {key: func(nc_data, None) for key, func in gridmet.weather_file_variables.items() if Resolution.DAILY in WEATHER_FILE_VARIABLES[key].resolution}

    return weather_data


def _interpolate_to_hourly(reanalysis: REANALYSIS, array: np.ndarray) -> np.ndarray:
    x0 = np.linspace(0, array.shape[0] - 1, array.shape[0])
    x = np.linspace(0, array.shape[0] - 1, array.shape[0] + (array.shape[0] - 1) * (reanalysis.data_interval - 1))

    f = interp1d(x0, array, axis=0, kind='linear')

    return f(x)
