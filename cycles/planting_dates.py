import itertools
import numpy as np
import pandas as pd
from collections.abc import Callable
from scipy.signal import argrelextrema

MONTHS = {
    '01': [1, 31],
    '02': [32, 59],
    '03': [60, 90],
    '04': [91, 120],
    '05': [121, 151],
    '06': [152, 181],
    '07': [182, 212],
    '08': [213, 243],
    '09': [244, 273],
    '10': [274, 304],
    '11': [305, 334],
    '12': [335, 366],
}

DAYS_IN_A_MONTH = 30
DAYS_IN_A_WEEK = 7

class Crop:
    def __init__(self, *, name: str, maximum_temperature: float, minimum_temperature: float, base_temperature: float, reference_precipitation: float, reference_temperature: float, func: Callable):
        self.name = name
        self.maximum_temperature = maximum_temperature
        self.minimum_temperature = minimum_temperature
        self.base_temperature = base_temperature
        self.reference_precipitation = reference_precipitation
        self.reference_temperature = reference_temperature
        self.func = func


    def planting_date(self, *args):
        return self.func(self, *args)


def _find_month(doy: int) -> str:
    for key, value in MONTHS.items():
        if value[0] <= doy <= value[1]: return key


def _normalize_doy(doy: int) -> int:
    while doy >= 365:
        doy -= 365

    while doy < 0:
        doy += 365

    return doy


def _reorder_doy(df: pd.DataFrame, ref_doy: int) -> pd.DataFrame:
    df['DOY'] = df.index
    df['DOY'] = df['DOY'].map(lambda x: x + 365 if x < ref_doy else x)
    df = df.set_index('DOY').sort_index()

    return df


def _moving_average(array: np.array, left_window: int, right_window: int, median: bool=False) -> np.array:
    ma_array = []

    _array = np.append(array, array[0:right_window])

    func = np.mean if median == False else np.median

    for k in range(len(array)):
        mask = np.r_[k - left_window : k + right_window + 1]
        if k + right_window + 1 >= len(array):
            ma_array.append(func(_array[mask]))
        else:
            ma_array.append(func(array[mask]))

    return np.array(ma_array)


def _calculate_slope(array: np.array, left_window: int):
    slope = []
    for k in range(len(array)):
        if left_window == 0:
            slope.append(array[k] - array[k - 1])
        else:
            slope.append(np.polyfit(list(range(left_window + 1)), array[np.r_[k - left_window : k + 1]], 1)[0])

    return np.array(slope)


def _start_of_rain_seasons(precipitation: np.array):
    # Find days with (local) minimum precipitation
    doys = argrelextrema(precipitation, np.less_equal, order=182)[0]

    # Day 0 or 364 could be identified as local minimum, which needs to be checked
    if 0 in doys and precipitation[0] > precipitation[-1]: doys = np.delete(doys, 0)
    if 364 in doys and precipitation[-1] >= precipitation[0]: doys = np.delete(doys, -1)

    # Remove consecutive days
    if len(doys) > 2: doys = doys[np.concatenate(([True], np.diff(doys) != 1))]

    if len(doys) == 1: return [_normalize_doy(doys[0] + 1)]

    # Find the start of the primary rain season, i.e., the day with a local minimum precipitation that is before the
    # primary rain season (maximum precipitation)
    doy_max_precipitation = np.argmax(precipitation)
    adjusted_doys = [d - 365 if d >= doy_max_precipitation else d for d in doys]
    primary_doy = _normalize_doy(np.max(adjusted_doys) + 1)

    return [primary_doy]


def select_maturity_type(crop, thermal_time):
    hybrid, _ = min(CROPS[crop]['relative_maturity_types'].items(), key=lambda x: abs(thermal_time * 0.85 - x[1]))

    return hybrid


def _planting_date_control(monthly_temperature: np.array, monthly_precipitation: np.array, reference_precipitation: float, reference_temperature: float) -> str:
    if np.any(monthly_precipitation < reference_precipitation):      # 100 mm
        return 'temperature' if np.any(monthly_temperature < reference_temperature) else 'precipitation'
    else:
        return 'temperature'


def _corn_sorghum_planting_date(self: Crop, doy_weather_df: pd.DataFrame) -> tuple[list[int], str]: #temperature: list[float], precipitation: list[float]):
    MOVING_AVERAGE_HALF_WINDOW = 45
    SLOPE_WINDOW = 7

    control = _planting_date_control(
        doy_weather_df.groupby('month').mean()['temperature'].values,
        doy_weather_df.groupby('month').sum()['PP'].values,
        self.reference_precipitation,
        self.reference_temperature,
    )

    temperature_smoothed = _moving_average(doy_weather_df['temperature'].values, MOVING_AVERAGE_HALF_WINDOW, MOVING_AVERAGE_HALF_WINDOW)
    temperature_moving_average = _moving_average(temperature_smoothed, DAYS_IN_A_WEEK, 0)

    precipitation_smoothed = _moving_average(doy_weather_df['PP'].values, MOVING_AVERAGE_HALF_WINDOW, MOVING_AVERAGE_HALF_WINDOW)
    precipitation_moving_average = _moving_average(precipitation_smoothed, 0, DAYS_IN_A_MONTH)

    # Calculate moving averages of temperature and precipitation, and their slopes
    temperature_slope = _calculate_slope(temperature_moving_average, SLOPE_WINDOW)
    temperature_slope = _moving_average(temperature_slope, MOVING_AVERAGE_HALF_WINDOW, MOVING_AVERAGE_HALF_WINDOW)

    precipitation_slope = _calculate_slope(precipitation_moving_average, SLOPE_WINDOW)
    precipitation_slope = _moving_average(precipitation_slope, MOVING_AVERAGE_HALF_WINDOW, MOVING_AVERAGE_HALF_WINDOW)

    daily_precipitation_max = np.max(precipitation_moving_average)

    temperature_levels = [
        self.minimum_temperature + 5.0,
        self.minimum_temperature + 4.0,
        self.minimum_temperature + 3.0,
        self.minimum_temperature + 2.0,
        self.minimum_temperature + 1.0,
        self.minimum_temperature,
    ]

    #precipitation_conditions = list(itertools.product(temperature_levels, [min(100 / 30.0, 0.67 * daily_precipitation_max)]))
    #precipitation_conditions = list(itertools.product(temperature_levels, [min(80 / 30.0, 0.67 * daily_precipitation_max)]))
    #precipitation_conditions = list(itertools.product(temperature_levels, [min(60 / 30.0, 0.67 * daily_precipitation_max)]))
    #precipitation_conditions = list(itertools.product(temperature_levels, [min(40 / 30.0, 0.67 * daily_precipitation_max)]))
    #precipitation_conditions += list(itertools.product(temperature_levels, [0]))
    precipitation_conditions = list(itertools.product(temperature_levels, [max(120 / 30.0, 0.67 * daily_precipitation_max)]))
    precipitation_conditions += list(itertools.product(temperature_levels, [100 / 30.0, 80 / 30.0]))
    precipitation_conditions += list(itertools.product(temperature_levels, [60 / 30.0, 40 / 30.0]))
    precipitation_conditions += list(itertools.product(temperature_levels, [0]))

    df = pd.DataFrame(
        {
            'precipitation': precipitation_moving_average,
            'precipitation_slope': precipitation_slope,
            'temperature': temperature_moving_average,
            'temperature_slope': temperature_slope,
        },
        index=range(1, 366),
    )

    ## Calculate planting date
    ## TODO: Add two-planting-season predictions
    #if n_plantings == 2: return np.nan

    ## Temperature limited planting date
    if control == 'temperature':
        df = _reorder_doy(df, df['temperature'].idxmin())
        for temperature in temperature_levels:
            try:
                return _normalize_doy(df[(df['temperature'] >= temperature) & (df['temperature_slope'] > 0)].index[0]), control
            except:
                pass
        else:
            return np.nan, control
    ## Precipitation limited planting date
    else:
        # Find start of rainy season
        doy = _start_of_rain_seasons(df['precipitation'].values)

        df = _reorder_doy(df, doy[0])
        for temperature, precipitation in precipitation_conditions:
            try:
                return _normalize_doy(df[(df['temperature'] >= temperature) & (df['precipitation_slope'] > 0) & (df['precipitation'] > precipitation)].index[0]), control
            except:
                pass
        else:
            for temperature in temperature_levels:
                try:
                    return _normalize_doy(df[(df['temperature'] >= temperature) & (df['temperature_slope'] > 0)].index[0]), control
                except:
                    pass
            else:
                return np.nan, control


def calculate_thermal_time(crop: str, daily_temperature: pd.Series) -> float:

    _daily_thermal_time = lambda x: 0.0 if x < CROPS[crop].base_temperature else x - CROPS[crop].base_temperature

    return daily_temperature.map(_daily_thermal_time).sum()


CROPS = {
    'maize': Crop(
        name='maize',
        maximum_temperature=-999,
        minimum_temperature=12,
        base_temperature=6,
        reference_precipitation=100,
        reference_temperature=10,
        func=_corn_sorghum_planting_date,
    ),
    'sorghum': Crop(name='sorghum',
        maximum_temperature=-999,
        minimum_temperature=13,
        base_temperature=6,
        reference_precipitation=100,
        reference_temperature=11,
        func=_corn_sorghum_planting_date,
    ),
}


def calculate_planting_date(crop: str, weather_df: pd.DataFrame) -> dict:
    weather_df['temperature'] = 0.5 * (weather_df['TX'] + weather_df['TN'])

    doy_weather_df = weather_df.groupby('DOY').mean().drop(366)
    doy_weather_df['month'] = doy_weather_df.index.map(lambda x: _find_month(x))

    thermal_time = calculate_thermal_time(crop, doy_weather_df['temperature'])

    planting_date, control = CROPS[crop].planting_date(doy_weather_df)

    return {
        'total_precipitation': doy_weather_df['PP'].sum(),
        'thermal_time': thermal_time,
        'control': control,
        'planting_date': planting_date,
        #'relative_maturity': select_maturity_type(crop, thermal_time)
    }
