import io
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from matplotlib.axes import Axes
from .cycles_tools import read_soil as _read_soil
from .cycles_tools import read_weather as _read_weather
from .cycles_tools import read_output as _read_output
from .cycles_tools import read_operations as _read_operations
from .cycles_tools import plot_yield as _plot_yield
from .cycles_tools import plot_operations as _plot_operations

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

@dataclass
class Output:
    data: pd.DataFrame
    units: dict[str, str]


class Cycles:
    def __init__(self, path: str | Path, simulation: str):
        self.path: Path = Path(path)
        self.simulation: str | None = simulation
        self.output: dict[str, Output] = {}
        self.control: dict[str, Any] = {}
        self.operations: pd.DataFrame = pd.DataFrame()
        self.soil_profile: pd.DataFrame = pd.DataFrame()
        self.curve_number: int | None = None
        self.slope: float | None = None
        self.weather: pd.DataFrame = pd.DataFrame()


    def read_output(self, output_type: str) -> None:
        df, units = _read_output(self.path / 'output' / self.simulation, output_type)
        self.output[output_type] = Output(data=df, units=units)


    def read_control(self) -> None:
        with open(self.path / 'input' / f'{self.simulation}.ctrl') as f:
            lines = f.read().splitlines()

        lines = [line for line in lines if (not line.strip().startswith('#')) and len(line.strip()) > 0]

        control: dict[str, Any] = {line.strip().split()[0].lower(): line.strip().split()[1] for line in lines}

        if len(control['simulation_start_date']) > 4:
            control['simulation_start_date'] = datetime.strptime(control['simulation_start_date'], '%Y-%m-%d')
        else:
            control['simulation_start_date'] = datetime.strptime(control['simulation_start_date'] + '-01-01', '%Y-%m-%d')
        if len(control['simulation_end_date']) > 4:
            control['simulation_end_date'] = datetime.strptime(control['simulation_end_date'], '%Y-%m-%d')
        else:
            control['simulation_end_date'] = datetime.strptime(control['simulation_end_date'] + '-12-31', '%Y-%m-%d')
        control['rotation_size'] = int(control['rotation_size'])

        self.control = control
    

    def read_operations(self) -> None:
        if not self.control:
            self.read_control()

        self.operations = _read_operations(self.path / 'input' / self.control["operation_file"])


    def read_soil(self) -> None:
        if not self.control:
            self.read_control()
        soil = self.control['soil_file']

        self.soil_profile, self.curve_number, self.slope = read_soil(self.path / 'input' / soil)


    def read_weather(self, *, start_year: int=0, end_year: int=9999, subdaily: bool=False) -> None:
        if not self.control:
            self.read_control()
        weather = self.control['weather_file']

        self.weather = read_weather(self.path / 'input' / weather, start_year=start_year, end_year=end_year, subdaily=subdaily)


    def plot_yield(self, *, ax: Axes | None=None, fontsize: int | None=None) -> Axes:
        if 'harvest' not in self.output:
            self.read_output('harvest')
        
        return _plot_yield(self.output['harvest'], ax=ax, fontsize=fontsize)


    def plot_operations(self, rotation_size: int | None=None, *, axes: Axes | np.ndarray | None=None, fontsize: int | None=None):
        if self.operations.empty:
            self.read_operations()
        
        if rotation_size is None:
            if not self.control:
                self.read_control()
            rotation_size = int(self.control['rotation_size'])

        return _plot_operations(self.operations, rotation_size, axes=axes, fontsize=fontsize)


def _read_operation_parameter(type: type, line_no: int, lines: list[str]) -> str:
    return type(lines[line_no].split()[1])
