import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from matplotlib.axes import Axes
from .cycles_tools import SoilLayer, ControlConfig
from .cycles_tools import read_control_file as _read_control_file
from .cycles_tools import read_soil_file as _read_soil_file
from .cycles_tools import read_weather_file as _read_weather_file
from .cycles_tools import read_output as _read_output
from .cycles_tools import read_operation_file as _read_operation_file
from .cycles_tools import generate_reinit_file as _generate_reinit_file
from .cycles_tools import plot_yield as _plot_yield
from .cycles_tools import plot_operations as _plot_operations

@dataclass
class Output:
    data: pd.DataFrame
    units: dict[str, str]

@dataclass
class Cycles:
    path: Path | str
    simulation: str
    output: dict[str, Output] = field(default_factory=dict[str, Output])
    control: ControlConfig | None = None
    operations: list | None = None
    soil_profile: list[SoilLayer] | None = None
    curve_number: int | None = None
    slope: float | None = None
    weather: pd.DataFrame | None = None

    def __post_init__(self):
        self.path = Path(self.path)
        self.control = _read_control_file(self.path / 'input' / f'{self.simulation}.ctrl')


    def read_output(self, output_type: str) -> None:
        df, units = _read_output(self.path / 'output' / self.simulation, output_type)
        self.output[output_type] = Output(data=df, units=units)


    def read_operation_file(self) -> None:
        self.operations = _read_operation_file(self.path / 'input' / self.control.input_files.operation_file)


    def read_soil_file(self) -> None:
        self.soil_profile, meta = _read_soil_file(self.path / 'input' / self.control.input_files.soil_file)
        self.curve_number = meta['curve_number']
        self.slope = meta['slope']


    def read_weather_file(self, *, start_year: int=0, end_year: int=9999, subdaily: bool=False) -> None:
        self.weather = _read_weather_file(self.path / 'input' / self.control.input_files.weather_file, start_year=start_year, end_year=end_year, subdaily=subdaily)


    def generate_reinit_file(self, doy: int, *, reinit: str | None=None) -> None:
        _generate_reinit_file(self.path / 'input' / f'{self.simulation if reinit is None else reinit}.reinit', self.path / 'output' / self.simulation, doy)


    def plot_yield(self, *, ax: Axes | None=None, fontsize: int | None=None) -> Axes:
        if 'harvest' not in self.output:
            self.read_output('harvest')

        return _plot_yield(self.output['harvest'].data, ax=ax, fontsize=fontsize)


    def plot_operations(self, rotation_size: int | None=None, *, axes: Axes | np.ndarray | None=None, fontsize: int | None=None):
        if self.operations is None:
            self.read_operation_file()

        if rotation_size is None:
            rotation_size = self.control.simulation_years.rotation_size

        return _plot_operations(self.operations, rotation_size, axes=axes, fontsize=fontsize)
