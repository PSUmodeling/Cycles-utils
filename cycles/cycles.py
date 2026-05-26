from __future__ import annotations
import numpy as np
import os
import pandas as pd
import subprocess
from collections.abc import Collection
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
    """Interface for executing one Cycles simulation and reading its files.

    Provides methods to run simulations, read outputs, and inspect soil/weather/operation
    configurations. Automatically loads the control file upon initialization.

    Attributes:
        path: Path to the simulation directory (containing input/ and output/ subdirs).
        simulation: Name of the simulation (base name of control file without extension).
        output: Dictionary mapping output table names to Output objects with data and units.
        control: Parsed control file configuration from the input directory.
        operations: List of parsed operation records from the operation file.
        soil_profile: List of SoilLayer objects describing the soil profile.
        curve_number: Runoff curve number for hydrologic calculations.
        slope: Land slope used in erosion and runoff models.
        weather: DataFrame of weather forcing data (temperature, precipitation, etc.).
        executable: Absolute path to the Cycles executable binary.
    """

    path: Path | str
    simulation: str
    output: dict[str, Output] = field(default_factory=dict[str, Output])
    control: ControlConfig | None = None
    operations: list | None = None
    soil_profile: list[SoilLayer] | None = None
    curve_number: int | None = None
    slope: float | None = None
    weather: pd.DataFrame | None = None
    executable: Path | str | None = None


    def __post_init__(self):
        self.path = Path(self.path)
        assert isinstance(self.path, Path)
        self.control = _read_control_file(self.path / 'input' / f'{self.simulation}.ctrl')
        if self.executable is not None:
            self.executable = str(Path(self.executable).resolve())


    def run(self, options: str, silence: bool=False) -> tuple[int, str]:
        """Run the Cycles executable for this simulation.

        Args:
            options: Command-line options passed to Cycles.
            silence: If True, suppress stdout and stderr printing.

        Returns:
            A tuple with process return code and stdout text.
        """
        cmd = [self.executable, *(options.split() if options else []), self.simulation]
        result = subprocess.run(
            cmd,
            shell=os.name == 'nt',
            capture_output=True,
            text=True,
        )
        if not silence:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.returncode, result.stdout


    def read_output(self, output_types: Collection) -> None:
        """Read one or more output tables into memory.

        Args:
            output_types: Output table name or collection of names.
        """
        assert isinstance(self.path, Path)
        if isinstance(output_types, str):
            output_types = output_types,

        for output_type in output_types:
            df, units = _read_output(self.path / 'output' / self.simulation, output_type)
            self.output[output_type] = Output(data=df, units=units)


    def read_operation_file(self) -> None:
        """Load operation records defined in the control file."""
        assert isinstance(self.path, Path)
        assert self.control is not None
        self.operations = _read_operation_file(self.path / 'input' / self.control.input_files.operation_file)


    def read_soil_file(self) -> None:
        """Load soil profile layers and metadata from the configured soil file."""
        assert isinstance(self.path, Path)
        assert self.control is not None
        self.soil_profile, meta = _read_soil_file(self.path / 'input' / self.control.input_files.soil_file)
        self.curve_number = meta['curve_number']
        self.slope = meta['slope']


    def read_weather_file(self, *, start_year: int=-9999, end_year: int=9999, subdaily: bool=False) -> None:
        """Read weather forcing data for the configured weather file.

        Args:
            start_year: Inclusive first year to keep.
            end_year: Inclusive last year to keep.
            subdaily: If True, parse hourly format instead of daily.
        """
        assert isinstance(self.path, Path)
        assert self.control is not None
        self.weather = _read_weather_file(self.path / 'input' / self.control.input_files.weather_file, start_year=start_year, end_year=end_year, subdaily=subdaily)


    def generate_reinit_file(self, doy: int, *, reinit: str | None=None) -> None:
        """Generate a reinitialization file from model output.

        Args:
            doy: Day-of-year to extract from reinit output.
            reinit: Optional output stem for the reinit file.
        """
        assert isinstance(self.path, Path)
        _generate_reinit_file(self.path / 'input' / f'{self.simulation if reinit is None else reinit}.reinit', self.path / 'output' / self.simulation, doy)


    def plot_yield(self, *, ax: Axes | None=None, fontsize: int | None=None) -> Axes:
        """Plot grain and forage yields from harvest output."""
        if 'harvest' not in self.output:
            self.read_output('harvest')

        return _plot_yield(self.output['harvest'].data, ax=ax, fontsize=fontsize)


    def plot_operations(self, rotation_size: int | None=None, *, axs: Axes | np.ndarray | None=None, fontsize: int | None=None):
        """Plot operation timelines grouped by rotation year.

        Args:
            rotation_size: Number of years in the plotted rotation.
            axs: Optional axes object(s) to draw on.
            fontsize: Global matplotlib font size override.

        Returns:
            The axes used for plotting.
        """
        if self.operations is None:
            self.read_operation_file()

        assert self.control is not None
        assert self.operations is not None

        if rotation_size is None:
            rotation_size = self.control.simulation_years.rotation_size

        return _plot_operations(self.operations, rotation_size, axs=axs, fontsize=fontsize)
