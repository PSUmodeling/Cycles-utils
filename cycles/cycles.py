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
    def __init__(self, *, cycles_path: str | Path | None=None, input_path: str | Path | None=None, output_path: str | None=None, simulation: str | None=None):
        if cycles_path is None and (input_path is None or output_path is None):
            raise ValueError('Either cycles_path or both input_path and output_path must be provided.')

        self.input_path: Path = Path(input_path) if input_path is not None else Path(cycles_path) / 'input' # type: ignore
        self.output_path: Path = Path(output_path) if output_path is not None else Path(cycles_path) / 'output' # type: ignore
        self.simulation: str | None = simulation
        self.output: dict[str, Output] = {}
        self.control: dict[str, Any] = {}
        self.operations: pd.DataFrame = pd.DataFrame()
        self.rotation_size: int | None = None
        self.soil: pd.DataFrame = pd.DataFrame()


    def read_output(self, output_type: str) -> None:
        self._check_simulation_name()
        assert self.simulation is not None

        fname = self.output_path / self.simulation / f'{output_type}.csv'

        df = pd.read_csv(
            fname,
            comment='#',
        )

        for col in ['date', 'plant_date']:
            if col in df.columns: df[col] = pd.to_datetime(df[col])

        with open(fname) as f:
            lines = f.read().splitlines()

        units = {col: lines[1].strip()[1:].split(',')[ind] for ind, col in enumerate(df.columns)}

        self.output[output_type] = Output(data=df, units=units)


    def read_control(self) -> None:
        self._check_simulation_name()
        assert self.simulation is not None

        with open(self.input_path / f'{self.simulation}.ctrl') as f:
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

        fname = self.input_path / self.control["operation_file"]

        with open(fname) as f:
            lines = f.read().splitlines()
        lines = [line for line in lines if (not line.strip().startswith('#')) and line.strip()]

        operations = []
        k = 0
        while k < len(lines):
            match lines[k]:
                case 'FIXED_FERTILIZATION':
                    operations.append({
                        'type': 'fertilization',
                        'year': _read_operation_parameter(int, k + 1, lines),
                        'doy': _read_operation_parameter(int, k + 2, lines),
                        'source': _read_operation_parameter(str, k + 3, lines),
                        'mass': _read_operation_parameter(float, k + 4, lines),
                    })
                    k += 5
                case 'TILLAGE':
                    tool = _read_operation_parameter(str, k + 3, lines)
                    year = _read_operation_parameter(int, k + 1, lines)
                    doy = _read_operation_parameter(int, k + 2, lines)
                    crop = _read_operation_parameter(str, k + 7, lines)

                    if tool.strip().lower() in HARVEST_TOOLS:
                        operations.append({
                            'type': 'harvest',
                            'year': year,
                            'doy': doy,
                            'crop': crop,
                        })
                    elif tool.strip().lower() == 'kill_crop':
                        operations.append({
                            'type': 'kill',
                            'year': year,
                            'doy': doy,
                            'crop': crop,
                        })
                    else:
                        operations.append({
                            'type': 'tillage',
                            'year': year,
                            'doy': doy,
                            'tool': tool,
                        })
                    k += 8
                case 'PLANTING':
                    operations.append({
                        'type': 'planting',
                        'year': _read_operation_parameter(int, k + 1, lines),
                        'doy': _read_operation_parameter(int, k + 2, lines),
                        'crop': _read_operation_parameter(str, k + 8, lines),
                    })
                    k += 9
                case _:
                    k += 1

        self.operations = pd.DataFrame(operations)


    def read_soil(self, soil: str | None=None) -> None:
        NUM_HEADER_LINES = 3

        if soil is None:
            if not self.control:
                self.read_control()
            soil = self.control['soil_file']
        assert soil is not None

        with open(self.input_path / soil) as f:
            lines = f.read().splitlines()

        lines = [line for line in lines if (not line.strip().startswith('#')) and line.strip()]

        self.soil = pd.read_csv(
            io.StringIO('\n'.join(lines[NUM_HEADER_LINES:])),
            sep=r'\s+',
            na_values='-999',
            index_col='LAYER',
        )


    def read_weather(self, weather: str | None=None, *, start_year: int=0, end_year: int=9999, subdaily: bool=False) -> None:
        NUM_HEADER_LINES = 4

        if weather is None:
            if not self.control:
                self.read_control()
            weather = self.control['weather_file']
        assert weather is not None

        if subdaily:
            columns = {
                'YEAR': int,
                'DOY': int,
                'HOUR': int,
                'PP': float,
                'TMP': float,
                'SOLAR': float,
                'RH': float,
                'WIND': float,
            }
        else:
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
            self.input_path / weather,
            usecols=list(range(len(columns))),
            names=list(columns.keys()),
            comment='#',
            sep=r'\s+',
            na_values='-999',
        )
        df = df.iloc[NUM_HEADER_LINES:, :]
        df = df.astype(columns)
        if subdaily:
            df['date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['DOY'].astype(str) + ' ' + df['HOUR'].astype(str), format='%Y-%j %H')
        else:
            df['date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['DOY'].astype(str), format='%Y-%j')
        df.set_index('date', inplace=True)

        self.weather = df[(df['YEAR'] <= end_year) & (df['YEAR'] >= start_year)]


    def _check_simulation_name(self) -> None:
        if self.simulation is None:
            raise ValueError('Simulation name must be provided to read output files.')


    def plot_yield(self, *, ax: Axes | None=None, fontsize: int | None=None) -> Axes:
        if 'harvest' not in self.output:
            self.read_output('harvest')

        if ax is None:
            _, ax = plt.subplots()

        if fontsize is not None:
            plt.rcParams.update({'font.size': fontsize})

        #Get a list of crops
        crops = self.output['harvest'].data['crop'].unique()

        harvests = {
            'grain': 'd',
            'forage': 'o',
        }

        crop_colors = []
        for c in crops:
            _line, = plt.plot([], [])
            crop_colors.append(_line.get_color())

            for h in harvests:
                # Plot grain yield
                sub_df = self.output['harvest'].data[(self.output['harvest'].data['crop'] == c) & (self.output['harvest'].data[f'{h}_yield'] > 0) ]
                plt.plot(
                    sub_df['date'], sub_df[f'{h}_yield'],
                    harvests[h],
                    color=_line.get_color(),
                    alpha=0.8,
                    ms=8,
                )

        ax.set_ylabel('Crop yield (Mg ha$^{-1}$)')

        # Add grids
        ax.set_axisbelow(True)
        plt.grid(True, color="#93a1a1", alpha=0.2)

        # Add legend: colors for different crops and shapes for grain or forage
        lh = []
        lh.append(mlines.Line2D([], [],
            linestyle='',
            marker='d',
            label='Grain',
            mfc='None',
            color='k',
            ms=10,
        ))
        lh.append(mlines.Line2D([], [],
            linestyle='',
            marker='o',
            label='Forage',
            mfc='None',
            color='k',
            ms=10,
        ))

        for i, c in enumerate(crops):
            lh.append(mlines.Line2D([], [],
                linestyle='None',
                marker='s',
                label=c,
                color=crop_colors[i],
                alpha=0.8,
                ms=10,
            ))

        ax.legend(handles=lh,
            handletextpad=0,
            bbox_to_anchor=(1.0, 0.5),
            loc='center left',
            shadow=True,
            frameon=False,
        )

        return ax


    def plot_operations(self, rotation_size: int | None=None, *, axes: Axes | np.ndarray | None=None, fontsize: int | None=None):
        @dataclass
        class OperationType:
            yloc: int
            color: str

        if self.operations.empty:
            self.read_operations()

        if rotation_size is None:
            if not self.control:
                self.read_control()
            rotation_size = int(self.control['rotation_size'])

        if axes is None:
            _, axes = plt.subplots(rotation_size, 1, sharex=True)
        assert axes is not None

        if isinstance(axes, Axes):
            axes = np.array(axes).reshape((1,))

        if rotation_size != axes.shape[0]:
            raise ValueError('The number of axes must match the rotation size.')

        if fontsize is not None: plt.rcParams.update({'font.size': fontsize})

        operation_types = {
            'planting': OperationType(0, 'tab:green'),
            'tillage': OperationType(1, 'tab:blue'),
            'fertilization': OperationType(2, 'tab:purple'),
            'harvest': OperationType(3, 'tab:orange'),
            'kill': OperationType(4, 'tab:red'),
        }

        months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        mdoys = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

        for y in range(rotation_size):
            for key, value in operation_types.items():
                sub_df = self.operations[(self.operations['type'] == key) & (self.operations['year'] == y + 1)]

                if len(sub_df) == 0: continue

                label = key
                if key in ['planting', 'harvest', 'kill']:
                    doys = sub_df['doy'].to_list()
                    crops = sub_df['crop'].to_list()
                    for i in range(len(crops)):
                        if crops[i].lower() == 'n/a':
                            crops[i] = 'All'
                    for i in range(len(doys)):
                        label += f'\n{doys[i]}: {crops[i]}'
                elif key == 'tillage':
                    doys = sub_df['doy'].to_list()
                    tools = sub_df['tool'].to_list()
                    for i in range(len(doys)):
                        label += f'\n{doys[i]}: {tools[i]}'
                elif key == 'fertilization':
                    doys = sub_df['doy'].to_list()
                    sources = sub_df['source'].to_list()
                    for i in range(len(doys)):
                        label += f'\n{doys[i]}: {sources[i]}'

                axes[y].plot(
                    sub_df['doy'],
                    np.ones(sub_df['doy'].shape) * value.yloc,
                    'o',
                    label=label,
                    color=value.color,
                    ms=10,
                )

            axes[y].set_xlim(-1, 370)
            axes[y].grid(False)
            axes[y].spines['right'].set_color('none')
            axes[y].spines['left'].set_color('none')
            axes[y].yaxis.set_ticks_position('none')
            axes[y].yaxis.set_tick_params(left=False, right=False, which='both', labelleft=False)
            axes[y].set_ylim(-3, 6)
            axes[y].text(184, 3, f'Year {y + 1}', ha='center')

            # set the y-spine
            axes[y].spines['bottom'].set_position('zero')

            # turn off the top spine/ticks
            axes[y].spines['top'].set_color('none')
            axes[y].xaxis.tick_bottom()
            axes[y].set_xticks(mdoys)
            axes[y].set_xticklabels(months)

            handles, _ = axes[y].get_legend_handles_labels()
            if handles:
                axes[y].legend(
                    loc='center left',
                    bbox_to_anchor=(1.1, 0.5),
                    ncols=5,
                    frameon=False,
                )

        return axes


def _read_operation_parameter(type: type, line_no: int, lines: list[str]) -> str:
    return type(lines[line_no].split()[1])
