import cartopy.crs as ccrs
import cartopy.feature as feature
import geopandas as gpd
import io
import matplotlib.axes
import matplotlib.colors
import matplotlib.figure
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.mpl.geoaxes import GeoAxes
from dataclasses import dataclass
from matplotlib.axes import Axes
from pathlib import Path

SOIL_PARAMETERS = ['clay', 'sand', 'soc', 'bulk_density', 'coarse_fragments', 'pH']
SOIL_LAYERS = [
    # units: top (m), bottom (m), thickness (m), NO3 (kg/ha), NH4 (Kg/ha)
    {'top': 0, 'bottom': 0.05, 'thickness': 0.05, 'no3': 10, 'nh4': 1},
    {'top': 0.05, 'bottom': 0.1, 'thickness': 0.05, 'no3': 10, 'nh4': 1},
    {'top': 0.1, 'bottom': 0.2, 'thickness': 0.1, 'no3': 7, 'nh4': 1},
    {'top': 0.2, 'bottom': 0.4, 'thickness': 0.2, 'no3': 4, 'nh4': 1},
    {'top': 0.4, 'bottom': 0.6, 'thickness': 0.2, 'no3': 2, 'nh4': 1},
    {'top': 0.6, 'bottom': 0.8, 'thickness': 0.2, 'no3': 1, 'nh4': 1},
    {'top': 0.8, 'bottom': 1.0, 'thickness': 0.2, 'no3': 1, 'nh4': 1},
    {'top': 1.0, 'bottom': 1.2, 'thickness': 0.2, 'no3': 1, 'nh4': 1},
    {'top': 1.2, 'bottom': 1.4, 'thickness': 0.2, 'no3': 1, 'nh4': 1},
    {'top': 1.4, 'bottom': 1.6, 'thickness': 0.2, 'no3': 1, 'nh4': 1},
    {'top': 1.6, 'bottom': 1.8, 'thickness': 0.2, 'no3': 1, 'nh4': 1},
    {'top': 1.8, 'bottom': 2.0, 'thickness': 0.2, 'no3': 1, 'nh4': 1},
]
CURVE_NUMBERS = {
    #row crops, SR, Good
    'A': 67,
    'B': 78,
    'C': 85,
    'D': 89,
}
CONTROL_PARAMETERS = {
    'simulation years': {
        'simulation_start_date': None,
        'simulation_end_date': None,
        'rotation_size': None,
    },
    'other input files': {
        'crop_file': 'GenericCrops.crop',
        'operation_file': None,
        'soil_file': None,
        'weather_file': None,
        'reinit_file': 'N/A',
    },
    'simulation options': {
        'soil_layers': None,
        'co2_level': -999,
        'use_reinitialization': 0,
        'adjusted_yields': 0,
        'hourly_infiltration': 1,
        'automatic_nitrogen': 0,
        'automatic_phosphorus': 0,
        'automatic_sulfur': 0,
    },
    'output control': {
        'daily_weather_out': 0,
        'daily_crop_out': 0,
        'daily_residue_out': 0,
        'daily_water_out': 0,
        'daily_nitrogen_out': 0,
        'daily_soil_carbon_out': 0,
        'daily_soil_lyr_cn_out': 0,
        'annual_soil_out': 0,
        'annual_profile_out': 0,
        'annual_nflux_out': 0,
    }
}
CALIBRATION_PARAMETERS = {
    'calibration multipliers': {
        'soc_decomp_rate' : 1.0,
        'residue_decomp_rate' : 1.0,
        'root_decomp_rate' : 1.0,
        'rhizo_decomp_rate' : 1.0,
        'manure_decomp_rate' : 1.0,
        'ferment_decomp_rate' : 1.0,
        'microb_decomp_rate' : 1.0,
        'soc_humif_power' : 1.0,
        'nitrif_rate' : 1.0,
        'pot_denitrif_rate' : 1.0,
        'denitrif_half_rate' : 1.0,
        'decomp_half_resp' : 1.0,
        'decomp_resp_power' : 1.0,
        'root_progression' : 1.0,
        'radiation_use_efficiency' : 1.0,
    },
    'parameter values': {
        'kd_no3' : 0.0,
        'kd_nh4' : 5.6,
    }
}


def _overlapping_depth(top1, bottom1, top2, bottom2):
    return max(0.0, min(bottom1, bottom2) - max(top1, top2))


def _calculate_parameter(soil_df, parameter, top, bottom):
    soil_df['weight'] = soil_df.apply(lambda x: _overlapping_depth(x['top'], x['bottom'], top, bottom) / (bottom - top), axis=1)
    soil_df = soil_df[soil_df['weight'] > 0]

    return np.sum(np.array(soil_df[parameter] * soil_df['weight'])) / sum(soil_df['weight'])


def generate_soil_file(fn: str | Path, soil_df: pd.DataFrame, *, desc: str='', hsg: str='', slope: float=0.0, soil_depth: float | None=None) -> None:
    layer_depths = np.array([layer['bottom'] for layer in SOIL_LAYERS])

    if soil_depth is not None:
        layer_depths = layer_depths[layer_depths <= soil_depth]

    soil_depth = min(layer_depths, key=lambda x: abs(x - soil_df.iloc[-1]['bottom']))

    df = pd.DataFrame({v: [layer[v] for layer in SOIL_LAYERS if layer['bottom'] <= soil_depth] for v in SOIL_LAYERS[0]})

    df['layer'] = range(1, len(df) + 1)

    for v in SOIL_PARAMETERS:
        df[v] = df.apply(lambda x: _calculate_parameter(soil_df, v, x['top'], x['bottom']), axis=1)

    cn = -999 if not hsg else CURVE_NUMBERS[hsg[0]]

    with open(Path(fn), 'w') as f:
        if desc: f.write(desc)

        f.write("%-15s\t%d\n" % ("CURVE_NUMBER", cn))
        f.write("%-15s\t%.2f\n" % ("SLOPE", slope))

        f.write(('%-7s\t'*14 + '%s\n') % ("LAYER", "THICK", "CLAY", "SAND", "SOC", "BD", "FC", "PWP", "SON", "NO3", "NH4", "ROCK", "BYP_H", "BYP_V", "pH"))

        f.write(('%-7s\t'*14 + '%s\n') % ("#", "m", "%", "%", "%", "Mg/m3", "m3/m3", "m3/m3", "kg/ha", "kg/ha", "kg/ha", "m3/m3", "-", "-", "-"))

        for _, row in df.iterrows():
            f.write('%-7d\t' % row['layer'])
            f.write('%-7.2f\t' % float(row['thickness']))
            f.write('%-7s\t' % '-999' if np.isnan(row['clay']) else '%-7.1f\t' % float(row['clay']))
            f.write('%-7s\t' % '-999' if np.isnan(row['sand']) else '%-7.1f\t' % float(row['sand']))
            f.write('%-7s\t' % '-999' if np.isnan(row['soc']) else '%-7.2f\t' % float(row['soc']))
            f.write('%-7s\t' % '-999' if np.isnan(row['bulk_density']) else '%-7.2f\t' % float(row['bulk_density']))
            f.write(('%-7d\t'*3) % (-999, -999, -999))
            f.write(('%-7.1f\t' * 2) % (float(row['no3']), float(row['nh4'])))
            f.write('%-7s\t' % '-999' if np.isnan(row['coarse_fragments']) else '%-7.2f\t' % float(row['coarse_fragments']))
            f.write(('%-7.1f\t' *2) % (0.0, 0.0))
            f.write('%s\n' % '-999' if np.isnan(row['pH']) else '%.1f\n' % float(row['pH']))


def generate_control_file(fn: str | Path, user_dict: dict) -> None:
    fn = Path(fn)
    with open(fn, 'w') as f:
        for block, parameters in CONTROL_PARAMETERS.items():
            f.write(f'## {block.upper()} ##\n')
            for name, value in parameters.items():
                if name in user_dict:
                    # Overwrite default values with user input values
                    value = user_dict[name]
                elif value is None:
                    if name == 'soil_layers':
                        value = _get_soil_layers(fn.parent / user_dict['soil_file'])
                    else:
                        raise KeyError(f'Parameter {name.upper()} must be defined')

                f.write('%-23s\t%s\n' % (name.upper(), str(value)))
            f.write('\n')


def generate_nudge_file(fn: str | Path, user_dict: dict) -> None:
    fn = Path(fn)
    with open(fn, 'w') as f:
        for block, parameters in CALIBRATION_PARAMETERS.items():
            f.write(f'## {block.upper()} ##\n')
            for name, value in parameters.items():
                if name in user_dict:
                    # Overwrite default values with user input values
                    value = user_dict[name]
                f.write('%-27s\t%s\n' % (name.upper(), str(value)))
            f.write('\n')


def _get_soil_layers(fn: Path) -> int:
    NUM_HEADER_LINES = 2

    try:
        with open(fn) as f:
            lines = f.read().splitlines()
    except:
        raise FileNotFoundError("Soil file was not found")

    return len([line for line in lines if (not line.strip().startswith('#')) and line.strip()]) - NUM_HEADER_LINES - 1


def read_soil(soil: str | Path) -> tuple[pd.DataFrame, int, float]:
    NUM_HEADER_LINES = 2

    with open(Path(soil)) as f:
        lines = f.read().splitlines()

    lines = [line for line in lines if (not line.strip().startswith('#')) and line.strip()]

    df = pd.read_csv(
        io.StringIO('\n'.join(lines[NUM_HEADER_LINES:])),
        sep=r'\s+',
        na_values='-999',
        index_col='LAYER',
    )

    return df, int(lines[0].split()[-1]), float(lines[1].split()[-1])


def read_weather(weather: str | Path, *, start_year: int=0, end_year: int=9999, subdaily: bool=False) -> pd.DataFrame:
    NUM_HEADER_LINES = 4

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
        Path(weather),
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

    return df[(df['YEAR'] <= end_year) & (df['YEAR'] >= start_year)]


def read_output(path: str | Path, output_type: str) -> tuple[pd.DataFrame, dict[str, str]]:
    with open(Path(path) / f'{output_type}.csv') as f:
        lines = f.read().splitlines()

    df = pd.read_csv(
        io.StringIO('\n'.join(lines)),
        comment='#',
    )

    for col in ['date', 'plant_date']:
        if col in df.columns: df[col] = pd.to_datetime(df[col])

    units = {col: lines[1].strip()[1:].split(',')[ind] for ind, col in enumerate(df.columns)}

    return df, units


def read_operations(operation: str | Path) -> pd.DataFrame:
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

    with open(Path(operation)) as f:
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

    return pd.DataFrame(operations)


def plot_yield(harvest_df: pd.DataFrame, *, ax: Axes | None=None, fontsize: int | None=None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()

    if fontsize is not None:
        plt.rcParams.update({'font.size': fontsize})

    #Get a list of crops
    crops = harvest_df['crop'].unique()

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
            sub_df = harvest_df[(harvest_df['crop'] == c) & (harvest_df[f'{h}_yield'] > 0) ]
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


def plot_operations(operation_df: pd.DataFrame, rotation_size: int, *, axes: Axes | np.ndarray | None=None, fontsize: int | None=None):
    @dataclass
    class OperationType:
        yloc: int
        color: str

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
            sub_df = operation_df[(operation_df['type'] == key) & (operation_df['year'] == y + 1)]

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


def plot_map(gdf: gpd.GeoDataFrame, column: str, *, projection: ccrs.Projection =ccrs.PlateCarree(), cmap: matplotlib.colors.Colormap | str='viridis',
        fig: matplotlib.figure.Figure | None=None, axes: tuple[float, float, float, float] | None=None,
        colorbar: bool=True, cb_axes: tuple[float, float, float, float] | None=None,
        title: str | None=None, vmin: float | None=None, vmax: float | None=None, extend: str='neither', cb_orientation: str='horizontal',
        fontsize: int | None=None, frameon: bool=False) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    if fontsize is not None: plt.rcParams.update({'font.size': fontsize})

    fig = plt.figure(figsize=(9, 6)) if fig is None else fig

    ax: GeoAxes = fig.add_axes(
        (0.025, 0.09, 0.95, 0.93) if axes is None else axes,
        projection=projection,
        frameon=frameon,
    )   # type: ignore
    if colorbar is True:
        cax = fig.add_axes((0.3, 0.07, 0.4, 0.02) if cb_axes is None else cb_axes)

    gdf.plot(
        column=column,
        cmap=cmap,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
    )
    ax.add_feature(feature.STATES, edgecolor=[0.7, 0.7, 0.7], linewidth=0.5)
    ax.add_feature(feature.LAND, facecolor=[0.8, 0.8, 0.8])
    ax.add_feature(feature.LAKES)
    ax.add_feature(feature.OCEAN)

    if frameon:
        gl = ax.gridlines(
            draw_labels=True,
            color='gray',
            dms=True,
            x_inline=False,
            y_inline=False,
            linestyle='--',
        )
        gl.bottom_labels = None # type: ignore
        gl.right_labels = None  # type: ignore

    if colorbar is True:
        cbar = plt.colorbar(
            ax.collections[0],
            cax=cax,
            orientation=cb_orientation,
            extend=extend,
        )
        if title is not None: cbar.set_label(title)
        cbar.ax.xaxis.set_label_position('top' if cb_orientation == 'horizontal' else 'right')  # type: ignore

    return fig, ax


def _read_operation_parameter(type: type, line_no: int, lines: list[str]) -> str:
    return type(lines[line_no].split()[1])
