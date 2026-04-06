import cartopy.crs as ccrs
import cartopy.feature as feature
import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.mpl.geoaxes import GeoAxes
from collections.abc import Callable
from dataclasses import dataclass
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from pathlib import Path

HARVEST_MARKERS: dict[str, str] = {
    'grain':  'd',
    'forage': 'o',
}

YIELD_UNIT_LABEL: str = 'Crop yield (Mg ha$^{-1}$)'

@dataclass
class OperationType:
    yloc: int
    color: str
    label: str | None
    title: str

OPERATION_TYPES = {
    'planting': OperationType(0, 'tab:green', 'crop', 'Planting'),
    'tillage': OperationType(1, 'tab:cyan', 'tool', 'Tillage'),
    'fixedfertilization': OperationType(2, 'tab:purple', 'source', 'Fertilization'),
    'fixedirrigation': OperationType(3, 'tab:blue', None, 'Irrigation'),
    'harvest': OperationType(4, 'tab:orange', 'crop_name', 'Harvest'),
    'kill': OperationType(5, 'tab:red', 'crop_name', 'Kill'),
}

MONTHS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
MDOYS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

def _assign_crop_colors(crops: list[str], ax: Axes) -> dict[str, str]:
    """Assign a unique color to each crop by cycling through the axes color cycle."""
    colors = {}
    for crop in crops:
        line, = ax.plot([], [])
        colors[crop] = line.get_color()
    return colors


def _plot_harvest_type(
    ax: Axes,
    df: pd.DataFrame,
    crop: str,
    harvest: str,
    marker: str,
    color: str,
) -> None:
    sub = df[(df['crop'] == crop) & (df[f'{harvest}_yield'] > 0)]
    ax.plot(
        sub['date'], sub[f'{harvest}_yield'],
        marker,
        color=color,
        alpha=0.8,
        ms=8,
    )


def _build_legend_handles(crops: list[str], crop_colors: dict[str, str]) -> list[mlines.Line2D]:
    """Build legend handles: one per harvest type, one per crop."""
    marker_handles = [
        mlines.Line2D([], [],
            linestyle='',
            marker=marker,
            label=harvest.capitalize(),
            mfc='None',
            color='k',
            ms=10,
        ) for harvest, marker in HARVEST_MARKERS.items()
    ]
    crop_handles = [
        mlines.Line2D(
            [], [],
            linestyle='None',
            marker='s',
            label=crop,
            color=crop_colors[crop],
            alpha=0.8,
            ms=10,
        ) for crop in crops
    ]
    return marker_handles + crop_handles


def plot_yield(harvest_df: pd.DataFrame, *, ax: Axes | None = None, fontsize: int | None = None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    if fontsize is not None:
        plt.rcParams.update({'font.size': fontsize})

    crops       = harvest_df['crop'].unique().tolist()
    crop_colors = _assign_crop_colors(crops, ax)

    for crop in crops:
        for harvest, marker in HARVEST_MARKERS.items():
            _plot_harvest_type(ax, harvest_df, crop, harvest, marker, crop_colors[crop])

    ax.set_ylabel(YIELD_UNIT_LABEL)
    ax.set_axisbelow(True)
    ax.grid(True, color='#93a1a1', alpha=0.2)
    ax.legend(
        handles=_build_legend_handles(crops, crop_colors),
        handletextpad=0,
        bbox_to_anchor= (1.0, 0.5),
        loc='center left',
        shadow=True,
        frameon=False,
    )
    return ax


def plot_operations(operations: list, rotation_size: int, *, axes: Axes | np.ndarray | None=None, fontsize: int | None=None):
    if axes is None:
        _, axes = plt.subplots(rotation_size, 1, sharex=True)
    assert axes is not None

    if isinstance(axes, Axes):
        axes = np.array(axes).reshape((1,))

    if rotation_size != axes.shape[0]:
        raise ValueError('The number of axes must match the rotation size.')

    if fontsize is not None: plt.rcParams.update({'font.size': fontsize})

    for y in range(rotation_size):
        for key, value in OPERATION_TYPES.items():
            sub_list = [op for op in operations if type(op).__name__.lower() == key and op.year == y + 1]

            if len(sub_list) == 0: continue

            axes[y].plot(
                [op.doy for op in sub_list], [value.yloc] * len(sub_list),
                'o',
                label=value.title + ':\n' + '\n'.join(f'{op.doy}: {getattr(op, value.label)}' if value.label is not None else f'{op.doy}' for op in sub_list),
                color=value.color,
                ms=10,
            )

        axes[y].set_xlim(-1, 370)
        axes[y].grid(False)
        axes[y].spines['right'].set_color('none')
        axes[y].spines['left'].set_color('none')
        axes[y].yaxis.set_ticks_position('none')
        axes[y].yaxis.set_tick_params(left=False, right=False, which='both', labelleft=False)
        axes[y].set_ylim(-3, 7)
        axes[y].text(184, 5, f'Year {y + 1}', ha='center')

        # set the y-spine
        axes[y].spines['bottom'].set_position('zero')

        # turn off the top spine/ticks
        axes[y].spines['top'].set_color('none')
        axes[y].xaxis.tick_bottom()
        axes[y].set_xticks(MDOYS)
        axes[y].set_xticklabels(MONTHS)

        handles, _ = axes[y].get_legend_handles_labels()
        if handles:
            axes[y].legend(
                loc='center left',
                bbox_to_anchor=(1.1, 0.5),
                ncols=5,
                frameon=False,
            )

    return axes


def plot_map(gdf: gpd.GeoDataFrame, column: str, *, projection: ccrs.Projection=ccrs.PlateCarree(), ax: tuple[float, float, float, float] | GeoAxes | None=None,
    cmap: Colormap | str='viridis', vmin: float | None=None, vmax: float | None=None,
    colorbar: bool=True, cb_axes: tuple[float, float, float, float] | None=None, extend: str='neither', cb_orientation: str='horizontal',
    label: str | None=None, title: str | None=None,
    fontsize: float | None=None,
    frameon: bool=False) -> tuple[Figure, GeoAxes]:

    if fontsize is not None: plt.rcParams.update({'font.size': fontsize})

    if ax is None:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_axes((0.025, 0.09, 0.95, 0.93), projection=projection, frameon=frameon)
    elif isinstance(ax, tuple):
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_axes(ax, projection=projection, frameon=frameon)
    elif isinstance(ax, GeoAxes):
        fig = ax.get_figure()

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
        if label is not None: cbar.set_label(label)
        cbar.ax.xaxis.set_label_position('top' if cb_orientation == 'horizontal' else 'right')  # type: ignore
    if title is not None:
        ax.set_title(title)

    return fig, ax