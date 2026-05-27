from __future__ import annotations
import cartopy.crs as ccrs
import cartopy.feature as feature
import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.mpl.geoaxes import GeoAxes
from collections.abc import Sequence
from dataclasses import dataclass
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from pathlib import Path

HARVEST_MARKERS: dict[str, str] = {
    'grain': 'd',
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
    colors = {}
    for crop in crops:
        line, = ax.plot([], [])
        colors[crop] = line.get_color()
    return colors


def _plot_harvest_type(ax: Axes, df: pd.DataFrame, crop: str, harvest: str, marker: str, color: str) -> None:
    sub = df[(df['crop'] == crop) & (df[f'{harvest}_yield'] > 0)]
    ax.plot(
        sub['date'], sub[f'{harvest}_yield'],
        marker,
        color=color,
        alpha=0.8,
        ms=8,
    )


def _build_legend_handles(crops: list[str], crop_colors: dict[str, str]) -> list[mlines.Line2D]:
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


def plot_yield(harvest_df: pd.DataFrame, *, ax: Axes | None=None, fontsize: int | None=None) -> Axes:
    """Plot grain and forage yields by crop.

    Args:
        harvest_df: Harvest output DataFrame.
        ax: Optional axes to draw on.
        fontsize: Optional global font size override.

    Returns:
        Axes containing the yield plot.
    """
    if ax is None:
        _, ax = plt.subplots()
    if fontsize is not None:
        plt.rcParams.update({'font.size': fontsize})

    crops = harvest_df['crop'].unique().tolist()
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


def plot_operations(operations: list, rotation_size: int, *, axs: Axes | np.ndarray | None=None, fontsize: int | None=None):
    """Plot operations by day-of-year for each rotation year.

    Args:
        operations: Sequence of parsed operation objects.
        rotation_size: Number of years in the rotation.
        axs: Optional axes object(s) for rendering.
        fontsize: Optional global font size override.

    Returns:
        Axes array used to render timelines.
    """
    if axs is None:
        _, axs = plt.subplots(rotation_size, 1, sharex=True)
    assert axs is not None

    if isinstance(axs, Axes):
        axs = np.array(axs).reshape((1,))

    if rotation_size != axs.shape[0]:
        raise ValueError('The number of axes must match the rotation size.')

    if fontsize is not None: plt.rcParams.update({'font.size': fontsize})

    for y in range(rotation_size):
        for key, value in OPERATION_TYPES.items():
            sub_list = [op for op in operations if type(op).__name__.lower() == key and op.year == y + 1]

            if len(sub_list) == 0: continue

            axs[y].plot(
                [op.doy if not op.relative_doy else op.resolved_doy for op in sub_list], [value.yloc] * len(sub_list),
                'o',
                label=value.title + ':\n' + '\n'.join(f'{op.doy if not op.relative_doy else op.resolved_doy}: {getattr(op, value.label)}' if value.label is not None else f'{op.doy}' for op in sub_list),
                color=value.color,
                ms=10,
            )

        axs[y].set_xlim(-1, 370)
        axs[y].grid(False)
        axs[y].spines['right'].set_color('none')
        axs[y].spines['left'].set_color('none')
        axs[y].yaxis.set_ticks_position('none')
        axs[y].yaxis.set_tick_params(left=False, right=False, which='both', labelleft=False)
        axs[y].set_ylim(-3, 7)
        axs[y].text(184, 5, f'Year {y + 1}', ha='center')

        # set the y-spine
        axs[y].spines['bottom'].set_position('zero')

        # turn off the top spine/ticks
        axs[y].spines['top'].set_color('none')
        axs[y].xaxis.tick_bottom()
        axs[y].set_xticks(MDOYS)
        axs[y].set_xticklabels(MONTHS)

        handles, _ = axs[y].get_legend_handles_labels()
        if handles:
            axs[y].legend(
                loc='center left',
                bbox_to_anchor=(1.1, 0.5),
                ncols=5,
                frameon=False,
            )

    return axs


def plot_map(gdf: gpd.GeoDataFrame, column: str, *, projection: ccrs.Projection=ccrs.PlateCarree(), ax: Sequence[float] | GeoAxes | None=None,
    cmap: Colormap | str='viridis', vmin: float | None=None, vmax: float | None=None,
    colorbar: bool=True, cb_axes: tuple[float, float, float, float] | None=None, extend: str='neither', cb_orientation: str='horizontal',
    label: str | None=None, title: str | None=None,
    fontsize: float | None=None,
    frameon: bool=False) -> tuple[Figure, GeoAxes]:
    """Render a thematic map from a GeoDataFrame column.

    Args:
        gdf: GeoDataFrame to visualize.
        column: Column name to visualize.
        projection: Map projection for output axes.
        ax: Existing GeoAxes or add_axes rectangle.
        cmap: Matplotlib colormap.
        vmin: Optional lower bound for colormap normalization.
        vmax: Optional upper bound for colormap normalization.
        colorbar: Whether to draw a colorbar.
        cb_axes: Optional colorbar axes rectangle.
        extend: Colorbar extension mode.
        cb_orientation: Colorbar orientation.
        label: Optional colorbar label.
        title: Optional plot title.
        fontsize: Optional global font size override.
        frameon: Whether to draw map frame and grid labels.

    Returns:
        Tuple of figure and GeoAxes.
    """

    if fontsize is not None: plt.rcParams.update({'font.size': fontsize})

    if ax is None:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_axes((0.025, 0.09, 0.95, 0.93), projection=projection, frameon=frameon)    # type: ignore
    elif isinstance(ax, Sequence):
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_axes(ax, projection=projection, frameon=frameon)   # type: ignore
    elif isinstance(ax, GeoAxes):
        fig = ax.get_figure()

    if colorbar is True:
        cax = fig.add_axes((0.3, 0.07, 0.4, 0.02) if cb_axes is None else cb_axes)  # type: ignore

    gdf.plot(
        column=column,
        cmap=cmap,
        ax=ax,  # type: ignore
        vmin=vmin,
        vmax=vmax,
    )
    ax.add_feature(feature.STATES, edgecolor=[0.7, 0.7, 0.7], linewidth=0.5)    # type: ignore
    ax.add_feature(feature.LAND, facecolor=[0.8, 0.8, 0.8])     # type: ignore
    ax.add_feature(feature.LAKES)   # type: ignore
    ax.add_feature(feature.OCEAN)   # type: ignore

    if frameon:
        gl = ax.gridlines(      # type: ignore
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
            ax.collections[0],  # type: ignore
            cax=cax,
            orientation=cb_orientation,
            extend=extend,
        )
        if label is not None: cbar.set_label(label)
        cbar.ax.xaxis.set_label_position('top' if cb_orientation == 'horizontal' else 'right')  # type: ignore
    if title is not None:
        ax.set_title(title) # type: ignore

    return fig, ax  # type: ignore
