from __future__ import annotations
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from matplotlib.axes import Axes

HARVEST_MARKERS: dict[str, str] = {
    'grain': 'd',
    'forage': 'o',
}

YIELD_UNIT_LABEL: str = 'Crop yield (Mg/ha)'

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


def _plot_harvest_type(ax: Axes, df: pd.DataFrame, crop: str, harvest: str, marker: str, color: str, **kwargs) -> None:
    sub = df[(df['crop'] == crop) & (df[f'{harvest}_yield'] > 0)]
    ax.plot(
        sub['date'], sub[f'{harvest}_yield'],
        marker,
        color=color,
        alpha=0.8,
        ms=8,
        **kwargs,
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


def plot_yield(harvest_df: pd.DataFrame, *, ax: Axes | None=None, crop_colors: dict | None=None, fontsize: int | None=None, **kwargs) -> Axes:
    """Plot grain and forage yields by crop.

    Args:
        harvest_df: Harvest output DataFrame.
        ax: Optional axes to draw on.
        crop_colors: Optional mapping from crop names to plot colors. If omitted, colors are assigned from the axes' color cycle.
        fontsize: Optional global font size override.
        **kwargs: Options to pass to the matplotlib plotting method.

    Returns:
        Axes containing the yield plot.
    """
    if ax is None:
        _, ax = plt.subplots()
    if fontsize is not None:
        plt.rcParams.update({'font.size': fontsize})

    crops = harvest_df['crop'].unique().tolist()
    if crop_colors is None:
        crop_colors = _assign_crop_colors(crops, ax)

    for crop in crops:
        for harvest, marker in HARVEST_MARKERS.items():
            _plot_harvest_type(ax, harvest_df, crop, harvest, marker, crop_colors[crop], **kwargs)

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


def plot_operations(operations: list, rotation_size: int, *, axs: Axes | np.ndarray | None=None, fontsize: int | None=None, **kwargs) -> np.ndarray:
    """Plot operations by day-of-year for each rotation year.

    Args:
        operations: Sequence of parsed operation objects.
        rotation_size: Number of years in the rotation.
        axs: Optional axes object(s) for rendering.
        fontsize: Optional global font size override.
        **kwargs: Additional options passed to matplotlib plotting method.

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
            if not sub_list: continue

            axs[y].plot(
                [op.doy if not op.relative_doy else op.resolved_doy for op in sub_list], [value.yloc] * len(sub_list),
                'o',
                label=value.title + ':\n' + '\n'.join(
                    f'{op.doy if not op.relative_doy else op.resolved_doy}: {getattr(op, value.label)}' if value.label is not None else f'{op.doy}' for op in sub_list
                ),
                color=value.color,
                ms=10,
                **kwargs,
            )

            if key == 'planting':
                for op in sub_list:
                    if op.end_doy == -999: continue
                    axs[y].fill_betweenx([value.yloc - 0.5, value.yloc + 0.5], op.doy, op.end_doy, alpha=0.25, color=value.color)

        axs[y].set_xlim(-1, 370)
        axs[y].grid(False)
        axs[y].yaxis.set_ticks_position('none')
        axs[y].yaxis.set_tick_params(left=False, right=False, which='both', labelleft=False)
        axs[y].set_ylim(-3, 7)
        axs[y].text(184, 5, f'Year {y + 1}', ha='center')
        for spine in ('right', 'left', 'top'):
            axs[y].spines[spine].set_color('none')

        # set the y-spine
        axs[y].spines['bottom'].set_position('zero')

        # turn off the top spine/ticks
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
