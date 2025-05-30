import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


def conus_plot(gdf, column, *, projection=ccrs.PlateCarree(), cmap='viridis', title=None, vmin=None, vmax=None, extend='neither'):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_axes(
        [0.025, 0.09, 0.95, 0.93],
        projection=projection,
        frameon=False,
    )
    cax = fig.add_axes(
        [0.3, 0.07, 0.4, 0.02],
    )

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

    cbar = plt.colorbar(
        ax.collections[0],
        cax=cax,
        orientation='horizontal',
        extend=extend,
    )
    cbar.ax.tick_params(labelsize=14)
    if title is not None: cbar.set_label(title, size=16)
    cbar.ax.xaxis.set_label_position('top')

    return fig, ax


def yield_plot(harvest_df, *, ax=None, fontsize=None):
    if ax is None:
        _, ax = plt.subplots()

    if fontsize is not None: plt.rcParams.update({'font.size': fontsize})

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
            sub_df = harvest_df[(harvest_df['crop'] == c) & harvest_df[f'{h}_yield'] > 0 ]
            plt.plot(
                sub_df['date'], sub_df[f'{h}_yield'],
                harvests[h],
                color=_line.get_color(),
                alpha=0.8,
                ms=8,
            )

    # Set Y label
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

    # Add legend to figure
    ax.legend(handles=lh,
        #fontsize=fontsize,
        handletextpad=0,
        bbox_to_anchor=(1.0, 0.5),
        loc='center left',
        shadow=True,
        frameon=False,
    )


def operation_plot(operation_df, rotation_size, *, axes=None, fontsize=None):
    if axes is None:
        _, axes = plt.subplots(rotation_size, 1, sharex=True)
        if rotation_size == 1: axes = [axes]

    if fontsize is not None: plt.rcParams.update({'font.size': fontsize})

    operation_types = {
        'planting': [0, 'tab:green'],
        'tillage': [1, 'tab:blue'],
        'fertilization': [2, 'tab:purple'],
        'harvest': [3, 'tab:orange'],
        'kill': [4, 'tab:red'],
    }

    months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    mdoys = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

    for y in range(rotation_size):
        for op in operation_types:
            sub_df = operation_df[(operation_df['type'] == op) & (operation_df['year'] == y + 1)]

            if len(sub_df) == 0: continue

            label = op
            if op in ['planting', 'harvest', 'kill']:
                doys = sub_df['doy'].to_list()
                crops = sub_df['crop'].to_list()
                for i in range(len(crops)):
                    if crops[i].lower() == 'n/a':
                        crops[i] = 'All'
                for i in range(len(doys)):
                    label += f'\n{doys[i]}: {crops[i]}'
            elif op == 'tillage':
                doys = sub_df['doy'].to_list()
                tools = sub_df['tool'].to_list()
                for i in range(len(doys)):
                    label += f'\n{doys[i]}: {tools[i]}'
            elif op == 'fertilization':
                doys = sub_df['doy'].to_list()
                sources = sub_df['source'].to_list()
                for i in range(len(doys)):
                    label += f'\n{doys[i]}: {sources[i]}'

            axes[y].plot(
                sub_df['doy'],
                np.ones(sub_df['doy'].shape) * operation_types[op][0],
                'o',
                label=label,
                color=operation_types[op][1],
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
