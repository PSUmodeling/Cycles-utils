import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.pyplot as plt


def conus_plot(gdf, column, projection=ccrs.PlateCarree(), cmap='viridis', title=None, vmin=None, vmax=None, extend='neither'):
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
