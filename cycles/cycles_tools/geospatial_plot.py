from __future__ import annotations
import cartopy.crs as ccrs
import cartopy.feature as feature
import cartopy.io.img_tiles as cimgt
import geopandas as gpd
import math
import matplotlib.pyplot as plt
from cartopy.mpl.geoaxes import GeoAxes
from collections.abc import Sequence
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from pathlib import Path
from shapely.geometry import Polygon
from ._geo_file import _read_geospatial_file

MapExtent = Sequence[float] | gpd.GeoDataFrame | Polygon | str | Path

def plot_map(gdf: gpd.GeoDataFrame, column: str, *, projection: ccrs.Projection=ccrs.PlateCarree(), ax: Sequence[float] | GeoAxes | None=None,
    cmap: Colormap | str='viridis', vmin: float | None=None, vmax: float | None=None,
    colorbar: bool=True, cb_axes: tuple[float, float, float, float] | None=None, extend: str='neither', cb_orientation: str='horizontal',
    label: str | None=None, title: str | None=None,
    fontsize: float | None=None,
    frameon: bool=False,
    **kwargs) -> tuple[Figure, GeoAxes]:
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
        **kwargs,
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
            spacing='proportional',
            extend=extend,
        )
        if label is not None: cbar.set_label(label)
        cbar.ax.xaxis.set_label_position('top' if cb_orientation == 'horizontal' else 'right')  # type: ignore
    if title is not None:
        ax.set_title(title) # type: ignore

    return fig, ax  # type: ignore


def _zoom_from_extent(extent, *, desired_pixels: int=1024) -> int:
    """Determine the appropriate zoom level for a given map extent.
    Number of global tiles at each zoom level is 2 ^ zoom_level
    """
    PIXELS_PER_TILE = 256

    dx = abs(extent[1] - extent[0])
    dy = abs(extent[3] - extent[2])

    return round(max(math.log(desired_pixels / PIXELS_PER_TILE * 180 / dy) / math.log(2), math.log(desired_pixels / PIXELS_PER_TILE * 360 / dx) / math.log(2)))


def _get_extent_from_geometry(geometry: gpd.GeoDataFrame | Polygon) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = geometry.to_crs('epsg:4326').total_bounds if isinstance(geometry, gpd.GeoDataFrame) else geometry.bounds
    xc = 0.5 * xmin + 0.5 * xmax
    yc = 0.5 * ymin + 0.5 * ymax

    return (
        (xmin - xc) * 1.2 + xc,
        (xmax - xc) * 1.2 + xc,
        (ymin - yc) * 1.2 + yc,
        (ymax - yc) * 1.2 + yc,
    )


def plot_satellite_map(bound: MapExtent, *,
    fig: Figure | None=None, figsize: tuple[float, float] | None=None, ax: Sequence[float] | None=None, desired_pixels: int=1024, alpha: float=1.0, style: str | None=None) -> tuple[GeoAxes, ccrs.Projection]:
    """Render a satellite basemap over a geographic extent.

    Args:
        bound: Geographic bounds as `(west, east, south, north)` in degrees, or a GeoDataFrame, Polygon, or path to a geospatial file.
        fig: Optional existing figure to which the map axes are added. If omitted, a new figure is created.
        figsize: Optional figure size passed to `matplotlib.pyplot.figure` when creating a new figure.
        alpha: Basemap transparency where `1.0` is fully opaque.
        ax: Optional axes rectangle passed to `Figure.add_axes`. If omitted, a full-figure subplot is created.
        desired_pixels: Target figure resolution in pixels used to estimate tile zoom.
        style: Optional Cartopy GoogleTiles style string. If omitted, ArcGIS World Imagery tiles are used.

    Returns:
        Tuple containing:
            - GeoAxes with the satellite basemap.
            - Projection associated with the tile source.
    """
    GOOGLE_URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}.jpg'

    google_satellite = cimgt.GoogleTiles(url=GOOGLE_URL) if style is None else cimgt.GoogleTiles(style=style)

    if fig is None:
        fig = plt.figure(figsize=figsize)

    if ax is None:
        ax = fig.add_subplot(1, 1, 1, projection=google_satellite.crs)  # type: ignore
    else:
        ax = fig.add_axes(ax, projection=google_satellite.crs)  # type: ignore
    assert isinstance(ax, GeoAxes)

    if isinstance(bound, (str, Path)):
        gdf = _read_geospatial_file(bound)
        extent = _get_extent_from_geometry(gdf)
    elif isinstance(bound, (gpd.GeoDataFrame, Polygon)):
        extent = _get_extent_from_geometry(bound)
    else:
        extent = bound

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(google_satellite, _zoom_from_extent(extent, desired_pixels=desired_pixels), alpha=alpha)

    return ax, google_satellite.crs
