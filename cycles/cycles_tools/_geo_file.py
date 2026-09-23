from __future__ import annotations
import fiona
import geopandas as gpd
import pandas as pd
from pathlib import Path

# Geospatial file readers (shapefile/KML boundary files).
# Isolated in this module so that fiona/geopandas are only imported when a
# geospatial feature is actually used (e.g. plot_tools), not by every
# consumer of the core Cycles file-format modules.
def read_geospatial_file(file_path: str | Path) -> gpd.GeoDataFrame:
    file_path = Path(file_path)
    ext = file_path.suffix.lstrip('.').lower()
    match ext:
        case 'shp':
            return gpd.read_file(file_path)
        case 'kml':
            return _read_kml(file_path)
        case _:
            raise ValueError(f"Unsupported boundary format: '.{ext}'")


def _read_kml(file_path: Path) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        pd.concat(
            [gpd.read_file(file_path, driver='KML', layer=layer) for layer in fiona.listlayers(file_path)],
            ignore_index=True,
        )
    )
