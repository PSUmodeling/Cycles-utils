from __future__ import annotations

import geopandas as gpd
import pandas as pd
import rioxarray
import sys
import xarray
from dataclasses import dataclass
from owslib.wcs import WebCoverageService
from pathlib import Path
from pyproj import Transformer
from rasterio.enums import Resampling
from shapely.geometry import Point, Polygon
from cycles.cycles_tools import generate_soil_file as _generate_soil_file
from cycles.cycles_tools import SoilLayer, MAPPABLE_PARAMETERS

HOMOLOSINE = (
    'PROJCS["Interrupted_Goode_Homolosine",'
    'GEOGCS["GCS_unnamed ellipse",DATUM["D_unknown",SPHEROID["Unknown",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Interrupted_Goode_Homolosine"],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
    'AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
)

BBOX_BUFFER: float = 2.0   # degrees, used when deriving bbox from boundary polygon

@dataclass
class SoilGridsLayers:
    # units: m
    top: float
    bottom: float

    @property
    def thickness(self) -> float:
        return self.bottom - self.top

@dataclass
class SoilGridsProperties:
    soilgrids_name: str
    layers: list[str]
    multiplier: float
    unit: str

SOILGRIDS_LAYERS: dict[str, SoilGridsLayers] = {
    '0-5cm': SoilGridsLayers(0, 0.05),
    '5-15cm': SoilGridsLayers(0.05, 0.15),
    '15-30cm': SoilGridsLayers(0.15, 0.3),
    '30-60cm': SoilGridsLayers(0.3, 0.6),
    '60-100cm': SoilGridsLayers(0.6, 1.0),
    '100-200cm': SoilGridsLayers(1.0, 2.0),
}

_ALL_LAYERS = list(SOILGRIDS_LAYERS)

SOILGRIDS_PROPERTIES: dict[str, SoilGridsProperties] = {
    'clay': SoilGridsProperties('clay', _ALL_LAYERS, 0.1, '%'),
    'sand': SoilGridsProperties('sand', _ALL_LAYERS, 0.1, '%'),
    'soc': SoilGridsProperties('soc', _ALL_LAYERS, 0.01, '%'),
    'bulk_density': SoilGridsProperties('bdod', _ALL_LAYERS, 0.01, 'Mg/m3'),
    'coarse_fragments': SoilGridsProperties('cfvo', _ALL_LAYERS, 0.001, 'm3/m3'),
    'pH': SoilGridsProperties('phh2o', _ALL_LAYERS, 0.1, '-'),
    'organic_carbon_density': SoilGridsProperties('ocd', _ALL_LAYERS, 0.1, 'kg/m3'),
    'organic_carbon_stocks': SoilGridsProperties('ocs', ['0-30cm'], 1.0, 'Mg/ha'),
}

ALL_MAPS: list[str] = [f'{p}@{l}' for p in MAPPABLE_PARAMETERS for l in SOILGRIDS_LAYERS]

LatLon = tuple[float, float]


class SoilGrids:
    def __init__(self, path: str | Path, *, maps: list[str]=ALL_MAPS, crs: str | None=None, aggregated: int | None=None) -> None:
        if aggregated is not None and aggregated not in [1000, 5000]:
            raise ValueError(f'Invalid value for aggregated: {aggregated}. Supported values are 1000, and 5000.')
        self.crs: str = crs if crs is not None else HOMOLOSINE
        self.maps: dict[str, xarray.DataArray] = _read_maps(Path(path), maps, crs, aggregated)
        self.transformer = Transformer.from_crs('epsg:4326', self.crs, always_xy=True)
        self.matched_maps: pd.DataFrame | None = None


    def reproject_match(self, *, reference_xds: xarray.DataArray, reference_name: str, boundary: gpd.GeoDataFrame) -> None:
        reference_xds = reference_xds.rio.clip([boundary], from_disk=True)
        df = pd.DataFrame(reference_xds[0].to_series().rename(reference_name))

        for n, m in self.maps.items():
            reprojected = m.rio.reproject_match(reference_xds, resampling=Resampling.nearest)
            reprojected = reprojected.rio.clip([boundary], from_disk=True)
            multiplier = SOILGRIDS_PROPERTIES[n.split('@')[0]].multiplier
            soil_df = pd.DataFrame(reprojected[0].to_series().rename(n)) * multiplier
            df = pd.concat([df, soil_df], axis=1)

        self.matched_maps = df


    def get_soil_profile(self, lat_lon: LatLon) -> list[SoilLayer]:
        values = self._extract_values(lat_lon)

        return [SoilLayer(
            top=layer.top,
            bottom=layer.bottom,
            **{p: values[f'{p}@{key}'] for p in MAPPABLE_PARAMETERS},
        ) for key, layer in SOILGRIDS_LAYERS.items()]


    def generate_soil_file(self, fn: Path | str, lat_lon: LatLon, *, desc: str | None=None, hsg: str='', slope: float=0.0) -> None:
        profile: list[SoilLayer] = self.get_soil_profile(lat_lon)
        desc = desc if desc is not None else _build_desc(lat_lon, hsg)

        #if desc is None:
        #    desc = f"# Soil file sampled at Latitude {lat_lon[0]:.3f}, Longitude {lat_lon[1]:.3f}.\n" if lat_lon is not None else ""
        #    desc += "# NO3, NH4, and fractions of horizontal and vertical bypass flows are default empirical values.\n"
        #    if hsg == '':
        #        desc += "# Hydrologic soil group MISSING DATA.\n"
        #    else:
        #        desc += f"# Hydrologic soil group {hsg}.\n"
        #        desc += "# The curve number for row crops with straight row treatment is used.\n"

        _generate_soil_file(Path(fn), profile, desc=desc, hsg=hsg, slope=slope)


    def _extract_values(self, lat_lon: LatLon) -> dict[str, float]:
        x, y = self.transformer.transform(lat_lon[1], lat_lon[0])

        return {
            n: m.sel(x=x, y=y, method='nearest').values[0] * SOILGRIDS_PROPERTIES[n.split('@')[0]].multiplier for n, m in self.maps.items()
        }


def _build_desc(lat_lon: LatLon, hsg: str) -> str:
    """Build the default soil file description header."""
    lines = [
        f"# Soil file sampled at Latitude {lat_lon[0]:.3f}, Longitude {lat_lon[1]:.3f}.",
        "# NO3, NH4, and fractions of horizontal and vertical bypass flows are default empirical values.",
    ]
    if not hsg:
        lines.append("# Hydrologic soil group MISSING DATA.")
    else:
        lines.append(f"# Hydrologic soil group {hsg}.")
        lines.append("# The curve number for row crops with straight row treatment is used.")
    return '\n'.join(lines)


def _read_maps(path: Path, maps: list[str], crs: str | None, aggregated: int | None) -> dict[str, xarray.DataArray]:
    """Read SoilGrids data
    Map names follow the convention 'variable@layer', e.g. 'bulk_density@0-5cm'.
    map name for 0-5 cm bulk density should be "bulk_density@0-5cm".
    """
    soilgrids_xds = {}
    for m in maps:
        v, layer = m.split('@')
        prop = SOILGRIDS_PROPERTIES[v]
        agg_suffix = f'_mean_{aggregated}' if aggregated else ''
        fn = path / f'{prop.soilgrids_name}_{layer}{agg_suffix}.tif'
        xds = rioxarray.open_rasterio(fn, masked=True)
        soilgrids_xds[m] = xds.rio.reproject(crs) if crs is not None else xds

    return soilgrids_xds


def _get_bounding_box(bbox: tuple[float, float, float, float], crs) -> tuple[float, float, float, float]:
    """Convert bounding boxes to SoilGrids CRS

    bbox should be in the order of [west, south, east, north]
    """
    corners = gpd.GeoDataFrame(
        {'geometry': [Point(bbox[0], bbox[3]), Point(bbox[2], bbox[1])]},
        index=['NW', 'SE'],
        crs=crs,
    ).to_crs(HOMOLOSINE)

    return (
        corners.loc['NW', 'geometry'].xy[0][0],
        corners.loc['SE', 'geometry'].xy[1][0],
        corners.loc['SE', 'geometry'].xy[0][0],
        corners.loc['NW', 'geometry'].xy[1][0],
    )


def _bbox_from_boundary(boundary: Polygon) -> tuple[float, float, float, float]:
    """Derive a buffered bounding box from a boundary polygon."""
    w, s, e, n = boundary.bounds
    buf_x = min(BBOX_BUFFER, 0.5 * (e - w))
    buf_y = min(BBOX_BUFFER, 0.5 * (n - s))
    return (w - buf_x, s - buf_y, e + buf_x, n + buf_y)


def download_soilgrids_data(path: str | Path, *,
    maps: list[str]=ALL_MAPS, boundary: Polygon | None=None, bbox: tuple[float, float, float, float] | None = None, crs: str='epsg:4326') -> None:
    """Download SoilGrids data via WCS for a given bounding box or boundary polygon.
    bbox should be ordered [west, south, east, north].
    Map names follow the convention 'variable@layer', e.g. 'bulk_density@0-5cm'.
    """
    if boundary is not None and bbox is None:
        bbox = _bbox_from_boundary(boundary)
    if bbox is None:
        raise ValueError("Either boundary or bbox must be provided.")

    bbox = _get_bounding_box(bbox, crs)

    for m in maps:
        parameter, layer = m.split('@')
        v   = SOILGRIDS_PROPERTIES[parameter].soilgrids_name
        wcs = WebCoverageService(f'http://maps.isric.org/mapserv?map=/map/{v}.map', version='1.0.0')

        while True:
            try:
                response = wcs.getCoverage(
                    identifier=f'{v}_{layer}_mean',
                    crs='urn:ogc:def:crs:EPSG::152160',
                    bbox=bbox,
                    resx=250,
                    resy=250,
                    format='GEOTIFF_INT16',
                )
                (Path(path) / f'{v}_{layer}.tif').write_bytes(response.read())
                break
            except Exception:
                continue
