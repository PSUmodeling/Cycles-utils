import geopandas as gpd
import os
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cycles_tools import generate_soil_file as _generate_soil_file
from cycles_tools import SOIL_PARAMETERS

@dataclass
class SoilGridsProperties:
    soilgrids_name: str
    layers: list[str]
    multiplier: float
    unit: str

SOILGRIDS_PROPERTIES = {
    'clay': SoilGridsProperties('clay', ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm'], 0.1, '%'),
    'sand': SoilGridsProperties('sand', ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm'], 0.1, '%'),
    'soc': SoilGridsProperties('soc', ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm'], 0.01, '%'),
    'bulk_density': SoilGridsProperties('bdod', ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm'], 0.01, 'Mg/m3'),
    'coarse_fragments': SoilGridsProperties('cfvo', ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm'], 0.001, 'm3/m3'),
    'pH': SoilGridsProperties('phh2o', ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm'], 0.1, '-'),
    'organic_carbon_density': SoilGridsProperties('ocd', ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm'], 0.1, 'kg/m3'),
    'organic_carbon_stocks': SoilGridsProperties('ocs', ['0-30cm'], 1.0, 'Mg/ha'),
}

@dataclass
class SoilGridsLayers:
    # units: m
    top: float
    bottom: float
    thickness: float

SOILGRIDS_LAYERS = {
    '0-5cm': SoilGridsLayers(0, 0.05, 0.05),
    '5-15cm': SoilGridsLayers(0.05, 0.15, 0.10),
    '15-30cm': SoilGridsLayers(0.15, 0.3, 0.15),
    '30-60cm': SoilGridsLayers(0.3, 0.6, 0.3),
    '60-100cm': SoilGridsLayers(0.6, 1.0, 0.4),
    '100-200cm': SoilGridsLayers(1.0, 2.0, 1.0),
}

HOMOLOSINE = 'PROJCS["Interrupted_Goode_Homolosine",' \
    'GEOGCS["GCS_unnamed ellipse",DATUM["D_unknown",SPHEROID["Unknown",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],' \
    'PROJECTION["Interrupted_Goode_Homolosine"],' \
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],' \
    'AXIS["Easting",EAST],AXIS["Northing",NORTH]]'

ALL_MAPS = [f'{parameter}@{layer}' for parameter in SOIL_PARAMETERS for layer in SOILGRIDS_LAYERS]


class SoilGrids:
    def __init__(self, path: str | Path, *, maps: list[str]=ALL_MAPS, crs: str | None=None):
        self.maps: dict[str, xarray.DataArray] = _read_soilgrids_maps(Path(path), maps, crs)
        self.crs: str = crs if crs is not None else HOMOLOSINE
        self.matched_maps: df.DataFrame | None = None
        self.soil_profile: df.DataFrame | None = None


    def reproject_match_soilgrids_maps(self, *, reference_xds: xarray.DataArray, reference_name: str, boundary: gpd.GeoDataFrame) -> pd.DataFrame:
        reference_xds = reference_xds.rio.clip([boundary], from_disk=True)
        df = pd.DataFrame(reference_xds[0].to_series().rename(reference_name))

        for m in self.maps:
            soil_xds = self.maps[m].rio.reproject_match(reference_xds, resampling=Resampling.nearest)
            soil_xds = soil_xds.rio.clip([boundary], from_disk=True)

            soil_df = pd.DataFrame(soil_xds[0].to_series().rename(m)) * SOILGRIDS_PROPERTIES[m.split('@')[0]].multiplier
            df = pd.concat([df, soil_df], axis=1)

        self.matched_maps = df


    def _extract_values(self, coordinate: tuple[float, float]) -> dict[str, float]:
        transformer = Transformer.from_crs('epsg:4326', self.crs, always_xy=True)
        x, y = transformer.transform(coordinate[1], coordinate[0])

        values = {m: xds.sel(x=x, y=y, method='nearest').values[0] * SOILGRIDS_PROPERTIES[m.split('@')[0]].multiplier for m, xds in self.maps.items()}

        return values


    def get_soil_profile(self, coordinate: tuple[float, float]) -> None:
        values = self._extract_values(coordinate)

        self.soil_profile = pd.DataFrame.from_dict({
            'top': [layer.top for _, layer in SOILGRIDS_LAYERS.items()],
            'bottom': [layer.bottom for _, layer in SOILGRIDS_LAYERS.items()],
            **{v: [values[f'{v}@{key}'] for key in SOILGRIDS_LAYERS] for v in SOIL_PARAMETERS},
        })


    def generate_soil_file(self, fn: Path | str, coordinate: tuple[float, float] | None=None, *, desc: str | None=None, hsg: str='', slope: float=0.0) -> None:
        if coordinate is not None:
            self.get_soil_profile(coordinate)

        if desc is None:
            desc = f"# Soil file sampled at Latitude {coordinate[0]}, Longitude {coordinate[1]}.\n" if coordinate is not None else ""
            desc += "# NO3, NH4, and fractions of horizontal and vertical bypass flows are default empirical values.\n"
            if hsg == '':
                desc += "# Hydrologic soil group MISSING DATA.\n"
            else:
                desc += f"# Hydrologic soil group {hsg}.\n"
                desc += "# The curve number for row crops with straight row treatment is used.\n"

        assert self.soil_profile is not None
        _generate_soil_file(Path(fn), self.soil_profile, desc=desc, hsg=hsg, slope=slope)


def _read_soilgrids_maps(path: Path, maps: list[str], crs: str | None=None) -> dict[str, xarray.DataArray]:
    """Read SoilGrids data

    Parameter maps should be a list of map name strings, with each map name defined as variable@layer. For example, the
    map name for 0-5 cm bulk density should be "bulk_density@0-5cm".
    """
    soilgrids_xds = {}
    for m in maps:
        [v, layer] = m.split('@')
        soilgrids_xds[m] = rioxarray.open_rasterio(f'{path}/{SOILGRIDS_PROPERTIES[v].soilgrids_name}_{layer}.tif', masked=True)

        if crs is not None: soilgrids_xds[m] = soilgrids_xds[m].rio.reproject(crs)

    return soilgrids_xds


def _get_bounding_box(bbox: tuple[float, float, float, float], crs) -> tuple[float, float, float, float]:
    """Convert bounding boxes to SoilGrids CRS
    """
    d = {'col1': ['NW', 'SE'], 'geometry': [Point(bbox[0], bbox[3]), Point(bbox[2], bbox[1])]}
    gdf = gpd.GeoDataFrame(d, crs=crs).set_index('col1')

    converted = gdf.to_crs(HOMOLOSINE)

    return (
        converted.loc['NW', 'geometry'].xy[0][0],   # type: ignore
        converted.loc['SE', 'geometry'].xy[1][0],   # type: ignore
        converted.loc['SE', 'geometry'].xy[0][0],   # type: ignore
        converted.loc['NW', 'geometry'].xy[1][0],   # type: ignore
    )


def download_soilgrids_data(maps: dict[str, xarray.DataArray], path: str | Path, *, boundary: Polygon | None=None, bbox: tuple[float, float, float, float] | None=None, crs: str='epsg:4326') -> None:
    """Use WebCoverageService to get SoilGrids data

    bbox should be in the order of [west, south, east, north]
    Parameter maps should be a list of map name strings, with each map name defined as variable@layer. For example, the map
    name for 0-5 cm bulk density should be "bulk_density@0-5cm".
    """
    # Convert bounding box to SoilGrids CRS
    if (boundary is not None) and (bbox is None):
        bbox = boundary.bounds

        # When using just the bounding box of the state boundaries, in some cases the downloaded data do not cover the
        # entire state. Therefore a buffer zone is being used to ensure data integrity.
        buffer = [min(2.0, 0.5 * (bbox[2] - bbox[0])), min(2.0, 0.5 * (bbox[3] - bbox[1]))]

        bbox = (
            bbox[0] - buffer[0],
            bbox[1] - buffer[1],
            bbox[2] + buffer[0],
            bbox[3] + buffer[1],
        )

    assert bbox is not None
    bbox = _get_bounding_box(bbox, crs)

    for m in maps:
        [parameter, layer] = m.split('@')
        v = SOILGRIDS_PROPERTIES[parameter].soilgrids_name
        wcs = WebCoverageService(f'http://maps.isric.org/mapserv?map=/map/{v}.map', version='1.0.0')
        while True:
            try:
                response = wcs.getCoverage( # type: ignore
                    identifier=f'{v}_{layer}_mean',
                    crs='urn:ogc:def:crs:EPSG::152160',
                    bbox=bbox,
                    resx=250, resy=250,
                    format='GEOTIFF_INT16',
                )

                with open(Path(path) / f'{v}_{layer}.tif', 'wb') as file: file.write(response.read())
                break
            except:
                continue
