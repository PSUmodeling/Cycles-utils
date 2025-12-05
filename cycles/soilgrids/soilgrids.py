import geopandas as gpd
import pandas as pd
import rioxarray
import xarray
from dataclasses import dataclass
from owslib.wcs import WebCoverageService
from pyproj import Transformer
from rasterio.enums import Resampling
from shapely.geometry import Point

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
    'coarse_fragments': SoilGridsProperties('cfvo', ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm'], 0.1, '%'),
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


def read_soilgrids_maps(path: str, maps: list[str], *, crs=None) -> dict[str, xarray.DataArray]:
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


def extract_values(soilgrids_xds: dict[str, xarray.DataArray], coordinate: tuple[float, float]) -> dict[str, float]:
    transformer = Transformer.from_crs('epsg:4326', HOMOLOSINE, always_xy=True)
    x, y = transformer.transform(coordinate[1], coordinate[0])

    values = {}

    values = {m: xds.sel(x=x, y=y, method='nearest').values[0] * SOILGRIDS_PROPERTIES[m.split('@')[0]].multiplier for m, xds in soilgrids_xds.items()}

    return values


def reproject_match_soilgrids_maps(soilgrids_xds: dict[str, xarray.DataArray], reference_xds: xarray.DataArray, reference_name: str, boundary: gpd.GeoDataFrame) -> pd.DataFrame:
    reference_xds = reference_xds.rio.clip([boundary], from_disk=True)
    df = pd.DataFrame(reference_xds[0].to_series().rename(reference_name))

    for m in soilgrids_xds:
        soil_xds = soilgrids_xds[m].rio.reproject_match(reference_xds, resampling=Resampling.nearest)
        soil_xds = soil_xds.rio.clip([boundary], from_disk=True)

        soil_df = pd.DataFrame(soil_xds[0].to_series().rename(m)) * SOILGRIDS_PROPERTIES[m.split('@')[0]].multiplier
        df = pd.concat([df, soil_df], axis=1)

    return df


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


def download_soilgrids_data(maps: dict[str, xarray.DataArray], path: str, bbox: tuple[float, float, float, float], *, crs='epsg:4326') -> None:
    """Use WebCoverageService to get SoilGrids data

    bbox should be in the order of [west, south, east, north]
    Parameter maps should be a list of map name strings, with each map name defined as variable@layer. For example, the map
    name for 0-5 cm bulk density should be "bulk_density@0-5cm".
    """
    # Convert bounding box to SoilGrids CRS
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

                with open(f'{path}/{v}_{layer}.tif', 'wb') as file: file.write(response.read())
                break
            except:
                continue


def get_soil_profile_parameters(maps: dict[str, xarray.DataArray], coordinate: tuple[float, float]):
        values = extract_values(maps, coordinate)

        return pd.DataFrame.from_dict([{
                'top': layer.top,
                'bottom': layer.bottom,
                **{v: values[f'{v}@{key}'] for v in ['clay', 'sand', 'soc', 'bulk_density']},
            } for key, layer in SOILGRIDS_LAYERS.items()]   # type: ignore
        )
