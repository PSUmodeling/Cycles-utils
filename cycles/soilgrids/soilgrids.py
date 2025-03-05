import geopandas as gpd
import pandas as pd
import rioxarray
from owslib.wcs import WebCoverageService
from rasterio.enums import Resampling
from shapely.geometry import Point

SOILGRIDS_PROPERTIES = {
    'clay': {'name': 'clay', 'multiplier': 0.1},    # %
    'sand': {'name': 'sand', 'multiplier': 0.1},    # %
    'soc': {'name': 'soc', 'multiplier': 0.01},     # %
    'bulk_density': {'name': 'bdod', 'multiplier': 0.01},   # Mg/m3
}
SOILGRIDS_LAYERS = {
    # units: m
    '0-5cm': {'top': 0, 'bottom': 0.05, 'thickness': 0.05},
    '5-15cm': {'top': 0.05, 'bottom': 0.15, 'thickness': 0.10},
    '15-30cm': {'top': 0.15, 'bottom': 0.3, 'thickness': 0.15},
    '30-60cm': {'top': 0.3, 'bottom': 0.6, 'thickness': 0.3},
    '60-100cm': {'top': 0.6, 'bottom': 1.0, 'thickness': 0.4},
    '100-200cm': {'top': 1.0, 'bottom': 2.0, 'thickness': 1.0},
}
HOMOLOSINE = 'urn:ogc:def:crs:EPSG::152160' # Interrupted Goode Homolosine, CRS of SoilGrids


# Read state SoilGrids data
def read_soilgrids_maps(path, state_id, layers, parameters, crs=HOMOLOSINE):
    soilgrids_xds = {}
    for layer in layers:
        for v in parameters:
            soilgrids_xds[f'{v}_{layer}'] = rioxarray.open_rasterio(f'{path}/{state_id}/{SOILGRIDS_PROPERTIES[v]["name"]}_{layer}.tif', masked=True).rio.reproject(crs)

    return soilgrids_xds


def reproject_match_soilgrids_maps(soilgrids_xds, reference_xds, reference_name, boundary, layers, parameters):
    reference_xds = reference_xds.rio.clip([boundary], from_disk=True)
    df = pd.DataFrame(reference_xds[0].to_series().rename(reference_name))

    for v in parameters:
        for layer in layers:
            soil_xds = soilgrids_xds[f'{v}_{layer}'].rio.reproject_match(reference_xds, resampling=Resampling.nearest)
            soil_xds = soil_xds.rio.clip([boundary], from_disk=True)

            soil_df = pd.DataFrame(soil_xds[0].to_series().rename(f'{v}_{layer}')) * SOILGRIDS_PROPERTIES[v]['multiplier']
            df = pd.concat([df, soil_df], axis=1)

    return df


"""Convert bounding boxes to SoilGrids CRS
"""
def get_bounding_box(reference_shp, bbox, crs):
    d = {'col1': ['NW', 'SE'], 'geometry': [Point(bbox[0], bbox[3]), Point(bbox[2], bbox[1])]}
    gdf = gpd.GeoDataFrame(d, crs=crs).set_index('col1')

    converted = gdf.to_crs(gpd.read_file(reference_shp).crs)

    return [
        converted.loc['NW', 'geometry'].xy[0][0],
        converted.loc['SE', 'geometry'].xy[1][0],
        converted.loc['SE', 'geometry'].xy[0][0],
        converted.loc['NW', 'geometry'].xy[1][0],
    ]


"""Use WebCoverageService to get SoilGrids data
bbox should be in the order of [west, south, east, north]
"""
def download_soilgrids_data(layers, parameters, path, bbox, crs):
    # Convert bounding box to SoilGrids CRS
    bbox = get_bounding_box(bbox, crs)

    for v in [SOILGRIDS_PROPERTIES[param]['name'] for param in parameters]:
        for layer in layers:
            wcs = WebCoverageService(f'http://maps.isric.org/mapserv?map=/map/{v}.map', version='1.0.0')
            while True:
                try:
                    response = wcs.getCoverage(
                        identifier=f'{v}_{layer}_mean',
                        crs=HOMOLOSINE,
                        bbox=bbox,
                        resx=250, resy=250,
                        format='GEOTIFF_INT16')

                    with open(f'{path}/{v}_{layer}.tif', 'wb') as file: file.write(response.read())
                    break
                except:
                    continue
