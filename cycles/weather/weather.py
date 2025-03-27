import geopandas as gpd
import math
import numpy as np
import os
import pandas as pd
import xarray
from shapely.geometry import Point

pt = os.path.dirname(os.path.realpath(__file__))
LAND_MASK_FILES = {
    'GLDAS': os.path.join(pt, '../data/GLDASp5_landmask_025d.nc4'),
    'gridMET': os.path.join(pt, '../data/gridMET_elevation_mask.nc'),
    'NLDAS': os.path.join(pt, '../data/NLDAS_masks-veg-soil.nc4'),
}
ELEVATION_FILES = {
    'GLDAS': os.path.join(pt, '../data/GLDASp5_elevation_025d.nc4'),
    'gridMET': os.path.join(pt, '../data/gridMET_elevation_mask.nc'),
    'NLDAS': os.path.join(pt, '../data/NLDAS_elevation.nc4'),
}
VARIABLES = {
    'ELEVATION': {
        'gridMET': 'elevation',
        'GLDAS': 'GLDAS_elevation',
        'NLDAS': 'NLDAS_elev',
    },
    'MASK': {
        'gridMET': 'elevation', # For gridMET, mask and elevation are the same file
        'GLDAS': 'GLDAS_mask',
        'NLDAS': 'CONUS_mask',
    },
}

LA1 = {
    'GLDAS': -59.875,
    'gridMET': 49.4,
    'NLDAS': 25.0625,
}
LO1 = {
    'GLDAS': -179.875,
    'gridMET': -124.76667,
    'NLDAS': -124.9375,
}
DI = {
    'GLDAS': 0.25,
    'gridMET': 1.0 / 24.0,
    'NLDAS': 0.125,
}
DJ = {
    'GLDAS': 0.25,
    'gridMET': -1.0 / 24.0,
    'NLDAS': 0.125,
}
IND_J = lambda ldas, lat: int(round((lat - LA1[ldas]) / DJ[ldas]))
IND_I = lambda ldas, lon: int(round((lon - LO1[ldas]) / DI[ldas]))
SHAPES = {
    'GLDAS': (600, 1440),
    'gridMET': (585, 1386),
    'NLDAS': (224, 464),
}


def read_land_mask(ldas):
    ds = xarray.open_dataset(LAND_MASK_FILES[ldas])

    lats, lons = np.meshgrid(ds['lat'].values, ds['lon'].values, indexing='ij')

    df = pd.DataFrame({
        'latitude': lats.flatten(),
        'longitude': lons.flatten(),
        'mask': ds[VARIABLES['MASK'][ldas]].values.flatten(),
    })

    if ldas == 'gridMET':
        df.loc[~df['mask'].isna(), 'mask'] = 1
        df.loc[df['mask'].isna(), 'mask'] = 0

    df['mask'] = df['mask'].astype(int)

    return df


def find_grids(ldas, locations):
    df = read_land_mask(ldas)

    grids = []
    for (lat, lon) in locations:
        grid_ind = np.ravel_multi_index((IND_J(ldas, lat), IND_I(ldas, lon)), SHAPES[ldas])

        if df.loc[grid_ind]['mask'] == 0:
            df['distance'] = df.apply(
                lambda x: math.sqrt((x['latitude'] - lat) ** 2 + (x['longitude'] - lon) ** 2),
                axis=1,
            )
            df.loc[df['mask'] == 0, 'distance'] = 1E6
            grid_ind = df['distance'].idxmin()

        grids.append(grid_ind)

    return grids
