import geopandas as gpd
import math
import numpy as np
import os
import pandas as pd
from netCDF4 import Dataset

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
    'elevation': {
        'GLDAS': 'GLDAS_elevation',
        'gridMET': 'elevation',
        'NLDAS': 'NLDAS_elev',
    },
    'mask': {
        'GLDAS': 'GLDAS_mask',
        'gridMET': 'elevation', # For gridMET, mask and elevation are the same file
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
IND_J = lambda reanalysis, lat: int(round((lat - LA1[reanalysis]) / DJ[reanalysis]))
IND_I = lambda reanalysis, lon: int(round((lon - LO1[reanalysis]) / DI[reanalysis]))
SHAPES = {
    'GLDAS': (600, 1440),
    'gridMET': (585, 1386),
    'NLDAS': (224, 464),
}


def read_land_mask(reanalysis):
    with Dataset(LAND_MASK_FILES[reanalysis]) as nc:
        mask =  nc[VARIABLES['mask'][reanalysis]][:, :] if reanalysis == 'gridMET' else nc[VARIABLES['mask'][reanalysis]][0]
        lats, lons = np.meshgrid(nc['lat'][:], nc['lon'][:], indexing='ij')

    with Dataset(ELEVATION_FILES[reanalysis]) as nc:
        elevations = nc[VARIABLES['elevation'][reanalysis]][:, :] if reanalysis == 'gridMET' else nc[VARIABLES['elevation'][reanalysis]][0][:, :]

    df = pd.DataFrame({
        'latitude': lats.flatten(),
        'longitude': lons.flatten(),
        'mask': mask.flatten(),
        'elevation': elevations.flatten(),
    })

    if reanalysis == 'gridMET':
        df.loc[~df['mask'].isna(), 'mask'] = 1
        df.loc[df['mask'].isna(), 'mask'] = 0

    df['mask'] = df['mask'].astype(int)

    return df


def find_grids(reanalysis, locations=None, model=None, rcp=None):
    mask_df = read_land_mask(reanalysis)

    if locations is None:
        indices = [ind for ind, row in mask_df.iterrows() if row['mask'] > 0]
    else:
        indices = []

        for (lat, lon) in locations:
            ind = np.ravel_multi_index((IND_J(reanalysis, lat), IND_I(reanalysis, lon)), SHAPES[reanalysis])

            if mask_df.loc[ind]['mask'] == 0:
                mask_df['distance'] = mask_df.apply(
                    lambda x: math.sqrt((x['latitude'] - lat) ** 2 + (x['longitude'] - lon) ** 2),
                    axis=1,
                )
                mask_df.loc[mask_df['mask'] == 0, 'distance'] = 1E6
                ind = mask_df['distance'].idxmin()

            indices.append(ind)

    grids = []
    for ind in indices:
        grid_lat, grid_lon = mask_df.loc[ind, ['latitude', 'longitude']]

        grid_str = '%.3f%sx%.3f%s' % (
            abs(grid_lat), 'S' if grid_lat < 0.0 else 'N', abs(grid_lon), 'W' if grid_lon < 0.0 else 'E'
        )

        if reanalysis == 'MACA':
            fn = f'macav2metdata_{model}_rcp{rcp}_{grid_str}.weather'
        else:
            fn = f'{reanalysis}_{grid_str}.weather'

        grids.append({
            'grid_index': ind,
            'weather_file': fn,
            'elevation': mask_df.loc[ind, 'elevation'],
        })

    return pd.DataFrame(grids)
