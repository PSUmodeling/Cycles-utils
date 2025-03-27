import geopandas as gpd
import pandas as pd
import shapely

GSSURGO = lambda path, state: f'{path}/gSSURGO_{state}.gdb/'
GSSURGO_LUT = lambda path, lut, state: f'{path}/{lut}_{state}.csv'
GSSURGO_PARAMETERS = {
    'clay': {'variable': 'claytotal_r', 'multiplier': 1.0}, # %
    'sand': {'variable': 'sandtotal_r', 'multiplier': 1.0}, # %
    'soc': {'variable': 'om_r', 'multiplier': 0.58},    # %
    'bulk_density': {'variable': 'dbthirdbar_r', 'multiplier': 1.0},    # Mg/m3
    'top': {'variable': 'hzdept_r', 'multiplier': 0.01},    # m
    'bottom': {'variable': 'hzdepb_r', 'multiplier': 0.01}, # m
}
GSSURGO_NON_SOIL_TYPES = [
    'Acidic rock land',
    'Area not surveyed',
    'Dam',
    'Dumps',
    'Levee',
    'No Digital Data Available',
    'Pits',
    'Water',
]
GSSURGO_URBAN_TYPES = [
    'Udorthents',
    'Urban land',
]
NAD83 = 'epsg:5070'     # NAD83 / Conus Albers, CRS of gSSURGO


def read_state_luts(path, state_abbreviation, group=False):
    tables = {
        'component': ['mukey', 'cokey', 'majcompflag'],
        'chorizon': ['hzname', 'hzdept_r', 'hzdepb_r', 'sandtotal_r', 'silttotal_r', 'claytotal_r', 'om_r', 'dbthirdbar_r', 'cokey'],
        'muaggatt': ['hydgrpdcd', 'muname', 'slopegradwta', 'mukey'],
    }

    gssurgo_luts = {}
    for t in tables:
        gssurgo_luts[t] = pd.read_csv(
            GSSURGO_LUT(path, t, state_abbreviation),
            usecols=tables[t],
        )

    # Rename table columns
    gssurgo_luts['chorizon'] = gssurgo_luts['chorizon'].rename(
        columns={GSSURGO_PARAMETERS[v]['variable']: v for v in GSSURGO_PARAMETERS}
    )
    # Convert units (note that organic matter is also converted to soil organic carbon in this case)
    for v in GSSURGO_PARAMETERS: gssurgo_luts['chorizon'][v] *= GSSURGO_PARAMETERS[v]['multiplier']

    # In the gSSURGO database many map units are the same soil texture with different slopes, etc. To find the dominant
    # soil series, same soil texture with different slopes should be aggregated together. Therefore we use the map unit
    # names to identify the same soil textures among different soil map units.
    if group:
        gssurgo_luts['muaggatt']['muname'] = gssurgo_luts['muaggatt']['muname'].map(lambda name: name.split(',')[0])

    return gssurgo_luts


def read_state_gssurgo(path, state_abbreviation, boundary=None, group=False):
    gdf = gpd.read_file(
            GSSURGO(path, state_abbreviation),
            layer='MUPOLYGON',
            mask=shapely.union_all(boundary['geometry'].values) if boundary is not None else None
        )
    if boundary is not None: gdf = gpd.clip(gdf, boundary, keep_geom_type=False)
    gdf.columns = [x.lower() for x in gdf.columns]
    gdf.mukey = gdf.mukey.astype(int)

    luts = read_state_luts(path, state_abbreviation, group=group)

    # Merge the mapunit polygon table with the mapunit aggregated attribute table
    gdf = gdf.merge(luts['muaggatt'], on='mukey')

    return gdf, luts


def get_soil_profile_parameters(luts, mukey):
    df = luts['component'][luts['component'].mukey == int(mukey)].merge(luts['chorizon'], on='cokey')
    if not df[df['majcompflag'] == 'Yes'].empty:
        df = df[df['majcompflag'] == 'Yes'].sort_values(by='top')
    else:
        print(f'{index} {t} no major component')
        df = df.sort_values(by='top')

    return df[df['hzname'] != 'R']
