import geopandas as gpd
import pandas as pd
import shapely

GSSURGO = lambda path, state: f'{path}/gSSURGO_{state}.gdb'
GSSURGO_LUT = lambda path, lut, state: f'{path}/{lut}_{state}.csv'
GSSURGO_PARAMETERS = {
    'clay': {'variable': 'claytotal_r', 'multiplier': 1.0, 'table': 'horizon'}, # %
    'silt': {'variable': 'silttotal_r', 'multiplier': 1.0, 'table': 'horizon'}, # %
    'sand': {'variable': 'sandtotal_r', 'multiplier': 1.0, 'table': 'horizon'}, # %
    'soc': {'variable': 'om_r', 'multiplier': 0.58, 'table': 'horizon'},    # %
    'bulk_density': {'variable': 'dbthirdbar_r', 'multiplier': 1.0, 'table': 'horizon'},    # Mg/m3
    'coarse_fragments': {'variable': 'fragvol_r', 'multiplier': 1.0, 'table': 'horizon'},   # %
    'area_fraction': {'variable': 'comppct_r', 'multiplier': 1.0, 'table': 'component'},    # %
    'top': {'variable': 'hzdept_r', 'multiplier': 0.01, 'table': 'horizon'},    # m
    'bottom': {'variable': 'hzdepb_r', 'multiplier': 0.01, 'table': 'horizon'}, # m
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
    TABLES = {
        'mapunit':{
            'muaggatt': ['hydgrpdcd', 'muname', 'slopegradwta', 'mukey'],
        },
        'component':{
            'component': ['comppct_r', 'majcompflag', 'mukey', 'cokey'],
        },
        'horizon': {
            'chorizon': ['hzname', 'hzdept_r', 'hzdepb_r', 'sandtotal_r', 'silttotal_r', 'claytotal_r', 'om_r', 'dbthirdbar_r', 'cokey', 'chkey'],
            'chfrags': ['fragvol_r', 'chkey'],
        },
    }

    gssurgo_luts = {}
    for t in TABLES:
        gssurgo_luts[t] = pd.DataFrame()
        for tt in TABLES[t]:
            _df = pd.read_csv(
                GSSURGO_LUT(path, tt, state_abbreviation),
                usecols=TABLES[t][tt],
            )

            if tt == 'chfrags': _df = _df.groupby('chkey').sum().reset_index()
            try:
                gssurgo_luts[t] = pd.merge(gssurgo_luts[t], _df, how='outer')
            except:
                gssurgo_luts[t] = _df.copy()

        # Rename table columns
        gssurgo_luts[t] = gssurgo_luts[t].rename(
            columns={GSSURGO_PARAMETERS[v]['variable']: v for v in GSSURGO_PARAMETERS}
        )

    # Convert units (note that organic matter is also converted to soil organic carbon in this case)
    for v in GSSURGO_PARAMETERS:
        gssurgo_luts[GSSURGO_PARAMETERS[v]['table']][v] *= GSSURGO_PARAMETERS[v]['multiplier']

    # In the gSSURGO database many map units are the same soil texture with different slopes, etc. To find the dominant
    # soil series, same soil texture with different slopes should be aggregated together. Therefore we use the map unit
    # names to identify the same soil textures among different soil map units.
    if group:
        gssurgo_luts['mapunit']['muname'] = gssurgo_luts['mapunit']['muname'].map(lambda name: name.split(',')[0])

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
    gdf = gdf.merge(luts['mapunit'], on='mukey')

    return gdf, luts


def get_soil_profile_parameters(luts, mukey, major_only=True):
    df = luts['component'][luts['component']['mukey'] == int(mukey)].copy()

    if major_only is True: df = df[df['majcompflag'] == 'Yes']

    df = pd.merge(df, luts['horizon'], on='cokey')

    return df[df['hzname'] != 'R'].sort_values(by=['cokey', 'top'], ignore_index=True)


def musym(str):
    if str == 'N/A' or len(str) < 2:
        return str

    if str[-1].isupper() and (str[-2].isnumeric() or str[-2].islower()):
        return str[:-1]

    if str[-1].isnumeric() and str[-2].isupper() and (str[-3].isnumeric() or str[-3].islower()):
        return str[:-2]

    return str


def non_soil_mask(df):
    return df['mukey'].isna() | df['muname'].isin(GSSURGO_NON_SOIL_TYPES) | df['muname'].str.contains('|'.join(GSSURGO_URBAN_TYPES), na=False)


def group_map_units(soil_df):
    # Combine the soil map units that have the same names
    df = soil_df.dissolve(by='muname', aggfunc={'mukey': 'first', 'musym': 'first', 'area': sum, 'shape_area': sum}).reset_index()

    # Use the same name for all non-soil map units
    mask = non_soil_mask(df)
    df.loc[mask, 'muname'] = 'Water, urban, etc.'
    df.loc[mask, 'mukey'] = None
    df.loc[mask, 'musym'] = 'N/A'

    # Combine non-soil map units
    df = df.dissolve(by='muname', aggfunc={'mukey': 'first', 'musym': 'first', 'area': sum, 'shape_area': sum}).reset_index()

    df['musym'] = df['musym'].map(musym)

    return df
