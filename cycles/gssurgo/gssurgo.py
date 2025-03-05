import pandas as pd

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

def read_state_gssurgo_luts(path, state_abbreviation, group=False):
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
