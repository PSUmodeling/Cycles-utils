import geopandas as gpd

GADM = lambda path, country, level: f'{path}/gadm41_{country}_{level}.shp'
GADM_LEVELS = {
    'country': 0,
    'state': 1,
    'county': 2,
}

# For USA
STATE_ABBREVIATIONS = {
    'USA.1_1': 'AL',
    'USA.2_1': 'AK',
    'USA.3_1': 'AZ',
    'USA.4_1': 'AR',
    'USA.5_1': 'CA',
    'USA.6_1': 'CO',
    'USA.7_1': 'CT',
    'USA.8_1': 'DE',
    'USA.9_1': 'DC',
    'USA.10_1': 'FL',
    'USA.11_1': 'GA',
    'USA.12_1': 'HI',
    'USA.13_1': 'ID',
    'USA.14_1': 'IL',
    'USA.15_1': 'IN',
    'USA.16_1': 'IA',
    'USA.17_1': 'KS',
    'USA.18_1': 'KY',
    'USA.19_1': 'LA',
    'USA.20_1': 'ME',
    'USA.21_1': 'MD',
    'USA.22_1': 'MA',
    'USA.23_1': 'MI',
    'USA.24_1': 'MN',
    'USA.25_1': 'MS',
    'USA.26_1': 'MO',
    'USA.27_1': 'MT',
    'USA.28_1': 'NE',
    'USA.29_1': 'NV',
    'USA.30_1': 'NH',
    'USA.31_1': 'NJ',
    'USA.32_1': 'NM',
    'USA.33_1': 'NY',
    'USA.34_1': 'NC',
    'USA.35_1': 'ND',
    'USA.36_1': 'OH',
    'USA.37_1': 'OK',
    'USA.38_1': 'OR',
    'USA.39_1': 'PA',
    'USA.40_1': 'RI',
    'USA.41_1': 'SC',
    'USA.42_1': 'SD',
    'USA.43_1': 'TN',
    'USA.44_1': 'TX',
    'USA.45_1': 'UT',
    'USA.46_1': 'VT',
    'USA.47_1': 'VA',
    'USA.48_1': 'WA',
    'USA.49_1': 'WV',
    'USA.50_1': 'WI',
    'USA.51_1': 'WY',
}


def read_gadm(path, country, level, conus=True):
    level = GADM_LEVELS[level.lower()]
    gdf = gpd.read_file(GADM(path, country, level))
    gdf.rename(columns={f'GID_{level}': 'GID'}, inplace=True)
    gdf.set_index('GID', inplace=True)


        # Generate a CONUS GeoDataFrame by removing Alaska and Hawaii
    return gdf[~gdf['NAME_1'].isin(['Alaska', 'Hawaii'])] if country == 'USA' and conus else gdf
