import geopandas as gpd
import pandas as pd
import shapely
from dataclasses import dataclass

@dataclass
class GssurgoParameters:
    gssurgo_name: str
    multiplier: float
    table: str
    unit: str

GSSURGO = lambda path, state: f'{path}/gSSURGO_{state}.gdb'
GSSURGO_LUT = lambda path, lut, state: f'{path}/{lut}_{state}.csv'
GSSURGO_PARAMETERS = {
    'clay': GssurgoParameters('claytotal_r', 1.0, 'horizon', '%'),
    'silt': GssurgoParameters('silttotal_r', 1.0, 'horizon', '%'),
    'sand': GssurgoParameters('sandtotal_r', 1.0, 'horizon', '%'),
    'soc': GssurgoParameters('om_r', 0.58, 'horizon', '%'),
    'bulk_density': GssurgoParameters('dbthirdbar_r', 1.0, 'horizon', 'g/m3'),
    'coarse_fragments': GssurgoParameters('fragvol_r', 1.0, 'horizon', '%'),
    'area_fraction': GssurgoParameters('comppct_r', 1.0, 'component', '%'),
    'top': GssurgoParameters('hzdept_r', 0.01, 'horizon', 'm'),
    'bottom': GssurgoParameters('hzdepb_r', 0.01, 'horizon', 'm'),
}

GSSURGO_NON_SOIL_TYPES = (
    'Acidic rock land',
    'Area not surveyed',
    'Dam',
    'Dumps',
    'Levee',
    'No Digital Data Available',
    'Pits',
    'Water',
)
GSSURGO_URBAN_TYPES = (
    'Udorthents',
    'Urban land',
)
NAD83 = 'epsg:5070'     # NAD83 / Conus Albers, CRS of gSSURGO

def _read_lut(path: str, state: str, table: str, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(
        GSSURGO_LUT(path, table, state),
        usecols=columns,
    )

    if table == 'chfrags':
        df = df.groupby('chkey').sum().reset_index()

    df.rename(
        columns={value.gssurgo_name: key for key, value in GSSURGO_PARAMETERS.items()},
        inplace=True,
    )

    for key, value in GSSURGO_PARAMETERS.items():
        if key in df.columns:
            df[key] *= value.multiplier

    return df


def _read_all_luts(path: str, state: str) -> dict[str, pd.DataFrame]:
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

    lookup_tables = {}
    for key in TABLES:
        lookup_tables[key] = pd.DataFrame()

        for table, columns in TABLES[key].items():
            if lookup_tables[key].empty:
                lookup_tables[key] = _read_lut(path, state, table, columns)
            else:
                lookup_tables[key] = lookup_tables[key].merge(_read_lut(path, state, table, columns), how='outer')

    return lookup_tables


def _read_mupolygon(path, state, boundary=None) -> gpd.GeoDataFrame:
    if boundary is not None:
        boundary = boundary.to_crs(NAD83)

    gdf: gpd.GeoDataFrame = gpd.read_file(
            GSSURGO(path, state),
            layer='MUPOLYGON',
            mask=shapely.union_all(boundary['geometry'].values) if boundary is not None else None
        )

    if boundary is not None: gdf = gpd.clip(gdf, boundary, keep_geom_type=False)

    gdf.columns = [x.lower() for x in gdf.columns]
    gdf['mukey'] = gdf['mukey'].astype(int)

    return gdf


def _musym(str):
    if str == 'N/A' or len(str) < 2:
        return str

    if str[-1].isupper() and (str[-2].isnumeric() or str[-2].islower()):
        return str[:-1]

    if str[-1].isnumeric() and str[-2].isupper() and (str[-3].isnumeric() or str[-3].islower()):
        return str[:-2]

    return str


class Gssurgo:
    def __init__(self, *, path: str, state: str, boundary: gpd.GeoDataFrame | None =None, lut_only: bool=False):
        self.state = state

        luts = _read_all_luts(path, state)

        if lut_only is False:
            gdf = _read_mupolygon(path, state, boundary)
            self.mapunits = gdf.merge(luts['mapunit'], on='mukey', how='left')
        else:
            self.mapunits = luts['mapunit']

        self.components = luts['component']
        self.horizons = luts['horizon']

        if boundary is not None:
            self.components = self.components[self.components['mukey'].isin(self.mapunits['mukey'].unique())]
            self.horizons = self.horizons[self.horizons['cokey'].isin(self.components['cokey'].unique())]


    def group_map_units(self, *, geometry: bool=False):
        # In gSSURGO database many map units are the same soil texture with different slopes, etc. To find the dominant
        # soil series, same soil texture with different slopes should be aggregated together. Therefore we use the map
        # unit names to identify the same soil textures among different soil map units.
        self.mapunits['muname'] = self.mapunits['muname'].map(lambda name: name.split(',')[0])
        self.mapunits['musym'] = self.mapunits['musym'].map(_musym)

        # Use the same name for all non-soil map units
        mask = self.non_soil_mask()
        self.mapunits.loc[mask, 'muname'] = 'Water, urban, etc.'
        self.mapunits.loc[mask, 'mukey'] = -999
        self.mapunits.loc[mask, 'musym'] = 'N/A'

        if geometry is True:
            self.mapunits = self.mapunits.dissolve(
                by='muname',
                aggfunc={'mukey': 'first', 'musym': 'first', 'shape_area': sum},
            ).reset_index()


    def non_soil_mask(self) -> pd.Series:
        return self.mapunits['mukey'].isna() | \
            self.mapunits['muname'].isin(GSSURGO_NON_SOIL_TYPES) | \
            self.mapunits['muname'].str.contains('|'.join(GSSURGO_URBAN_TYPES), na=False)


    def select_major_mapunit(self) -> tuple[int, str, str]:
        gdf = self.mapunits[~self.non_soil_mask()].copy()
        gdf['area'] = gdf.area

        mapunit = gdf.loc[gdf['area'].idxmax()]

        return int(mapunit['mukey']), mapunit['musym'], mapunit['muname']


    def average_slope_hsg(self) -> tuple[float, str]:
        gdf = self.mapunits[~self.non_soil_mask()].copy()
        gdf['area'] = gdf.area

        _df = gdf[['area', 'slopegradwta']].dropna()
        slope = (_df['slopegradwta'] * _df['area']).sum() / _df['area'].sum()

        _df = gdf[['area', 'hydgrpdcd']].dropna()

        if _df.empty:
            hsg = ''
        else:
            _df['hydgrpdcd'] = _df['hydgrpdcd'].map(lambda x: x[0])
            _df = _df.groupby('hydgrpdcd').sum()
            hsg = _df['area'].idxmax()

        return slope, hsg


    def get_soil_profile_parameters(self, mukey: int, *, major_only: bool=True) -> pd.DataFrame:
        df = self.components[self.components['mukey'] == int(mukey)].copy()

        if major_only is True:
            df = df[df['majcompflag'] == 'Yes']

        df = pd.merge(df, self.horizons, on='cokey')

        return df[df['hzname'] != 'R'].sort_values(by=['cokey', 'top'], ignore_index=True)


    def __str__(self):
        return f'gSSURGO data for {self.state}'
