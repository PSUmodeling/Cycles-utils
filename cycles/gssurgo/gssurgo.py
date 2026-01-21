import geopandas as gpd
import os
import pandas as pd
import shapely
import sys
from dataclasses import dataclass
from pathlib import Path
from shapely.geometry import Point
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cycles_tools import generate_soil_file as _generate_soil_file

@dataclass
class GssurgoParameters:
    gssurgo_name: str
    multiplier: float
    table: str
    unit: str

GSSURGO = lambda path, state: path / f'gSSURGO_{state}.gdb'
GSSURGO_LUT = lambda path, lut, state: path / f'{lut}_{state}.csv'
GSSURGO_PARAMETERS = {
    'clay': GssurgoParameters('claytotal_r', 1.0, 'horizon', '%'),
    'silt': GssurgoParameters('silttotal_r', 1.0, 'horizon', '%'),
    'sand': GssurgoParameters('sandtotal_r', 1.0, 'horizon', '%'),
    'soc': GssurgoParameters('om_r', 0.58, 'horizon', '%'),
    'bulk_density': GssurgoParameters('dbthirdbar_r', 1.0, 'horizon', 'g/m3'),
    'coarse_fragments': GssurgoParameters('fragvol_r', 0.01, 'horizon', 'm3/m3'),
    'pH': GssurgoParameters('ph1to1h2o_r', 1.0, 'horizon', '-'),
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

class Gssurgo:
    def __init__(self, path: str | Path, state: str, *, lat_lon: tuple[float, float] | None=None, boundary: gpd.GeoDataFrame | None=None, lut_only: bool=False):
        self.state: str = state
        self._mapunits: gpd.GeoDataFrame | pd.DataFrame | None = None
        self.grouped_mapunits: gpd.GeoDataFrame | pd.DataFrame | None = None
        self.components: pd.DataFrame | None = None
        self.horizons: pd.DataFrame | None = None
        self.mukey: int | None = None
        self.slope: float = 0.0
        self.hsg: str = ''

        luts = _read_all_luts(Path(path), state)

        if not lut_only:
            if (lat_lon is None) and (boundary is None):
                raise ValueError("Geographic coordinate (lat_lon) or field boundary (boundary) must be provided.")

            if (lat_lon is not None) and (boundary is not None):
                raise ValueError("Geographic coordinate (lat_lon) or field boundary (boundary) are mutually exclusive. Please provide only one.")

            if (lat_lon is not None) and (boundary is None):
                boundary = gpd.GeoDataFrame({'name': ['point']}, geometry=[Point(lat_lon[1], lat_lon[0])], crs='epsg:4326')

            gdf = _read_mupolygon(Path(path), state, boundary)

        self._mapunits = gdf.merge(luts['mapunit'], on='mukey', how='left') if lut_only is False else luts['mapunit']
        self.components = luts['component']
        self.horizons = luts['horizon']

        if boundary is not None:
            self.components = self.components[self.components['mukey'].isin(self._mapunits['mukey'].unique())]
            self.horizons = self.horizons[self.horizons['cokey'].isin(self.components['cokey'].unique())]

        if not lut_only:
            self._average_slope_hsg()


    @property
    def mapunits(self) -> gpd.GeoDataFrame | pd.DataFrame | None:
        return self._mapunits

    
    def _get_muname(self, mukey: int) -> str:
        return self._mapunits[self._mapunits['mukey'] == mukey]['muname'].iloc[0]


    def group_map_units(self, *, geometry: bool=False):
        """
        Group gSSURGO map units by soil series name.

        In gSSURGO database many map units are the same soil texture with different slopes, etc. To find the dominant
        soil series, same soil texture with different slopes should be aggregated together. Therefore we use the map
        unit names to identify the same soil textures among different soil map units.
        """
        if self.grouped_mapunits is not None:
            return

        assert self._mapunits is not None

        self.grouped_mapunits = self._mapunits.copy()
        self.grouped_mapunits['muname'] = self.grouped_mapunits['muname'].map(lambda name: name.split(',')[0])
        self.grouped_mapunits['musym'] = self.grouped_mapunits['musym'].map(_musym)

        # Use the same name for all non-soil map units
        mask = self.non_soil_mask(self.grouped_mapunits)
        self.grouped_mapunits.loc[mask, 'muname'] = 'Water, urban, etc.'
        self.grouped_mapunits.loc[mask, 'mukey'] = -999
        self.grouped_mapunits.loc[mask, 'musym'] = 'N/A'

        if geometry is True:
            self.grouped_mapunits = self.grouped_mapunits.dissolve(
                by='muname',
                aggfunc={'mukey': 'first', 'musym': 'first', 'shape_area': 'sum'},
            ).reset_index() # type: ignore


    def non_soil_mask(self, mapunits: pd.DataFrame | gpd.GeoDataFrame) -> pd.Series:
        return mapunits['mukey'].isna() | mapunits['muname'].isin(GSSURGO_NON_SOIL_TYPES) | mapunits['muname'].str.contains('|'.join(GSSURGO_URBAN_TYPES), na=False)


    def select_major_mapunit(self) -> None:
        if self.mukey is not None:
            return

        if self.grouped_mapunits is None:
            self.group_map_units(geometry=True)

        assert self.grouped_mapunits is not None
        gdf = self.grouped_mapunits[~self.non_soil_mask(self.grouped_mapunits)].copy()
        gdf['area'] = gdf.area

        mapunit = gdf.loc[gdf['area'].idxmax()]

        self.mukey = int(mapunit['mukey'])  # type: ignore


    def _average_slope_hsg(self) -> None:
        assert self._mapunits is not None

        gdf = self._mapunits[~self.non_soil_mask(self._mapunits)].copy()
        gdf['area'] = gdf.area

        _df = gdf[['area', 'slopegradwta']].dropna()
        self.slope = (_df['slopegradwta'] * _df['area']).sum() / _df['area'].sum() if len(_df) > 1 else _df['slopegradwta'].iloc[0]

        _df = gdf[['area', 'hydgrpdcd']].dropna()

        if _df.empty:
            hsg = ''
        else:
            _df['hydgrpdcd'] = _df['hydgrpdcd'].map(lambda x: x[0])
            _df = _df.groupby('hydgrpdcd').sum()
            hsg = str(_df['area'].idxmax())

        self.hsg = hsg


    def get_soil_profile(self, *, mukey: int | None=None, major_only: bool=True) -> pd.DataFrame:
        if mukey is None:
            if self.mukey is None:
                self.select_major_mapunit()
            mukey = self.mukey
        assert mukey is not None

        assert self.components is not None
        df = self.components[self.components['mukey'] == int(mukey)].copy()

        if major_only is True:
            df = df[df['majcompflag'] == 'Yes']

        assert self.horizons is not None
        df = pd.merge(df, self.horizons, on='cokey')

        return df[df['hzname'] != 'R'].sort_values(by=['cokey', 'top'], ignore_index=True)


    def generate_soil_file(self, fn: Path | str, *, mukey: int | None=None, desc: str | None=None, soil_depth: float | None=None) -> None:
        if mukey is None:
            self.group_map_units(geometry=True)
            self.select_major_mapunit()
            mukey = self.mukey

        df = self.get_soil_profile(mukey=mukey)

        if desc is None:
            desc = f"# Soil file for MUNAME: {self._get_muname(mukey)}, MUKEY: {mukey}\n"
            desc += "# NO3, NH4, and fractions of horizontal and vertical bypass flows are default empirical values.\n"
            if self.hsg == '':
                desc += "# Hydrologic soil group MISSING DATA.\n"
            else:
                desc += f"# Hydrologic soil group {self.hsg}.\n"
                desc += "# The curve number for row crops with straight row treatment is used.\n"

        _generate_soil_file(fn, df, desc=desc, hsg=self.hsg, slope=self.slope, soil_depth=soil_depth)


def _read_lut(path: Path, state: str, table: str, columns: list[str]) -> pd.DataFrame:
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


def _read_all_luts(path: Path, state: str) -> dict[str, pd.DataFrame]:
    TABLES = {
        'mapunit':{
            'muaggatt': ['hydgrpdcd', 'muname', 'slopegradwta', 'mukey'],
        },
        'component':{
            'component': ['comppct_r', 'majcompflag', 'mukey', 'cokey'],
        },
        'horizon': {
            'chorizon': ['hzname', 'hzdept_r', 'hzdepb_r', 'sandtotal_r', 'silttotal_r', 'claytotal_r', 'om_r', 'dbthirdbar_r', 'ph1to1h2o_r', 'cokey', 'chkey'],
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


def _read_mupolygon(path: Path, state: str, boundary=None) -> gpd.GeoDataFrame:
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


def _musym(s: str):
    if s == 'N/A' or len(s) < 2:
        return s

    if s[-1].isupper() and (s[-2].isnumeric() or s[-2].islower()):
        return s[:-1]

    if s[-1].isnumeric() and s[-2].isupper() and (s[-3].isnumeric() or s[-3].islower()):
        return s[:-2]

    return s
