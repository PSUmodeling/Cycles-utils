from __future__ import annotations

import geopandas as gpd
import pandas as pd
import shapely
from dataclasses import dataclass
from pathlib import Path
from shapely.geometry import Point

from cycles.cycles_tools import generate_soil_file as _generate_soil_file
from cycles.cycles_tools import SoilLayer, MAPPABLE_PARAMETERS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAD83: str = 'epsg:5070'    # NAD83 / Conus Albers, CRS of SSURGO

SSURGO_NON_SOIL_TYPES: frozenset[str] = frozenset({
    'Acidic rock land',
    'Area not surveyed',
    'Dam',
    'Dumps',
    'Levee',
    'No Digital Data Available',
    'Pits',
    'Water',
})

SSURGO_URBAN_TYPES: frozenset[str] = frozenset({
    'Udorthents',
    'Urban land',
})

# Lookup table structure: lut_key → {csv_table → [columns]}
LUT_TABLES: dict[str, dict[str, list[str]]] = {
    'mapunit': {
        'muaggatt': ['hydgrpdcd', 'muname', 'slopegradwta', 'mukey'],
    },
    'component': {
        'component': ['comppct_r', 'majcompflag', 'mukey', 'cokey'],
    },
    'horizon': {
        'chorizon': [
            'hzname', 'hzdept_r', 'hzdepb_r', 'sandtotal_r', 'silttotal_r',
            'claytotal_r', 'om_r', 'dbthirdbar_r', 'ph1to1h2o_r', 'cokey', 'chkey',
        ],
        'chfrags': ['fragvol_r', 'chkey'],
    },
}


# ---------------------------------------------------------------------------
# SSURGO parameter spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SsurgoParameter:
    ssurgo_name: str
    multiplier:  float
    table:       str
    unit:        str


SSURGO_PARAMETERS: dict[str, SsurgoParameter] = {
    'clay': SsurgoParameter('claytotal_r', 1.0, 'horizon', '%'),
    'silt': SsurgoParameter('silttotal_r', 1.0, 'horizon', '%'),
    'sand': SsurgoParameter('sandtotal_r', 1.0, 'horizon', '%'),
    'soc': SsurgoParameter('om_r', 0.58, 'horizon', '%'),
    'bulk_density': SsurgoParameter('dbthirdbar_r', 1.0, 'horizon', 'g/m3'),
    'coarse_fragments': SsurgoParameter('fragvol_r', 0.01, 'horizon', 'm3/m3'),
    'pH': SsurgoParameter('ph1to1h2o_r', 1.0, 'horizon', '-'),
    'area_fraction': SsurgoParameter('comppct_r', 1.0, 'component', '%'),
    'top': SsurgoParameter('hzdept_r', 0.01, 'horizon', 'm'),
    'bottom': SsurgoParameter('hzdepb_r', 0.01, 'horizon', 'm'),
}

LatLon = tuple[float, float]

# ---------------------------------------------------------------------------
# Ssurgo
# ---------------------------------------------------------------------------

class Ssurgo:
    def __init__(self, path: str | Path, state: str, *, lat_lon: LatLon | None=None, boundary: gpd.GeoDataFrame | None=None, lut_only: bool=False) -> None:
        _validate_geographic_input(lat_lon, boundary, lut_only)

        self.state: str = state
        self._mapunits: gpd.GeoDataFrame | pd.DataFrame | None = None
        self.grouped_mapunits: gpd.GeoDataFrame | pd.DataFrame | None = None
        self.components: pd.DataFrame | None = None
        self.horizons: pd.DataFrame | None = None
        self.mukey: int | None = None
        self.slope: float = 0.0
        self.hsg: str = ''

        path = Path(path)
        luts = _read_all_luts(path, state)

        if not lut_only:
            if lat_lon is not None:
                boundary = gpd.GeoDataFrame(
                    {'name': ['point']},
                    geometry=[Point(lat_lon[1], lat_lon[0])],
                    crs='epsg:4326',
                )
            gdf = _read_mupolygon(path, state, boundary)
            self._mapunits = gdf.merge(luts['mapunit'], on='mukey', how='left')
        else:
            self._mapunits = luts['mapunit']

        self.components = luts['component']
        self.horizons   = luts['horizon']

        if boundary is not None:
            self.components = self.components[
                self.components['mukey'].isin(self._mapunits['mukey'].unique())
            ]
            self.horizons = self.horizons[
                self.horizons['cokey'].isin(self.components['cokey'].unique())
            ]

        if not lut_only:
            self._average_slope_hsg()

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def mapunits(self) -> gpd.GeoDataFrame | pd.DataFrame | None:
        return self._mapunits

    @property
    def muname(self) -> str:
        self._ensure_mukey()
        return self._get_muname(self.mukey)

    @property
    def musym(self) -> str:
        self._ensure_mukey()
        assert self._mapunits is not None
        return self._mapunits[self._mapunits['mukey'] == self.mukey]['musym'].iloc[0]

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def group_map_units(self, *, geometry: bool = False) -> None:
        """Group SSURGO map units by soil series name.

        Many map units share the same soil texture but differ in slope or other
        attributes. Grouping by the base name (before the first comma) aggregates
        these into a single representative series.
        """
        if self.grouped_mapunits is not None:
            return
        assert self._mapunits is not None

        gmu = self._mapunits.copy()
        gmu['muname'] = gmu['muname'].map(lambda name: name.split(',')[0])
        gmu['musym']  = gmu['musym'].map(_strip_slope_suffix)

        mask = self.non_soil_mask(gmu)
        gmu.loc[mask, 'muname'] = 'Water, urban, etc.'
        gmu.loc[mask, 'mukey']  = -999
        gmu.loc[mask, 'musym']  = 'N/A'

        if geometry:
            gmu = gmu.dissolve(
                by='muname',
                aggfunc={'mukey': 'first', 'musym': 'first', 'shape_area': 'sum'},
            ).reset_index()

        self.grouped_mapunits = gmu

    def non_soil_mask(self, mapunits: pd.DataFrame | gpd.GeoDataFrame) -> pd.Series:
        return (
            mapunits['mukey'].isna()
            | mapunits['muname'].isin(SSURGO_NON_SOIL_TYPES)
            | mapunits['muname'].str.contains('|'.join(SSURGO_URBAN_TYPES), na=False)
        )

    def select_major_mapunit(self) -> None:
        if self.mukey is not None:
            return
        if self.grouped_mapunits is None:
            self.group_map_units(geometry=True)
        assert self.grouped_mapunits is not None

        gdf = self.grouped_mapunits[~self.non_soil_mask(self.grouped_mapunits)].copy()
        gdf['area'] = gdf.area
        self.mukey  = int(gdf.loc[gdf['area'].idxmax(), 'mukey'])

    def get_soil_profile(self, *, mukey: int | None=None, major_only: bool=True) -> list[SoilLayer]:
        mukey = mukey or self._ensure_mukey()
        assert self.components is not None and self.horizons is not None

        df = self.components[self.components['mukey'] == int(mukey)].copy()
        if major_only:
            df = df[df['majcompflag'] == 'Yes']

        df = pd.merge(df, self.horizons, on='cokey').query("hzname != 'R'").sort_values(by=['cokey', 'top'], ignore_index=True)

        return [SoilLayer(
                top    = row['top'],
                bottom = row['bottom'],
                **{p: row[p] for p in MAPPABLE_PARAMETERS},
            ) for _, row in df.iterrows()]

    def generate_soil_file(self, fn: Path | str, *, mukey: int | None=None, desc: str | None=None, soil_depth: float | None=None) -> None:
        if mukey is None:
            self.group_map_units(geometry=True)
            self.select_major_mapunit()
            mukey = self.mukey
        assert mukey is not None

        profile = self.get_soil_profile(mukey=mukey)
        desc = desc if desc is not None else _build_desc(self._get_muname(mukey), mukey, self.hsg)
        _generate_soil_file(fn, profile, desc=desc, hsg=self.hsg, slope=self.slope, soil_depth=soil_depth)

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _ensure_mukey(self) -> int:
        if self.mukey is None:
            self.select_major_mapunit()
        assert self.mukey is not None
        return self.mukey

    def _get_muname(self, mukey: int) -> str:
        assert self._mapunits is not None
        return self._mapunits[self._mapunits['mukey'] == mukey]['muname'].iloc[0]

    def _average_slope_hsg(self) -> None:
        assert self._mapunits is not None

        gdf        = self._mapunits[~self.non_soil_mask(self._mapunits)].copy()
        gdf['area'] = gdf.area

        self.slope = _weighted_average(gdf, 'slopegradwta')
        self.hsg   = _dominant_hsg(gdf)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _validate_geographic_input(lat_lon: LatLon | None, boundary: gpd.GeoDataFrame | None, lut_only: bool) -> None:
    if lut_only:
        return
    if lat_lon is None and boundary is None:
        raise ValueError("lat_lon or boundary must be provided when lut_only=False.")
    if lat_lon is not None and boundary is not None:
        raise ValueError("lat_lon and boundary are mutually exclusive — provide only one.")


def _build_desc(muname: str, mukey: int, hsg: str) -> str:
    lines = [
        f"# Soil file for MUNAME: {muname}, MUKEY: {mukey}",
        "# NO3, NH4, and fractions of horizontal and vertical bypass flows are default empirical values.",
    ]
    if not hsg:
        lines.append("# Hydrologic soil group MISSING DATA.")
    else:
        lines.append(f"# Hydrologic soil group {hsg}.")
        lines.append("# The curve number for row crops with straight row treatment is used.")
    return '\n'.join(lines) + '\n'


def _weighted_average(gdf: gpd.GeoDataFrame, col: str) -> float:
    df = gdf[['area', col]].dropna()
    if df.empty:
        return 0.0
    if len(df) == 1:
        return float(df[col].iloc[0])
    return float((df[col] * df['area']).sum() / df['area'].sum())


def _dominant_hsg(gdf: gpd.GeoDataFrame) -> str:
    df = gdf[['area', 'hydgrpdcd']].dropna()
    if df.empty:
        return ''
    df['hydgrpdcd'] = df['hydgrpdcd'].str[0]   # take first letter only
    return str(df.groupby('hydgrpdcd')['area'].sum().idxmax())


def _strip_slope_suffix(s: str) -> str:
    """Strip trailing slope class letter or letter+digit from a map unit symbol."""
    if s == 'N/A' or len(s) < 2:
        return s
    if s[-1].isupper() and (s[-2].isnumeric() or s[-2].islower()):
        return s[:-1]
    if s[-1].isnumeric() and s[-2].isupper() and (s[-3].isnumeric() or s[-3].islower()):
        return s[:-2]
    return s


def _read_lut(path: Path, state: str, table: str, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(_ssurgo_lut(path, state, table), usecols=columns)

    if table == 'chfrags':
        df = df.groupby('chkey').sum().reset_index()

    df.rename(
        columns={v.ssurgo_name: k for k, v in SSURGO_PARAMETERS.items()},
        inplace=True,
    )
    for key, param in SSURGO_PARAMETERS.items():
        if key in df.columns:
            df[key] *= param.multiplier

    return df


def _read_all_luts(path: Path, state: str) -> dict[str, pd.DataFrame]:
    luts = {}
    for lut_key, tables in LUT_TABLES.items():
        combined = pd.DataFrame()
        for table, columns in tables.items():
            df       = _read_lut(path, state, table, columns)
            combined = df if combined.empty else combined.merge(df, how='outer')
        luts[lut_key] = combined
    return luts


def _read_mupolygon(
    path:     Path,
    state:    str,
    boundary: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    if boundary is not None:
        boundary = boundary.to_crs(NAD83)

    gdf = gpd.read_file(
        _ssurgo_path(path, state),
        layer='MUPOLYGON',
        mask=shapely.union_all(boundary['geometry'].values) if boundary is not None else None,
    )
    if boundary is not None:
        gdf = gpd.clip(gdf, boundary, keep_geom_type=False)

    gdf.columns = [c.lower() for c in gdf.columns]
    gdf['mukey'] = gdf['mukey'].astype(int)
    return gdf


def _ssurgo_path(path: Path, state: str) -> Path:
    return path / f'gSSURGO_{state}.gdb'


def _ssurgo_lut(path: Path, state: str, table: str) -> Path:
    return path / f'{table}_{state}.csv'
