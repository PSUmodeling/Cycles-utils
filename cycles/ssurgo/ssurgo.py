from __future__ import annotations
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from dataclasses import dataclass
from matplotlib.axes import Axes
from pathlib import Path
from shapely.geometry import Point
from cycles.cycles_tools import generate_soil_file as _generate_soil_file
from cycles.cycles_tools import SoilLayer, MAPPABLE_PARAMETERS

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

@dataclass(frozen=True)
class SsurgoParameter:
    ssurgo_name: str
    multiplier: float
    table: str
    unit: str

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


class MapUnitGeoDataFrame(gpd.GeoDataFrame):
    def plot(self, **kwargs) -> Axes:
        n = len(self)
        kwargs.setdefault('column', 'musym')
        kwargs.setdefault('legend', True)
        kwargs.setdefault(
            'cmap',
            'tab20' if n > 20 else _truncate_colormap(plt.get_cmap('tab20'), 0, n / 20),
        )
        ax = super().plot(**kwargs)
        ax.axis('off')
        ax.set_aspect('equal')
        return ax


class Ssurgo:
    """Load SSURGO lookup tables and extract soil profiles for locations."""

    def __init__(self, path: str | Path, state: str, *, lat_lon: LatLon | None=None, boundary: gpd.GeoDataFrame | None=None) -> None:
        """Initialize SSURGO lookup tables and optional spatial subset.

        When a location (`lat_lon`) or boundary polygon is provided, the map-unit table is filtered to include only
        those map units that intersect the location or polygon; the map-units are also grouped by name and symbol, and
        the major map unit is selected for profile extraction.

        Args:
            path: Directory containing SSURGO geodatabase and lookup CSV files.
            state: State identifier used in SSURGO file naming.
            lat_lon: Optional latitude/longitude for point-based spatial filtering.
            boundary: Optional boundary GeoDataFrame for polygon-based filtering.

        Returns:
            None.

        Raises:
            ValueError: If both ``lat_lon`` and ``boundary`` are provided.
        """
        _validate_geographic_input(lat_lon, boundary)

        self.state: str = state
        self._mapunits: gpd.GeoDataFrame | pd.DataFrame
        self.components: pd.DataFrame
        self.horizons: pd.DataFrame
        self.grouped_mapunits: MapUnitGeoDataFrame | None = None
        self.mukey: int | None = None
        self.slope: float | None = None
        self.hsg: str = ''

        path = Path(path)
        luts = _read_all_luts(path, state)
        self.components = luts['component']
        self.horizons = luts['horizon']

        if lat_lon is None and boundary is None:
            self._mapunits = luts['mapunit']
            return

        if lat_lon is not None:
            boundary = gpd.GeoDataFrame(
                {'name': ['point']},
                geometry=[Point(lat_lon[1], lat_lon[0])],
                crs='epsg:4326',
            )
        gdf = _read_mupolygon(path, state, boundary)
        self._mapunits = gdf.merge(luts['mapunit'], on='mukey', how='left')
        self.components = self.components[self.components['mukey'].isin(self._mapunits['mukey'].unique())]
        self.horizons = self.horizons[self.horizons['cokey'].isin(self.components['cokey'].unique())]

        self._group_map_units()
        self._select_major_mapunit()
        self._average_slope_hsg()


    @property
    def mapunits(self) -> MapUnitGeoDataFrame | pd.DataFrame:
        """Return loaded map-unit table.

        Returns:
            Map-unit table as GeoDataFrame/DataFrame, or None if not loaded.
        """
        if isinstance(self._mapunits, gpd.GeoDataFrame):
            return MapUnitGeoDataFrame(self._mapunits, geometry=self._mapunits.geometry.name, crs=self._mapunits.crs)
        else:
            return self._mapunits


    @property
    def muname(self) -> str:
        """Return map-unit name for the currently selected MUKEY.

        Returns:
            Map-unit name string.
        """
        self._ensure_mukey()
        assert self.mukey is not None
        return self._get_muname(self.mukey)


    @property
    def musym(self) -> str:
        """Return map-unit symbol for the currently selected MUKEY.

        Returns:
            Map-unit symbol string.
        """
        self._ensure_mukey()
        assert self._mapunits is not None
        return self._mapunits[self._mapunits['mukey'] == self.mukey]['musym'].iloc[0]


    def _group_map_units(self) -> None:
        gmu = self._mapunits.copy()
        gmu['muname'] = gmu['muname'].map(lambda name: name.split(',')[0])  # type: ignore
        gmu['musym'] = gmu['musym'].map(_strip_slope_suffix)

        mask = self.non_soil_mask(gmu)
        gmu.loc[mask, 'muname'] = 'Water, urban, etc.'
        gmu.loc[mask, 'mukey'] = -999
        gmu.loc[mask, 'musym'] = 'N/A'

        gmu = gmu.dissolve(
            by='muname',
            aggfunc={'mukey': 'first', 'musym': 'first', 'shape_area': 'sum'},
        ).reset_index() # type: ignore

        self.grouped_mapunits = MapUnitGeoDataFrame(gmu, geometry=gmu.geometry.name, crs=gmu.crs)


    def non_soil_mask(self, mapunits: pd.DataFrame | gpd.GeoDataFrame) -> pd.Series:
        """Build a mask for non-soil or urban map units.

        Args:
            mapunits: Map-unit table to evaluate.

        Returns:
            Boolean Series where True indicates non-soil or urban classes.
        """
        return (
            mapunits['mukey'].isna() |
            mapunits['muname'].isin(SSURGO_NON_SOIL_TYPES) |
            mapunits['muname'].str.contains('|'.join(SSURGO_URBAN_TYPES), na=False)
        )


    def _select_major_mapunit(self) -> None:
        assert self.grouped_mapunits is not None
        gdf = self.grouped_mapunits[~self.non_soil_mask(self.grouped_mapunits)].copy()
        gdf['area'] = gdf.area
        self.mukey  = int(gdf.loc[gdf['area'].idxmax(), 'mukey'])   # type: ignore


    def get_soil_profile(self, *, mukey: int | None=None, major_only: bool=True) -> list[SoilLayer]:
        """Build a soil profile from SSURGO components and horizons.

        Args:
            mukey: Optional map-unit key. If omitted, the selected major MUKEY is used.
            major_only: If True, include only components marked as major.

        Returns:
            Soil profile as a list of ``SoilLayer`` records.
        """
        mukey = mukey or self._ensure_mukey()

        df = self.components[self.components['mukey'] == int(mukey)].copy()
        if major_only:
            df = df[df['majcompflag'] == 'Yes']

        df = pd.merge(df, self.horizons, on='cokey').query("hzname != 'R'").sort_values(by=['cokey', 'top'], ignore_index=True)

        return [SoilLayer(
                top = row['top'],
                bottom = row['bottom'],
                **{p: None if pd.isna(row[p]) else row[p] for p in MAPPABLE_PARAMETERS},    # type: ignore
            ) for _, row in df.iterrows()]


    def generate_soil_file(self, fn: Path | str, *,
        mukey: int | None=None, desc: str | None=None, hsg: str | None=None, slope: float | None=None, soil_depth: float | None=None) -> None:
        """Generate a Cycles soil file from SSURGO profile data.

        Args:
            fn: Output soil file path.
            mukey: Optional map-unit key. If omitted, dominant MUKEY is used.
            desc: Optional custom header text for the output file.
            hsg: Optional hydrologic soil group; inferred from map unit if omitted.
            slope: Optional slope value; inferred from map unit if omitted.
            soil_depth: Optional maximum depth (m) used during profile mapping.

        Returns:
            None.
        """
        if mukey is not None:
            hsg = self._mapunits[self._mapunits['mukey'] == mukey]['hydgrpdcd'].iloc[0] if hsg is None else hsg
            slope = self._mapunits[self._mapunits['mukey'] == mukey]['slopegradwta'].iloc[0] if slope is None else slope
        else:
            mukey = self._ensure_mukey()
            hsg = self.hsg if hsg is None else hsg
            slope = self.slope if slope is None else slope

        assert mukey is not None and hsg is not None and slope is not None

        profile = self.get_soil_profile(mukey=mukey)
        desc = desc if desc is not None else _build_desc(self._get_muname(mukey), mukey, hsg)
        _generate_soil_file(fn, profile, desc=desc, hsg=hsg, slope=slope, soil_depth=soil_depth)


    def _ensure_mukey(self) -> int:
        assert self.mukey is not None, "A major soil map unit has not been selected, because a location or boundary has not been provided."
        return self.mukey


    def _get_muname(self, mukey: int) -> str:
        return self._mapunits[self._mapunits['mukey'] == mukey]['muname'].iloc[0]


    def _average_slope_hsg(self) -> None:
        gdf = self._mapunits[~self.non_soil_mask(self._mapunits)].copy()
        gdf['area'] = gdf.area

        self.slope = _weighted_average(gdf, 'slopegradwta') # type: ignore
        self.hsg = _dominant_hsg(gdf)   # type: ignore


def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    return colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n))
    )


def _validate_geographic_input(lat_lon: LatLon | None, boundary: gpd.GeoDataFrame | None) -> None:
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


def _weighted_average(gdf: gpd.GeoDataFrame, col: str) -> float | None:
    df = gdf[['area', col]].dropna()
    if df.empty:
        return None
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


def _read_mupolygon(path: Path, state: str, boundary: gpd.GeoDataFrame | None=None) -> gpd.GeoDataFrame:
    if boundary is not None:
        boundary = boundary.to_crs(NAD83)

    gdf = gpd.read_file(
        _ssurgo_path(path, state),
        layer='MUPOLYGON',
        mask=shapely.union_all(boundary['geometry'].values) if boundary is not None else None,  # type: ignore
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
