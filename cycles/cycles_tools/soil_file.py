from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, asdict, fields
from pathlib import Path

CURVE_NUMBERS: dict[str, int] = {
    'A': 67,
    'B': 78,
    'C': 85,
    'D': 89,
}

MAPPABLE_PARAMETERS: list[str] = ['clay', 'sand', 'soc', 'bulk_density', 'coarse_fragments', 'pH']

@dataclass
class SoilParameter:
    name:     str
    header:   str
    unit:     str
    fmt:      str
    sentinel: str | None=None

    def format(self, value: float | None, *, last: bool=False) -> str:
        sep = '' if last else '\t'
        if value is None:
            placeholder = self.sentinel if self.sentinel else '-999'
            return ('%-7s' + sep) % placeholder
        return (self.fmt + sep) % value


VARIABLES: list[SoilParameter] = [
    SoilParameter('layer', 'LAYER', '#', '%-7d'),
    SoilParameter('thickness', 'THICK', 'm', '%-7.2f'),
    SoilParameter('clay', 'CLAY', '%', '%-7.1f'),
    SoilParameter('sand', 'SAND', '%', '%-7.1f'),
    SoilParameter('soc', 'SOC', '%', '%-7.2f'),
    SoilParameter('bulk_density', 'BD', 'Mg/m3', '%-7.2f'),
    SoilParameter('fc', 'FC', 'm3/m3', '%-7d'),
    SoilParameter('pwp', 'PWP', 'm3/m3', '%-7d'),
    SoilParameter('son', 'SON', 'kg/ha', '%-7d'),
    SoilParameter('no3', 'NO3', 'kg/ha', '%-7.1f'),
    SoilParameter('nh4', 'NH4', 'kg/ha', '%-7.1f'),
    SoilParameter('coarse_fragments', 'ROCK', 'm3/m3', '%-7.2f'),
    SoilParameter('byp_h', 'BYP_H', '-', '%-7.1f', '0.0'),
    SoilParameter('byp_v', 'BYP_V', '-', '%-7.1f', '0.0'),
    SoilParameter('pH', 'pH', '-', '%.1f'),
]

@dataclass(kw_only=True)
class SoilLayer:
    top: float
    bottom: float
    clay: float | None=None
    sand: float | None=None
    soc: float | None=None
    bulk_density: float | None=None
    no3: float | None=None
    nh4: float | None=None
    fc: float | None=None
    pwp: float | None=None
    son: float | None=None
    coarse_fragments: float | None=None
    byp_h: float=0.0
    byp_v: float=0.0
    pH: float | None=None

    @property
    def thickness(self) -> float:
        return self.bottom - self.top


    def overlap_with(self, top: float, bottom: float) -> float:
        """Fraction of this layer's thickness overlapping with [top, bottom]."""
        return max(0.0, min(self.bottom, bottom) - max(self.top, top)) / self.thickness


    def format_row(self, ind: int) -> str:
        strs = []
        for i, param in enumerate(VARIABLES):
            if param.name == 'layer':
                value = ind
            elif param.name == 'thickness':
                value = self.thickness
            else:
                value = getattr(self, param.name)
            strs.append(param.format(value, last=(i == len(VARIABLES) - 1)))

        return ''.join(strs)


    def to_dict(self) -> dict:
        return {**asdict(self), 'thickness': self.thickness}


DEFAULT_PROFILE: list[SoilLayer] = [
    SoilLayer(top=0.0,  bottom=0.05, no3=10.0, nh4=1.0),
    SoilLayer(top=0.05, bottom=0.1,  no3=10.0, nh4=1.0),
    SoilLayer(top=0.1,  bottom=0.2,  no3=7.0,  nh4=1.0),
    SoilLayer(top=0.2,  bottom=0.4,  no3=4.0,  nh4=1.0),
    SoilLayer(top=0.4,  bottom=0.6,  no3=2.0,  nh4=1.0),
    SoilLayer(top=0.6,  bottom=0.8,  no3=1.0,  nh4=1.0),
    SoilLayer(top=0.8,  bottom=1.0,  no3=1.0,  nh4=1.0),
    SoilLayer(top=1.0,  bottom=1.2,  no3=1.0,  nh4=1.0),
    SoilLayer(top=1.2,  bottom=1.4,  no3=1.0,  nh4=1.0),
    SoilLayer(top=1.4,  bottom=1.6,  no3=1.0,  nh4=1.0),
    SoilLayer(top=1.6,  bottom=1.8,  no3=1.0,  nh4=1.0),
    SoilLayer(top=1.8,  bottom=2.0,  no3=1.0,  nh4=1.0),
]


def _trim(target: list[SoilLayer], measured_bottom: float, soil_depth: float | None=None) -> list[SoilLayer]:
    """Keep target layers with more than 50% overlap with the effective depth."""
    cutoff = min(measured_bottom, soil_depth) if soil_depth is not None else measured_bottom
    return [l for l in target if l.overlap_with(l.top, cutoff) > 0.5]


def _map_layer(target: SoilLayer, measured: list[SoilLayer], parameters: list[str]) -> SoilLayer:
    return SoilLayer(
        top = target.top,
        bottom = target.bottom,
        no3 = target.no3,
        nh4 = target.nh4,
        **{p: _weighted_average(p, target, measured) for p in parameters},
    )


def _weighted_average(parameter: str, target: SoilLayer, measured: list[SoilLayer]) -> float | None:
    valid = [(w, v) for m in measured if (w := m.overlap_with(target.top, target.bottom)) > 0 and (v := getattr(m, parameter)) is not None]
    if not valid:
        return None
    total = sum(w for w, _ in valid)
    return sum(w * v for w, v in valid) / total


def _resolve_curve_number(curve_number: float | None, hsg: str) -> float:
    if curve_number is not None:
        return curve_number
    return CURVE_NUMBERS[hsg[0]] if hsg else -999


def _header_line() -> str:
    return ('%-7s\t' * (len(VARIABLES) - 1) + '%s') % tuple(v.header for v in VARIABLES)


def _unit_line() -> str:
    return ('%-7s\t' * (len(VARIABLES) - 1) + '%s') % tuple(v.unit for v in VARIABLES)


def _render_soil_file(layers: list[SoilLayer], desc: str, slope: float, curve_number: float | None, hsg: str) -> list[str]:
    lines = []
    if desc:
        lines.append(desc)
    lines.append("%-15s\t%d"   % ("CURVE_NUMBER", _resolve_curve_number(curve_number, hsg)))
    lines.append("%-15s\t%.2f" % ("SLOPE", slope))
    lines.append(_header_line())
    lines.append(_unit_line())
    lines.extend(layer.format_row(ind + 1) for ind, layer in enumerate(layers))
    return lines


def _parse_header(lines: list[str]) -> tuple[dict, list[str]]:
    """Extract curve_number and slope, return remaining lines."""
    meta = {}
    i = 0
    for key, cast in [('curve_number', float), ('slope', float)]:
        meta[key] = cast(lines[i].split()[1])
        i += 1
    return meta, lines[i:]   # remaining lines are column headers + data


def _parse_layers(lines: list[str]) -> list[SoilLayer]:
    """Parse the column header line and data rows into SoilLayer instances."""
    # First line is the column header — map column names to VARIABLES by header name
    col_names = [col.lower() for col in lines[0].split()]
    param_by_header = {v.header.lower(): v for v in VARIABLES}

    cumulative_depth = 0.0
    layers = []
    for row in lines[1:]:   # skip the header line
        tokens = row.split()
        layer = _parse_layer_row(col_names, tokens, param_by_header, cumulative_depth)
        layers.append(layer)
        cumulative_depth += layer.thickness
    return layers


def _parse_layer_row(col_names: list[str], tokens: list[str], param_by_header: dict[str, SoilParameter], cumulative_depth: float) -> SoilLayer:
    """Parse a single data row into a SoilLayer."""
    kwargs = {}
    thickness = None

    for col, token in zip(col_names, tokens):
        if col.lower() == 'layer':
            continue    # layer index is not stored in the file — we rebuild it from VARIABLES defaults
        param = param_by_header.get(col)
        if param is None:
            continue
        value = _parse_token(token, param)
        if param.name == 'thickness':
            thickness = value   # held separately — derived into top/bottom
        else:
            kwargs[param.name] = value

    # Reconstruct top/bottom from layer index and cumulative thickness
    # These are not stored in the file — we rebuild them from VARIABLES defaults
    kwargs['top'] = cumulative_depth
    kwargs['bottom'] = cumulative_depth + thickness

    return SoilLayer(**kwargs)


def _parse_token(token: str, param: SoilParameter) -> float | None:
    """Cast a token string to the appropriate type, returning None for sentinels."""
    if (param.sentinel and token.strip() == param.sentinel) or (token.strip() == '-999'):
        return None
    if param.fmt.endswith('d'):
        return int(token)
    if param.fmt.endswith('f'):
        return float(token)
    return token


def _row_to_layer(row: pd.Series) -> SoilLayer:
    """Convert a single DataFrame row to a SoilLayer."""
    valid_fields = {f.name for f in fields(SoilLayer)}
    kwargs = {
        col: (None if pd.isna(val) else val)
        for col, val in row.items()
        if col in valid_fields
    }
    return SoilLayer(**kwargs)


def map_layers(measured: list[SoilLayer], target: list[SoilLayer]=DEFAULT_PROFILE, parameters: list[str]=MAPPABLE_PARAMETERS, soil_depth: float | None=None) -> list[SoilLayer]:
    """Map measured soil properties onto a target profile via depth-weighted averaging."""
    trimmed = _trim(target, measured[-1].bottom, soil_depth)
    return [_map_layer(layer, measured, parameters) for layer in trimmed]


def map_to_dataframe(layers: list[SoilLayer]) -> pd.DataFrame:
    """Convert a list of SoilLayer instances to a DataFrame."""
    return pd.DataFrame([{**asdict(layer), 'thickness': layer.thickness} for layer in layers])


def generate_soil_file(fn: str | Path, measured: list[SoilLayer], *,
    target: list[SoilLayer]=DEFAULT_PROFILE, parameters: list[str]=MAPPABLE_PARAMETERS, soil_depth: float | None=None,
    desc: str = '', slope: float=0.0, curve_number: float | None=None, hsg: str = '') -> list[SoilLayer]:
    """Map measured soil layers onto the target profile and write a Cycles soil file.
    Returns the mapped layers for further use.
    """
    if curve_number is not None and hsg:
        raise ValueError("Only one of curve_number and hsg can be provided.")
    layers = map_layers(measured, target, parameters, soil_depth)
    Path(fn).write_text('\n'.join(_render_soil_file(layers, desc, slope, curve_number, hsg)) + '\n')
    return layers


def read_soil_file(fn: str | Path) -> tuple[list[SoilLayer], dict]:
    """Read a Cycles soil file and return:
    - a list of SoilLayer instances
    - a dict of header metadata (curve_number, slope)
    """
    lines = [line for line in Path(fn).read_text().splitlines() if line.strip() and not line.strip().startswith('#')]

    meta, data_lines = _parse_header(lines)
    layers = _parse_layers(data_lines)
    return layers, meta


def from_dataframe(df: pd.DataFrame) -> list[SoilLayer]:
    """Convert a DataFrame to a list of SoilLayer instances."""
    return [_row_to_layer(row) for _, row in df.iterrows()]
