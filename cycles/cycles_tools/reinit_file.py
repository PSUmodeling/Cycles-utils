from dataclasses import dataclass
from pathlib import Path
from .output_file import read_output

@dataclass(frozen=True)
class ReinitVar:
    header: str
    column: str | None       # None → write placeholder value


SURFACE_VARS: tuple[ReinitVar, ...] = (
    ReinitVar('STANRESIDUEC', 'standing_residue_carbon'),
    ReinitVar('FLATRESIDUEC', 'flat_residue_carbon'),
    ReinitVar('STANRESIDUEN', 'standing_residue_nitrogen'),
    ReinitVar('FLATRESIDUEN', 'flat_residue_nitrogen'),
    ReinitVar('MANURESURFACEC', 'surface_manure_residue_carbon'),
    ReinitVar('MANURESURFACEN', 'surface_manure_residue_nitrogen'),
    ReinitVar('FERMENTSURFACEC', 'surface_ferment_residue_carbon'),
    ReinitVar('FERMENTSURFACEN', 'surface_ferment_residue_nitrogen'),
    ReinitVar('STANRESIDUEWATER', 'standing_residue_moisture'),
    ReinitVar('FLATRESIDUEWATER', 'flat_residue_moisture'),
    ReinitVar('INFILTRATION', None),
)

PROFILE_VARS: tuple[ReinitVar, ...] = (
    ReinitVar('SMC', 'soil_moisture_content'),
    ReinitVar('NO3', 'NO3'),
    ReinitVar('NH4', 'NH4'),
    ReinitVar('SOC', 'soil_organic_carbon'),
    ReinitVar('SON', 'soil_organic_nitrogen'),
    ReinitVar('MBC', 'microbial_carbon'),
    ReinitVar('MBN', 'microbial_nitrogen'),
    ReinitVar('RESABGDC', 'shoot_residue_carbon'),
    ReinitVar('RESRTC', 'root_residue_carbon'),
    ReinitVar('RESRZC', 'rhizodeposit_residue_carbon'),
    ReinitVar('RESIDUEABGDN', 'shoot_residue_nitrogen'),
    ReinitVar('RESIDUERTN', 'root_residue_nitrogen'),
    ReinitVar('RESIDUERZN', 'rhizodeposit_residue_nitrogen'),
    ReinitVar('MANUREC', 'manure_residue_carbon'),
    ReinitVar('MANUREN', 'manure_residue_nitrogen'),
    ReinitVar('FERMENTC', 'ferment_residue_carbon'),
    ReinitVar('FERMENTN', 'ferment_residue_nitrogen'),
    ReinitVar('SATURATION', None),
)

_PLACEHOLDER = -999
_SURFACE_WIDTH = 20
_PROFILE_WIDTH = 16


def _surface_value(row: pd.Series, var: ReinitVar) -> float | int:
    return _PLACEHOLDER if var.column is None else row[var.column]


def _profile_value(row: pd.Series, var: ReinitVar, layer: int) -> float | int:
    return _PLACEHOLDER if var.column is None else row[f'{var.column}.{layer}']


def _format_reinit_block(row: pd.Series, doy: int, n_layers: int) -> list[str]:
    """Return all lines for a single year/doy reinit block."""
    w  = _SURFACE_WIDTH
    wp = _PROFILE_WIDTH

    lines = [
        f'{"YEAR":<8}{row["year"]:<8d}{"DOY":<8}{doy}',
        ''.join(f'{v.header:<{w}}' for v in SURFACE_VARS),
        ''.join(f'{_surface_value(row, v):<{w}}' for v in SURFACE_VARS),
        ''.join([f'{"LAYER":<{wp}}'] + [f'{v.header:<{wp}}' for v in PROFILE_VARS]),
        *[''.join([f'{layer:<{wp}}'] + [f'{_profile_value(row, v, layer):<{wp}}' for v in PROFILE_VARS]) for layer in range(1, n_layers + 1)],
        '',   # blank line between blocks
    ]
    return lines


def generate_reinit_file(out_path: str | Path, in_path: str | Path, doy: int) -> None:
    out_path = Path(out_path)
    in_path  = Path(in_path)

    output_df, _ = read_output(in_path, 'reinit')

    n_layers = sum(1 for col in output_df.columns if col.startswith('soil_moisture_content'))

    output_df['year'] = output_df['date'].dt.year
    filtered = output_df[output_df['date'].dt.dayofyear == doy].reset_index(drop=True)

    lines: list[str] = []
    for _, row in filtered.iterrows():
        lines.extend(_format_reinit_block(row, doy, n_layers))

    out_path.write_text('\n'.join(lines) + '\n')
