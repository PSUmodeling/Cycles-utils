import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from .output_file import read_output

@dataclass
class ReinitFile:
    header: str
    column: str | None
    profile: bool


REINIT_FILE = [
    ReinitFile(header='STANRESIDUEC', column='standing_residue_carbon', profile=False),
    ReinitFile(header='FLATRESIDUEC', column='flat_residue_carbon', profile=False),
    ReinitFile(header='STANRESIDUEN', column='standing_residue_nitrogen', profile=False),
    ReinitFile(header='FLATRESIDUEN', column='flat_residue_nitrogen', profile=False),
    ReinitFile(header='MANURESURFACEC', column='surface_manure_residue_carbon', profile=False),
    ReinitFile(header='MANURESURFACEN', column='surface_manure_residue_nitrogen', profile=False),
    ReinitFile(header='FERMENTSURFACEC', column='surface_ferment_residue_carbon', profile=False),
    ReinitFile(header='FERMENTSURFACEN', column='surface_ferment_residue_nitrogen', profile=False),
    ReinitFile(header='STANRESIDUEWATER', column='standing_residue_moisture', profile=False),
    ReinitFile(header='FLATRESIDUEWATER', column='flat_residue_moisture', profile=False),
    ReinitFile(header='INFILTRATION', column=None, profile=False),
    ReinitFile(header='SMC', column='soil_moisture_content', profile=True),
    ReinitFile(header='NO3', column='NO3', profile=True),
    ReinitFile(header='NH4', column='NH4', profile=True),
    ReinitFile(header='SOC', column='soil_organic_carbon', profile=True),
    ReinitFile(header='SON', column='soil_organic_nitrogen', profile=True),
    ReinitFile(header='MBC', column='microbial_carbon', profile=True),
    ReinitFile(header='MBN', column='microbial_nitrogen', profile=True),
    ReinitFile(header='RESABGDC', column='shoot_residue_carbon', profile=True),
    ReinitFile(header='RESRTC', column='root_residue_carbon', profile=True),
    ReinitFile(header='RESRZC', column='rhizodeposit_residue_carbon', profile=True),
    ReinitFile(header='RESIDUEABGDN', column='shoot_residue_nitrogen', profile=True),
    ReinitFile(header='RESIDUERTN', column='root_residue_nitrogen', profile=True),
    ReinitFile(header='RESIDUERZN', column='rhizodeposit_residue_nitrogen', profile=True),
    ReinitFile(header='MANUREC', column='manure_residue_carbon', profile=True),
    ReinitFile(header='MANUREN', column='manure_residue_nitrogen', profile=True),
    ReinitFile(header='FERMENTC', column='ferment_residue_carbon', profile=True),
    ReinitFile(header='FERMENTN', column='ferment_residue_nitrogen', profile=True),
    ReinitFile(header='SATURATION', column=None, profile=True),
]

def _parse_value(row: pd.Series, column: str | None, layer: int | None) -> float | int:
    if column is None:
        return -999
    else:
        return row[column if layer is None else f'{column}.{layer}']


def generate_reinit_file(out_path: str | Path, in_path: str | Path, doy: int) -> None:
    out_path = Path(out_path)
    in_path = Path(in_path)

    output_df, _ = read_output(in_path, 'reinit')
    n_soil_layers = len([col for col in output_df.columns if col.startswith('soil_moisture_content')])
    print(f'Found {n_soil_layers} soil layers in output file.')
    output_df['year'] = output_df['date'].dt.year
    output_df = output_df[output_df['date'].dt.dayofyear == doy].reset_index()

    strs = []
    for _, row in output_df.iterrows():
        strs.append(f'{"YEAR":<8}{row["year"]:<8d}{"DOY":<8}{doy}')
        strs.append(''.join([f'{var.header:<20}' for var in REINIT_FILE if not var.profile]))   # header
        strs.append(''.join([f'{_parse_value(row, var.column, None):<20}' for var in REINIT_FILE if not var.profile]))   # values

        strs.append(''.join([f'{"LAYER":<12}'] + [f'{var.header:<12}' for var in REINIT_FILE if var.profile]))   # header
        for layer in range(1, n_soil_layers + 1):
            strs.append(''.join([f'{layer:<12}'] + [f'{_parse_value(row, var.column, layer):<12}' for var in REINIT_FILE if var.profile]))   # values
        strs.append('')

    out_path.write_text('\n'.join(strs) + '\n')
