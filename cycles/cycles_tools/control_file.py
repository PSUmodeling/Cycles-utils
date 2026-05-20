from __future__ import annotations
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, get_type_hints
from ._base_file import write_file, resolve_dict_values, extract, parse_value, unwrap_optional

@dataclass(kw_only=True)
class SimulationYears:
    simulation_start_year: int
    simulation_end_year: int
    rotation_size: int

@dataclass(kw_only=True)
class InputFiles:
    crop_file: str='GenericCrops.crop'
    operation_file: str
    soil_file: str
    weather_file: str
    reinit_file: str='N/A'

@dataclass(kw_only=True)
class SimulationOptions:
    soil_layers: int
    co2_level: float=-999
    use_reinitialization: int=0
    adjusted_yields: int=0
    hydrology_option: int=1
    automatic_nitrogen: int=0
    automatic_phosphorus: int=0
    automatic_sulfur: int=0

@dataclass(kw_only=True)
class OutputControl:
    daily_weather_out: int=0
    daily_crop_out: int=0
    daily_residue_out: int=0
    daily_water_out: int=0
    daily_nitrogen_out: int=0
    daily_soil_carbon_out: int=0
    daily_soil_lyr_cn_out: int=0
    annual_soil_out: int=0
    annual_profile_out: int=0
    annual_nflux_out: int=0

@dataclass
class ControlConfig:
    simulation_years: SimulationYears
    input_files: InputFiles
    simulation_options: SimulationOptions
    output_control: OutputControl

def _build_control_config(control_dict: dict, simulation_dict: dict[str, Any] | None, input_dir: Path) -> ControlConfig:
    resolved = resolve_dict_values(control_dict, simulation_dict)

    if 'soil_layers' not in resolved:
        resolved['soil_layers'] = _get_soil_layers(input_dir / resolved['soil_file'])

    return ControlConfig(
        simulation_years=SimulationYears(**extract(SimulationYears, resolved)),
        input_files=InputFiles(**extract(InputFiles, resolved)),
        simulation_options=SimulationOptions(**extract(SimulationOptions, resolved)),
        output_control=OutputControl(**extract(OutputControl, resolved)),
    )


def _get_soil_layers(fn: Path) -> int:
    NUM_HEADER_LINES = 2
    try:
        lines = [line for line in fn.read_text().splitlines() if line.strip() and not line.strip().startswith('#')]
        return len(lines) - NUM_HEADER_LINES - 1
    except FileNotFoundError:
        warnings.warn(f"Soil file not found: {fn}")
        return -999


def generate_control_file(fn: str | Path, user_dict: dict, *, simulation_dict: dict[str, Any] | None = None) -> ControlConfig:
    fn = Path(fn)
    config = _build_control_config(user_dict, simulation_dict, fn.parent)
    write_file(fn, config)

    return config


def read_control_file(control: str | Path) -> ControlConfig:
    with open(Path(control)) as f:
        lines = f.read().splitlines()
    lines = iter([line for line in lines if (not line.strip().startswith('#')) and line.strip()])

    hints = get_type_hints(ControlConfig)   # resolves all string annotations → actual types

    control_dict = {}
    for f in fields(ControlConfig):
        target_class = unwrap_optional(hints[f.name])
        sub_hints = get_type_hints(target_class)
        control_dict[f.name] = target_class(**{sub_field.name: parse_value(next(lines), sub_field.name, sub_hints[sub_field.name]) for sub_field in fields(target_class)})

    return ControlConfig(**control_dict)
