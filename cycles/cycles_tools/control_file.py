from __future__ import annotations
import warnings
from dataclasses import dataclass, field, fields
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
    crop_file: str = 'GenericCrops.crop'
    operation_file: str
    soil_file: str
    weather_file: str
    reinit_file: str = 'N/A'

@dataclass(kw_only=True)
class SimulationOptions:
    soil_layers: int
    co2_level: float = field(default=-999, metadata={'description': 'atmospheric CO2 concentration (ppm). Use co2.txt file if set to -999'})
    use_reinitialization: int = 0
    adjusted_yields: int = 0
    hydrology_option: int = field(default=1, metadata={'description': "1: gravity driven, 2: Richards' equation with Crank-Nicholson, 3: Richards' equation with CVode"})
    automatic_nitrogen: int = 0
    automatic_phosphorus: int = 0
    automatic_sulfur: int = 0

@dataclass(kw_only=True)
class OutputControl:
    daily_weather_out: int = 0
    daily_crop_out: int = 0
    daily_residue_out: int = 0
    daily_water_out: int = 0
    daily_nitrogen_out: int = 0
    daily_soil_carbon_out: int = 0
    daily_soil_lyr_cn_out: int = 0
    annual_soil_out: int = 0
    annual_profile_out: int = 0
    annual_nflux_out: int = 0

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


def _get_soil_layers(file_path: Path) -> int:
    NUM_HEADER_LINES = 2
    try:
        lines = [line for line in file_path.read_text().splitlines() if line.strip() and not line.strip().startswith('#')]
        return len(lines) - NUM_HEADER_LINES - 1
    except FileNotFoundError:
        warnings.warn(f"Soil file not found: {file_path}")
        return -999


def generate_control_file(file_path: str | Path, user_dict: dict[str, Any], *, simulation_dict: dict[str, Any] | None=None) -> ControlConfig:
    """Generate and write a Cycles control file.

    Provide either direct values or callables that accept a simulation configuration and return a value in `user_dict`.
    The parameter names should be in lowercase and correspond to the fields in Cycles simulation control files. If a
    field is not provided, it will be filled with a default value. If a field's value is a callable, it will be called
    with the `simulation_dict` to resolve its value.

    The following fields are required in `user_dict`:

      - `simulation_start_year`
      - `simulation_end_year`
      - `rotation_size`
      - `operation_file`
      - `soil_file`
      - `weather_file`

    The default values for other fields are:

      - `crop_file`: `GenericCrops.crop`
      - `reinit_file`: `N/A`
      - `soil_layers`: inferred from the soil file (if not provided)
      - `co2_level`: `-999`
      - `use_reinitialization`: `0`
      - `adjusted_yields`: `0`
      - `hydrology_option`: `1`
      - `automatic_nitrogen`: `0`
      - `automatic_phosphorus`: `0`
      - `automatic_sulfur`: `0`

    All output control fields default to `0`.

    Args:
        file_path: Destination control file path.
        user_dict: Values or callables for control fields.
        simulation_dict: Optional simulation row for callable resolution.

    Returns:
        The generated control configuration.
    """
    file_path = Path(file_path)
    config = _build_control_config(user_dict, simulation_dict, file_path.parent)
    write_file(file_path, config)

    return config


def read_control_file(file_path: str | Path) -> ControlConfig:
    """Parse a Cycles control file into a `ControlConfig` dataclass instance.

    Args:
        file_path: Path to a Cycles control file path.

    Returns:
        Control configuration.
    """
    with open(Path(file_path)) as f:
        lines = f.read().splitlines()
    lines = iter([line for line in lines if (not line.strip().startswith('#')) and line.strip()])

    hints = get_type_hints(ControlConfig)   # resolves all string annotations → actual types

    control_dict = {}
    for f in fields(ControlConfig):
        target_class = unwrap_optional(hints[f.name])
        sub_hints = get_type_hints(target_class)
        control_dict[f.name] = target_class(**{sub_field.name: parse_value(next(lines), sub_field.name, sub_hints[sub_field.name]) for sub_field in fields(target_class)})

    return ControlConfig(**control_dict)
