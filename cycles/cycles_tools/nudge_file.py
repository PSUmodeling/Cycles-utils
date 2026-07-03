from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from ._base_file import write_file, resolve_dict_values, extract

@dataclass(kw_only=True)
class CalibrationMultipliers:
    soc_decomp_rate: float = field(default=1.0, metadata={'description': 'soil organic carbon decomposition rate'})
    residue_decomp_rate: float = field(default=1.0, metadata={'description': 'residue decomposition rate'})
    root_decomp_rate: float = field(default=1.0, metadata={'description': 'root decomposition rate'})
    rhizo_decomp_rate: float = field(default=1.0, metadata={'description': 'rhizodeposit decomposition rate'})
    manure_decomp_rate: float = field(default=1.0, metadata={'description': 'manure decomposition rate'})
    ferment_decomp_rate: float = field(default=1.0, metadata={'description': 'ferment decomposition rate'})
    microb_decomp_rate: float = field(default=1.0, metadata={'description': 'microbe decomposition rate'})
    soc_humif_power: float = field(default=1.0, metadata={'description': 'soil organic carbon humification exponent'})
    nitrif_rate: float = field(default=1.0, metadata={'description': 'nitrification rate'})
    pot_denitrif_rate: float = field(default=1.0, metadata={'description': 'potential denitrification rate'})
    denitrif_half_rate: float = field(default=1.0, metadata={'description': 'half saturation constant for denitrification'})
    decomp_half_resp: float = field(default=1.0, metadata={'description': 'decomposition half response to saturation (default 0.22)'})
    decomp_resp_power: float = field(default=1.0, metadata={'description': 'decomposition exponential response to saturation (default 3.0)'})
    root_progression: float = field(default=1.0, metadata={'description': 'rooting depth progression rate'})
    radiation_use_efficiency: float = field(default=1.0, metadata={'description': 'crop radiation use efficiency'})

@dataclass(kw_only=True)
class ParameterValues:
    kd_no3: float = field(default=0.0, metadata={'description': 'adsorption coefficient for NO3 (default 0.0 cm3/g)'})
    kd_nh4: float = field(default=5.6, metadata={'description': 'adsorption coefficient for NH4 (default 5.6 cm3/g)'})

@dataclass
class NudgeConfig:
    calibration_multipliers: CalibrationMultipliers
    parameter_values: ParameterValues


def _build_nudge_config(user_dict: dict, simulation_dict: dict[str, Any] | None) -> NudgeConfig:
    resolved = resolve_dict_values(user_dict, simulation_dict)

    return NudgeConfig(
        calibration_multipliers=CalibrationMultipliers(**extract(CalibrationMultipliers, resolved)),
        parameter_values=ParameterValues(**extract(ParameterValues, resolved)),
    )


def generate_nudge_file(fn: str | Path, user_dict: dict, *, simulation_dict: dict[str, Any] | None=None) -> None:
    """Write a Cycles nudge file from user-provided values.

    Provide either direct values or callables that accept a simulation row and return a value in `user_dict`. The
    parameter names should be in lowercase and correspond to the fields in Cycles nudge (calibration) files. If a field
    is not provided, it will be filled with a default value. If a field's value is a callable, it will be called with
    the `simulation_dict` to resolve its value.

    The default values for all calibration multipliers are `1.0`, and the default values for `kd_no3` and `kd_nh4` are
    `0.0` and `5.6`, respectively.

    Args:
        fn: Destination nudge file path.
        user_dict: Values or callables for nudge parameters.
        simulation_dict: Optional simulation row for callable resolution.
    """
    fn = Path(fn)
    config = _build_nudge_config(user_dict, simulation_dict)
    write_file(fn, config)
