from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from ._base_file import write_file, resolve_dict_values, extract
from ._base_file import FMT_2F

@dataclass(kw_only=True)
class CalibrationMultipliers:
    soc_decomp_rate: float = field(default=1.0, metadata={'description': 'soil organic carbon decomposition rate', 'fmt': FMT_2F})
    residue_decomp_rate: float = field(default=1.0, metadata={'description': 'residue decomposition rate', 'fmt': FMT_2F})
    root_decomp_rate: float = field(default=1.0, metadata={'description': 'root decomposition rate', 'fmt': FMT_2F})
    rhizo_decomp_rate: float = field(default=1.0, metadata={'description': 'rhizodeposit decomposition rate', 'fmt': FMT_2F})
    manure_decomp_rate: float = field(default=1.0, metadata={'description': 'manure decomposition rate', 'fmt': FMT_2F})
    ferment_decomp_rate: float = field(default=1.0, metadata={'description': 'ferment decomposition rate', 'fmt': FMT_2F})
    microb_decomp_rate: float = field(default=1.0, metadata={'description': 'microbe decomposition rate', 'fmt': FMT_2F})
    soc_humif_power: float = field(default=1.0, metadata={'description': 'soil organic carbon humification exponent', 'fmt': FMT_2F})
    nitrif_rate: float = field(default=1.0, metadata={'description': 'nitrification rate', 'fmt': FMT_2F})
    pot_denitrif_rate: float = field(default=1.0, metadata={'description': 'potential denitrification rate', 'fmt': FMT_2F})
    denitrif_half_rate: float = field(default=1.0, metadata={'description': 'half saturation constant for denitrification', 'fmt': FMT_2F})
    decomp_half_resp: float = field(default=1.0, metadata={'description': 'decomposition half response to saturation (default 0.22)', 'fmt': FMT_2F})
    decomp_resp_power: float = field(default=1.0, metadata={'description': 'decomposition exponential response to saturation (default 3.0)', 'fmt': FMT_2F})
    root_progression: float = field(default=1.0, metadata={'description': 'rooting depth progression rate', 'fmt': FMT_2F})
    radiation_use_efficiency: float = field(default=1.0, metadata={'description': 'crop radiation use efficiency', 'fmt': FMT_2F})
    drainage: float = field(default=1.0, metadata={'description': 'subsurface drainage', 'fmt': FMT_2F})

@dataclass(kw_only=True)
class ParameterValues:
    kd_no3: float = field(default=0.0, metadata={'description': 'adsorption coefficient for NO3 (default 0.0 cm3/g)', 'fmt': FMT_2F})
    kd_nh4: float = field(default=5.6, metadata={'description': 'adsorption coefficient for NH4 (default 5.6 cm3/g)', 'fmt': FMT_2F})

@dataclass
class NudgeConfig:
    calibration_multipliers: CalibrationMultipliers
    parameter_values: ParameterValues


def _build_nudge_config(user_dict: dict[str, Any], calibration_dict: dict[str, Any] | None) -> NudgeConfig:
    resolved = resolve_dict_values(user_dict, calibration_dict)

    return NudgeConfig(
        calibration_multipliers=CalibrationMultipliers(**extract(CalibrationMultipliers, resolved)),
        parameter_values=ParameterValues(**extract(ParameterValues, resolved)),
    )


def generate_nudge_file(file_path: str | Path, user_dict: dict[str, Any], *, calibration_dict: dict[str, Any] | None=None) -> None:
    """Write a Cycles nudge file from user-provided values.

    Provide either direct values or callables that accept a calibration configuration and return a value in `user_dict`.
    The parameter names should be in lowercase and correspond to the fields in Cycles nudge (calibration) files. If a
    field is not provided, it will be filled with a default value. If a field's value is a callable, it will be called
    with the `calibration_dict` to resolve its value.

    The default values for all calibration multipliers are `1.0`, and the default values for `kd_no3` and `kd_nh4` are
    `0.0` and `5.6`, respectively.

    Args:
        file_path: Destination nudge file path.
        user_dict: Values or callables for nudge parameters.
        calibration_dict: Optional simulation row for callable resolution.
    """
    file_path = Path(file_path)
    config = _build_nudge_config(user_dict, calibration_dict)
    write_file(file_path, config)
