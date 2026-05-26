from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from ._base_file import write_file, resolve_dict_values, extract

@dataclass(kw_only=True)
class CalibrationMultipliers:
    soc_decomp_rate: float = 1.0
    residue_decomp_rate: float = 1.0
    root_decomp_rate: float = 1.0
    rhizo_decomp_rate: float = 1.0
    manure_decomp_rate: float = 1.0
    ferment_decomp_rate: float = 1.0
    microb_decomp_rate: float = 1.0
    soc_humif_power: float = 1.0
    nitrif_rate: float = 1.0
    pot_denitrif_rate: float = 1.0
    denitrif_half_rate: float = 1.0
    decomp_half_resp: float = 1.0
    decomp_resp_power: float = 1.0
    root_progression: float = 1.0
    radiation_use_efficiency: float = 1.0

@dataclass(kw_only=True)
class ParameterValues:
    kd_no3: float = 0.0
    kd_nh4: float = 5.6

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

    Args:
        fn: Destination nudge file path.
        user_dict: Values or callables for nudge parameters.
        simulation_dict: Optional simulation row for callable resolution.
    """
    fn = Path(fn)
    config = _build_nudge_config(user_dict, simulation_dict)
    write_file(fn, config)
