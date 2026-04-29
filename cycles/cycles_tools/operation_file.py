from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, fields
from typing import get_type_hints, Protocol, Any
from ._base_file import parse_value

class Operation(Protocol):
    year: int | None
    doy: int | None

@dataclass(kw_only=True)
class Planting(Operation):
    year: int | None = None
    doy: int | None = None
    end_doy: int = -999
    max_smc: float = -999
    min_smc: float = -999
    max_soil_temp: float = -999
    min_soil_temp: float = -999
    crop: str | None = None
    use_auto_irr: int = 0
    use_auto_fert: int = 0
    density: float = 1.0
    clipping_start: int = 1
    clipping_end: int = 366

@dataclass(kw_only=True)
class Tillage(Operation):
    year: int | None = None
    doy: int | None = None
    tool: str
    crop_name: str = 'N/A'
    frac_thermal_time: float = 0.0
    kill_efficiency: float = 0.0

@dataclass(kw_only=True)
class Harvest(Operation):
    year: int | None = None
    doy: int | None = None
    tool: str | None = None
    crop_name: str = 'N/A'
    frac_thermal_time: float = 0.0
    kill_efficiency: float = 0.0

@dataclass(kw_only=True)
class Kill(Operation):
    year: int | None = None
    doy: int | None = None
    tool: str = 'Kill_Crop'
    crop_name: str = 'N/A'
    frac_thermal_time: float = 0.0
    kill_efficiency: float = 0.0

@dataclass(kw_only=True)
class FixedFertilization(Operation):
    year: int | None = None
    doy: int | None = None
    source: str | None = None
    mass: float | None = None
    form: str = 'Liquid'
    method: str = 'Broadcast'
    depth: float = 0.0

@dataclass(kw_only=True)
class FixedIrrigation(Operation):
    year: int | None = None
    doy: int | None = None
    volume: float | None = None

@dataclass(kw_only=True)
class AutoIrrigation:
    crop: str | None = None
    start_day: int = 1
    end_day: int = 366
    water_depletion: float | None = None
    depth: float | None = None

OPERATION_PARAMETERS = {
    'planting': Planting,
    'tillage': Tillage,
    'fixed_fertilization': FixedFertilization,
    'fixed_irrigation': FixedIrrigation,
    'auto_irrigation': AutoIrrigation,
}

def read_operation_file(operation: str | Path) -> list:
    with open(Path(operation)) as f:
        lines = f.read().splitlines()

    lines = iter([line for line in lines if (not line.strip().startswith('#')) and line.strip()])

    operations = []
    while True:
        try:
            operation = next(lines).lower()
            if operation not in OPERATION_PARAMETERS:
                raise ValueError(f"Unknown operation keyword found: {operation}")
            target_class = OPERATION_PARAMETERS[operation]
            hints = get_type_hints(target_class)
            operation_dict: dict[str, Any] = {field.name: parse_value(next(lines), field.name, hints[field.name]) for field in fields(target_class)}
            if operation == 'tillage':
                if operation_dict['tool'].lower().replace('_', '') in ['grainharvest', 'harvestgrain']:
                    operation_dict['tool'] = 'grain_harvest'
                    if operation_dict['crop_name'].lower() in ['n/a', 'na', 'all']:
                        operation_dict['crop_name'] = 'All'
                    target_class = Harvest
                elif operation_dict['tool'].lower().replace('_', '') in ['forageharvest', 'harvestforage']:
                    operation_dict['tool'] = 'forage_harvest'
                    if operation_dict['crop_name'].lower() in ['n/a', 'na', 'all']:
                        operation_dict['crop_name'] = 'All'
                    target_class = Harvest
                elif operation_dict['tool'].lower().replace('_', '') in ['kill', 'killcrop', 'killcrops']:
                    operation_dict['tool'] = 'kill'
                    if operation_dict['crop_name'].lower() in ['n/a', 'na', 'all']:
                        operation_dict['crop_name'] = 'All'
                    target_class = Kill

            operations.append(target_class(**operation_dict))
        except StopIteration:
            break

    return operations
