from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import get_type_hints, Protocol, Any
from ._base_file import parse_value

class Operation(Protocol):
    year: int | None
    doy: int | None
    relative_doy: bool

@dataclass(kw_only=True)
class Planting(Operation):
    year: int | None = None
    doy: int
    end_doy: int = -999
    crop: str
    max_smc: float = -999
    min_smc: float = -999
    max_soil_temp: float = -999
    min_soil_temp: float = -999
    use_auto_irr: int = 0
    use_auto_fert: int = 0
    density: float = 1.0
    clipping_start: int = 1
    clipping_end: int = 366
    maximum_soil_coverage: float = 100.0
    standing_residue_at_harvest: float = 50.0
    residue_removed: float = 0.0
    clipping_biomass_threshold_lower: float = 0.5
    clipping_biomass_threshold_upper: float = 999.0
    clipping_biomass_destiny: str = 'REMOVE'
    harvest_timing: float = -999
    kill_after_harvest: int = 1
    relative_doy: bool = field(default=False, metadata={'readable': False})


@dataclass(kw_only=True)
class Tillage(Operation):
    year: int | None = None
    doy: int
    tool: str
    crop_name: str = 'N/A'
    frac_thermal_time: float = 0.0
    kill_efficiency: float = 0.0
    relative_doy: bool = field(default=False, metadata={'readable': False})

@dataclass(kw_only=True)
class Harvest(Operation):
    year: int | None = None
    doy: int
    tool: str = 'Grain_Harvest'
    crop_name: str = 'N/A'
    frac_thermal_time: float = 0.0
    kill_efficiency: float = 0.0
    relative_doy: bool = field(default=False, metadata={'readable': False})

@dataclass(kw_only=True)
class Kill(Operation):
    year: int | None = None
    doy: int
    tool: str = 'Kill_Crop'
    crop_name: str = 'N/A'
    frac_thermal_time: float = 0.0
    kill_efficiency: float = 0.0
    relative_doy: bool = field(default=False, metadata={'readable': False})

@dataclass(kw_only=True)
class FixedFertilization(Operation):
    year: int | None = None
    doy: int
    source: str
    mass: float  = 0.0
    form: str = 'Liquid'
    method: str = 'Broadcast'
    depth: float = 0.0
    relative_doy: bool = field(default=False, metadata={'readable': False})

@dataclass(kw_only=True)
class FixedIrrigation(Operation):
    year: int | None = None
    doy: int
    volume: float = 0.0
    relative_doy: bool = field(default=False, metadata={'readable': False})

@dataclass(kw_only=True)
class AutoIrrigation:
    crop: str
    start_day: int = 1
    end_day: int = 366
    water_depletion: float = 0.5
    depth: float = 0.0

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

    lines = iter([line for line in lines if not line.strip().startswith('#') and line.strip()])
    operations = []
    while True:
        try:
            operation = next(lines).lower()
            if operation not in OPERATION_PARAMETERS:
                raise ValueError(f"Unknown operation keyword found: {operation}")

            target_class = OPERATION_PARAMETERS[operation]
            hints = get_type_hints(target_class)

            operation_dict: dict[str, Any] = {}
            for f in fields(target_class):
                if not f.metadata.get('readable', True):
                    continue
                raw = next(lines)
                if f.name == 'doy':
                    operation_dict[f.name] = raw    # keep as raw string for + detection
                else:
                    operation_dict[f.name] = parse_value(raw, f.name, hints[f.name])

            # Detect relative DOY before constructing the instance
            raw_doy = operation_dict.get('doy', '')
            if isinstance(raw_doy, str) and raw_doy.split()[1].strip().startswith('+'):
                operation_dict['doy'] = int(raw_doy.split()[1].strip()[1:])
                operation_dict['relative_doy'] = True
            else:
                operation_dict['doy'] = parse_value(raw_doy, 'doy', hints['doy'])
                operation_dict['relative_doy'] = False

            # Reclassify tillage operations
            if operation == 'tillage':
                tool = operation_dict['tool'].lower().replace('_', '')
                if tool in ('grainharvest', 'harvestgrain'):
                    operation_dict['tool'] = 'grain_harvest'
                    if operation_dict['crop_name'].lower() in ('n/a', 'na', 'all'):
                        operation_dict['crop_name'] = 'All'
                    target_class = Harvest
                elif tool in ('forageharvest', 'harvestforage'):
                    operation_dict['tool'] = 'forage_harvest'
                    if operation_dict['crop_name'].lower() in ('n/a', 'na', 'all'):
                        operation_dict['crop_name'] = 'All'
                    target_class = Harvest
                elif tool in ('kill', 'killcrop', 'killcrops'):
                    operation_dict['tool'] = 'kill'
                    if operation_dict['crop_name'].lower() in ('n/a', 'na', 'all'):
                        operation_dict['crop_name'] = 'All'
                    target_class = Kill

            operations.append(target_class(**operation_dict))

        except StopIteration:
            break

    return operations
