from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, fields
from typing import get_type_hints
from ._base_file import parse_value

@dataclass
class Planting:
    year: int
    doy: int
    end_doy: int
    max_smc: float
    min_smc: float
    max_soil_temp: float
    min_soil_temp: float
    crop: str
    use_auto_irr: int
    use_auto_fert: int
    density: float
    clipping_start: int
    clipping_end: int

@dataclass
class Tillage:
    year: int
    doy: int
    tool: str
    depth: float
    soil_disturb_ratio: float
    mixing_efficiency: float
    crop_name: str
    frac_thermal_time: float
    kill_efficiency: float

@dataclass
class Harvest:
    year: int
    doy: int
    tool: str
    depth: float
    soil_disturb_ratio: float
    mixing_efficiency: float
    crop_name: str
    frac_thermal_time: float
    kill_efficiency: float

@dataclass
class Kill:
    year: int
    doy: int
    tool: str
    depth: float
    soil_disturb_ratio: float
    mixing_efficiency: float
    crop_name: str
    frac_thermal_time: float
    kill_efficiency: float

@dataclass
class FixedFertilization:
    year: int
    doy: int
    source: str
    mass: float
    form: str
    method: str
    depth: float
    c_organic: float
    c_charcoal: float
    n_organic: float
    n_charcoal: float
    n_nh4: float
    n_no3: float
    p_organic: float
    p_charcoal: float
    p_inorganic: float
    k: float
    s: float

@dataclass
class FixedIrrigation:
    year: int
    doy: int
    volume: float

@dataclass
class AutoIrrigation:
    crop: str
    start_day: int
    end_day: int
    water_depletion: float
    last_soil_layer: int

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
            operation_dict = {field.name: parse_value(next(lines), field.name, hints[field.name]) for field in fields(target_class)}
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
