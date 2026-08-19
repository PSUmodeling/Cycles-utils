from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import get_type_hints, Protocol, Any
from ._base_file import parse_value

class Operation(Protocol):
    year: int | None
    doy: int
    relative_doy: bool
    resolved_doy: int | None

@dataclass(kw_only=True)
class Planting(Operation):
    year: int | None = None
    doy: int
    end_doy: int = -999
    crop: str = ''
    max_smc: float = field(default=-999, metadata={'description': 'fraction of plant available water between permanent wilting point and field capacity (-)'})
    min_smc: float = field(default=-999, metadata={'description': 'fraction of plant available water between permanent wilting point and field capacity (-)'})
    max_soil_temp: float = field(default=-999, metadata={'description': '(degree C)'})
    min_soil_temp: float = field(default=-999, metadata={'description': '(degree C)'})
    use_auto_irr: int = 0
    use_auto_fert: int = 0
    density: float = field(default=1.0, metadata={'description': '(-)'})
    clipping_start: int = 1
    clipping_end: int = 366
    maximum_soil_coverage: float = field(default=100.0, metadata={'description': '(%)'})
    standing_residue_at_harvest: float = field(default=50.0, metadata={'description': '(%)'})
    residue_removed: float = field(default=0.0, metadata={'description': '(%)'})
    clipping_biomass_threshold_lower: float = field(default=0.5, metadata={'description': '(Mg/ha)'})
    clipping_biomass_threshold_upper: float = field(default=999, metadata={'description': '(Mg/ha)'})
    clipping_biomass_destiny: str = 'REMOVE'
    harvest_timing: float = field(default=-999, metadata={'description': '(%)'})
    kill_after_harvest: int = 1
    relative_doy: bool = field(default=False, metadata={'readable': False})
    resolved_doy: int | None = field(default=None, metadata={'readable': False})

@dataclass(kw_only=True)
class Tillage(Operation):
    year: int | None = None
    doy: int
    tool: str
    crop_name: str = 'N/A'
    frac_thermal_time: float = 0.0
    kill_efficiency: float = 0.0
    relative_doy: bool = field(default=False, metadata={'readable': False})
    resolved_doy: int | None = field(default=None, metadata={'readable': False})

@dataclass(kw_only=True)
class Harvest(Operation):
    year: int | None = None
    doy: int
    tool: str = 'GrainHarvest'
    crop_name: str = 'N/A'
    frac_thermal_time: float = 0.0
    kill_efficiency: float = 0.0
    relative_doy: bool = field(default=False, metadata={'readable': False})
    resolved_doy: int | None = field(default=None, metadata={'readable': False})

@dataclass(kw_only=True)
class Kill(Operation):
    year: int | None = None
    doy: int
    tool: str = 'KillCrop'
    crop_name: str = 'N/A'
    frac_thermal_time: float = 0.0
    kill_efficiency: float = 0.0
    relative_doy: bool = field(default=False, metadata={'readable': False})
    resolved_doy: int | None = field(default=None, metadata={'readable': False})

@dataclass(kw_only=True)
class FixedFertilization(Operation):
    year: int | None = None
    doy: int
    source: str
    mass: float = field(default=0.0, metadata={'description': '(kg/ha)'})
    form: str = 'Liquid'
    method: str = 'Broadcast'
    depth: float = field(default=0.0, metadata={'description': '(m)'})
    relative_doy: bool = field(default=False, metadata={'readable': False})
    resolved_doy: int | None = field(default=None, metadata={'readable': False})

@dataclass(kw_only=True)
class FixedIrrigation(Operation):
    year: int | None = None
    doy: int
    volume: float = 0.0
    relative_doy: bool = field(default=False, metadata={'readable': False})
    resolved_doy: int | None = field(default=None, metadata={'readable': False})

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


def _resolve_reference_doy(operations: list[Operation], relative_doy: int) -> int:
    planting = next((op for op in reversed(operations) if isinstance(op, Planting)), None)
    if planting is None:
        raise ValueError("Relative DOY specified but no planting operation found.")
    return planting.doy + relative_doy


def read_operation_file(file_path: str | Path) -> list[Operation]:
    """Parse a Cycles operation file into operation objects.

    Args:
        file_path: Path to a Cycles operation file.

    Returns:
        A list of operation dataclass instances.
    """
    with open(Path(file_path)) as f:
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
                operation_dict['resolved_doy'] = _resolve_reference_doy(operations, operation_dict['doy'])
            else:
                operation_dict['doy'] = parse_value(raw_doy, 'doy', hints['doy'])
                operation_dict['relative_doy'] = False

            # Reclassify tillage operations
            if operation == 'tillage':
                tool = operation_dict['tool'].lower().replace('_', '')
                if tool in ('grainharvest', 'harvestgrain'):
                    operation_dict['tool'] = 'GrainHarvest'
                    if operation_dict['crop_name'].lower() in ('n/a', 'na', 'all'):
                        operation_dict['crop_name'] = 'All'
                    target_class = Harvest
                elif tool in ('forageharvest', 'harvestforage'):
                    operation_dict['tool'] = 'ForageHarvest'
                    if operation_dict['crop_name'].lower() in ('n/a', 'na', 'all'):
                        operation_dict['crop_name'] = 'All'
                    target_class = Harvest
                elif tool in ('kill', 'killcrop', 'killcrops'):
                    operation_dict['tool'] = 'KillCrop'
                    if operation_dict['crop_name'].lower() in ('n/a', 'na', 'all'):
                        operation_dict['crop_name'] = 'All'
                    target_class = Kill

            operations.append(target_class(**operation_dict))
        except StopIteration:
            break

    return operations


def _camel_to_snake(text: str) -> str:
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text).lower()


def format_operation(operation: Any, doy_override: dict[str, str] | None = None) -> list[str]:
    lines = [_camel_to_snake(type(operation).__name__).upper()]
    for f in fields(operation):
        if not f.metadata.get('readable', True):
            continue
        val = doy_override.get(f.name) if doy_override else None
        description = f.metadata.get('description', '')

        if val is None:
            val = getattr(operation, f.name)
            if f.name == 'doy' and operation.relative_doy:
                val = f'+{val}'
            if f.name == 'mass':
                val = f'{val:.2f}'
            if isinstance(val, float) and val == -999.0:
                val = '-999'
        lines.append(f'{f.name.upper():<36}{val:<12}# {description}' if description else f'{f.name.upper():<36}{val}')
    lines.append('')
    return lines


def generate_operation_file(file_path: Path | str, operations: list[Operation], *, desc: str='') -> None:
    """Write a Cycles operation file from structured operation records.

    The optional description is written as the first line when provided. Each operation is serialized with
    to preserves Cycles field naming and relative day-of-year formatting.

    Args:
        file_path: Destination operation file path.
        operations: Ordered operation records to serialize.
        desc: Optional header or description line to place at the top of the file.
    """
    lines: list[str] = []
    if desc:
        lines.append(desc)
    for op in operations:
        if isinstance(op, (Harvest, Kill)):
            op = Tillage(**op.__dict__)
        lines.extend(format_operation(op))
    Path(file_path).write_text('\n'.join(lines))
