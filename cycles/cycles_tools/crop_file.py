from __future__ import annotations
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_type_hints, Protocol
from ._base_file import _write_file, _resolve_dict_values, _extract, _parse_value, _unwrap_optional, _format_block
from ._base_file import FMT_1F, FMT_2F, FMT_3F, FMT_4F

@dataclass(kw_only=True)
class Phenology:
    thermal_time_to_emergence: float = field(metadata={'description': '(degree C day)', 'fmt': FMT_1F})
    flowering_tt: float = field(metadata={'description': '(degree C day)', 'fmt': FMT_1F})
    maturity_tt: float = field(metadata={'description': '(degree C day)', 'fmt': FMT_1F})
    base_temperature_for_development: float = field(metadata={'description': '(degree C)', 'fmt': FMT_1F})
    optimum_temperature_for_development: float = field(metadata={'description': '(degree C)', 'fmt': FMT_1F})
    max_temperature_for_development: float = field(metadata={'description': '(degree C)', 'fmt': FMT_1F})
    photoperiod_min: float = field(default=-999, metadata={'description': '(hours)', 'fmt': FMT_1F})
    photoperiod_max: float = field(default=-999, metadata={'description': '(hours)', 'fmt': FMT_1F})
    thermal_time_vernalization: float = field(default=-999, metadata={'description': '(degree C day)', 'fmt': FMT_1F})
    thermal_time_flowering_at_pmax: float = field(default=-999, metadata={'description': '(degree C day)', 'fmt': FMT_1F})
    thermal_time_grain_filling: float = field(default=-999, metadata={'description': '(degree C day)', 'fmt': FMT_1F})
    vernalization_temperature_threshold_lower: float = field(default=-999, metadata={'description': '(degree C)', 'fmt': FMT_1F})
    vernalization_temperature_threshold_optimal: float = field(default=-999, metadata={'description': '(degree C)', 'fmt': FMT_1F})
    vernalization_temperature_threshold_upper: float = field(default=-999, metadata={'description': '(degree C)', 'fmt': FMT_1F})
    temperature_post_flowering_lower: float = field(default=-999, metadata={'description': '(degree C)', 'fmt': FMT_1F})
    temperature_post_flowering_optimal: float = field(default=-999, metadata={'description': '(degree C)', 'fmt': FMT_1F})
    temperature_post_flowering_upper: float = field(default=-999, metadata={'description': '(degree C)', 'fmt': FMT_1F})
    devernalization_temperature: float = field(default=-999, metadata={'description': '(degree C)', 'fmt': FMT_1F})

@dataclass(kw_only=True)
class Allocation:
    initial_partitioning_to_shoot: float = field(metadata={'description': '(100%)', 'fmt': FMT_2F})
    final_partitioning_to_shoot: float = field(metadata={'description': '(100%)', 'fmt': FMT_2F})
    maximum_harvest_index: float = field(metadata={'description': '(-)', 'fmt': FMT_3F})
    minimum_harvest_index: float = field(metadata={'description': '(-)', 'fmt': FMT_4F})
    harvest_index_slope_multiplier: float = field(metadata={'description': '(-)', 'fmt': FMT_1F})
    n_max_concentration_grain: float | None = field(default=-999, metadata={'description': '(%)', 'fmt': FMT_2F})
    n_min_concentration_grain: float | None = field(default=-999, metadata={'description': '(%)', 'fmt': FMT_2F})
    n_max_concentration_straw: float | None = field(default=-999, metadata={'description': '(%)', 'fmt': FMT_2F})
    n_min_concentration_straw: float | None = field(default=-999, metadata={'description': '(%)', 'fmt': FMT_2F})
    n_partitioning_factor: float | None = field(default=-999, metadata={'description': '(-)', 'fmt': FMT_2F})

@dataclass(kw_only=True)
class Root:
    maximum_rooting_depth: float = field(metadata={'description': '(m)', 'fmt': FMT_2F})

@dataclass(kw_only=True)
class Growth:
    radiation_use_efficiency: float = field(metadata={'description': '(g/MJ solar radiation)', 'fmt': FMT_1F})
    transpiration_use_efficiency: float = field(metadata={'description': '(g/kg)', 'fmt': FMT_1F})

@dataclass(kw_only=True)
class Transpiration:
    min_temperature_for_transpiration: float = field(metadata={'description': '(degree C)', 'fmt': FMT_1F})
    threshold_temperature_for_transpiration: float = field(metadata={'description': '(degree C)', 'fmt': FMT_1F})
    kc: float = field(metadata={'description': '(-)', 'fmt': FMT_2F})
    lwp_stress_onset: float = field(metadata={'description': '(J/kg)', 'fmt': FMT_1F})
    lwp_wilting_point: float = field(metadata={'description': '(J/kg)', 'fmt': FMT_1F})
    transpiration_max: float = field(metadata={'description': '(mm/day)', 'fmt': FMT_1F})

@dataclass(kw_only=True)
class Stress:
    min_temperature_for_cold_damage: float = field(metadata={'description': '(degree C)', 'fmt': FMT_1F})
    threshold_temperature_for_cold_damage: float = field(metadata={'description': '(degree C)', 'fmt': FMT_1F})

@dataclass(kw_only=True)
class Nutrition:
    n_max_concentration: float = field(metadata={'description': '(g/g)', 'fmt': FMT_3F})
    n_dilution_slope: float = field(metadata={'description': '(-)', 'fmt': FMT_2F})

@dataclass(kw_only=True)
class Classification:
    annual: int = field(metadata={'description': '(-)'})
    legume: int = field(metadata={'description': '(-)'})
    c3: int = field(metadata={'description': '(-)'})

@dataclass
class Crop:
    phenology: Phenology
    allocation: Allocation
    root: Root
    growth: Growth
    transpiration: Transpiration
    stress: Stress
    nutrition: Nutrition
    classification: Classification
    def __eq__(self, other):
        if not isinstance(other, Crop):
            return NotImplemented
        for f in fields(Crop):
            for sub_field in fields(getattr(self, f.name)):
                if getattr(getattr(self, f.name), sub_field.name) != getattr(getattr(other, f.name), sub_field.name):
                    return False
        return True


def _read_individual_crop(lines: iter[str], hints: dict[str, type]) -> dict[str, Crop] | None:  # type: ignore
    crop_dict = {}
    try:
        name = str(_parse_value(next(lines), 'name', str))
        for f in fields(Crop):
            target_class = _unwrap_optional(hints[f.name])
            sub_hints = get_type_hints(target_class)
            crop_dict[f.name] = target_class(**{sub_field.name: _parse_value(next(lines), sub_field.name, sub_hints[sub_field.name]) for sub_field in fields(target_class)})
        return {name: Crop(**crop_dict)}
    except StopIteration:
        return None


def read_crop_file(file_path: str | Path) -> dict[str, Crop]:
    """Read a Cycles crop file into named `Crop` dataclass instances.

    Args:
        file_path: Path to a Cycles crop file.

    Returns:
        A dictionary mapping crop names to their structured crop definitions.
    """
    with open(Path(file_path)) as f:
        lines = f.read().splitlines()
    lines = iter([line for line in lines if (not line.strip().startswith('#')) and line.strip()])

    hints = get_type_hints(Crop)

    crops = {}
    while True:
        crop_dict = _read_individual_crop(lines, hints)
        if crop_dict is not None:
            crops.update(crop_dict)
        else:
            break

    return crops


def generate_crop_file(file_path: str | Path, crops: dict[str, Crop]) -> None:
    """Write a Cycles crop file from user-provided values.

    Args:
        file_path: Destination crop file path.
        crops: Dictionary of crop names and their corresponding `Crop` dataclass instances.
    """
    file_path = Path(file_path)

    contents = []
    for crop_name, crop in crops.items():
        contents.append(
            '\n'.join([
                '#' * 44,
                f'# {"CROP DESCRIPTION":<41s}#',
                '#' * 44,
                f'{"NAME":<44s}{crop_name}',
            ])
        )
        contents.append(''.join(
            _format_block(f.name, getattr(crop, f.name), widths=(44, 12), center=True, all_caps=False)
            for f in fields(Crop)
        ))

    file_path.write_text('\n'.join(contents))
