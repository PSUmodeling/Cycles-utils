from .operation_file import Planting, Tillage, FixedFertilization
from .operation_file import generate_operation_file
from .crop_file import Crop, generate_crop_file
from .soil_file import SoilLayer
from .soil_file import generate_soil_file
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import get_type_hints, Any
from .soil_file import DEFAULT_PROFILE
from ._base_file import unwrap_optional, format_field, FMT_1F, FMT_2F, FMT_3F

SOC_FRACTION = 0.58
OPERATION_TYPES = ('planting', 'tillage', 'fixed_fertilization', 'fixed_irrigation', 'auto_irrigation')

@dataclass
class TillageTool:
    tool: str = field(compare=False)    # label only; not part of identity
    depth: float = field(metadata={'description': '(m)', 'fmt': FMT_2F})
    soil_disturb_ratio: float = field(metadata={'description': '(-)', 'fmt': FMT_1F})
    mixing_efficiency: float = field(metadata={'description': '(-)', 'fmt': FMT_3F})

@dataclass
class Fertilizer:
    fertilizer: str = field(compare=False)  # label only; not part of identity
    c_organic: float = field(metadata={'description': '(kg/kg)', 'fmt': FMT_3F})
    c_charcoal: float = field(compare=False, metadata={'description': '(kg/kg)', 'fmt': FMT_3F})
    n_organic: float = field(metadata={'description': '(kg/kg)', 'fmt': FMT_3F})
    n_charcoal: float = field(compare=False, metadata={'description': '(kg/kg)', 'fmt': FMT_3F})
    n_nh4: float = field(metadata={'description': '(kg/kg)', 'fmt': FMT_3F})
    n_no3: float = field(metadata={'description': '(kg/kg)', 'fmt': FMT_3F})
    p_organic: float = field(compare=False, metadata={'description': '(kg/kg)', 'fmt': FMT_3F})
    p_charcoal: float = field(compare=False, metadata={'description': '(kg/kg)', 'fmt': FMT_3F})
    p_inorganic: float = field(compare=False, metadata={'description': '(kg/kg)', 'fmt': FMT_3F})
    k: float = field(compare=False, metadata={'description': '(kg/kg)', 'fmt': FMT_3F})
    s: float = field(compare=False, metadata={'description': '(kg/kg)', 'fmt': FMT_3F})


def _create_object_from_dict(target_class: type, data_dict: dict[str, Any], *, override_defaults: dict={}) -> Any:
    hints = get_type_hints(target_class)

    keys = {
        f.name: f.name if f.name not in override_defaults else override_defaults[f.name]
        for f in fields(target_class) if f.metadata.get('readable', True)
    }

    return target_class(
        **{
            f.name: unwrap_optional(hints[f.name])(data_dict.get(keys[f.name], f.default))
            for f in fields(target_class)
            if f.name in keys and keys[f.name] in data_dict
        }
    )


def _add_resource(operation_dict: dict[str, Any], resources: list, resource_class: type, name_field: str, dict_key: str, override_defaults: dict | None=None) -> None:
    resource = _create_object_from_dict(resource_class, operation_dict, override_defaults=override_defaults or {})
    name = getattr(resource, name_field)
    existing_names = [getattr(r, name_field).lower() for r in resources]

    # Exact match — reuse existing resource's name
    for existing in resources:
        if resource == existing:
            operation_dict[dict_key] = getattr(existing, name_field)
            return

    # Same name, different properties — find a unique suffixed name
    if name.lower() in existing_names:
        k = 1
        while True:
            candidate = f'{name}_{k}'
            if candidate.lower() not in existing_names:
                setattr(resource, name_field, candidate)
                operation_dict[dict_key] = candidate
                break
            k += 1
    else:
        operation_dict[dict_key] = name

    resources.append(resource)


def _add_tillage_tool(operation_dict: dict[str, Any], tillage_tools: list[TillageTool]) -> None:
    _add_resource(operation_dict, tillage_tools, TillageTool, name_field='tool', dict_key='tool')


def _add_fertilizer(operation_dict: dict[str, Any], fertilizers: list[Fertilizer]) -> None:
    _add_resource(operation_dict, fertilizers, Fertilizer, name_field='fertilizer', dict_key='source', override_defaults={'fertilizer': 'source'})


def _read_obsolete_crop_file(file_path: str | Path) -> dict[str, dict[str, str]]:
    with open(Path(file_path)) as f:
        lines = f.read().splitlines()
    lines = iter([line for line in lines if (not line.strip().startswith('#')) and line.strip()])

    crops = {}
    crop_dict = {}
    while True:
        try:
            line = next(lines).strip()
            field_name = line.split()[0].lower()
            if field_name == 'name':
                if crop_dict:
                    crops.update({crop_name: crop_dict})

                # start of a new crop entry
                crop_name = line.split()[1]
                crop_dict = {}
            else:
                crop_dict[field_name] = line.split()[1]
        except StopIteration:
            if crop_dict:
                crops.update({crop_name: crop_dict})
            break

    return crops


def convert_obsolete_crop_files(crop_fns: dict[str | Path, str | Path], *, planted_crops: set[str] | None=None) -> None:
    for obsolete_fn, new_fn in crop_fns.items():
        print(f'{obsolete_fn} -> {new_fn}')
        obsolete_crops: dict[str, dict[str, str]] = _read_obsolete_crop_file(obsolete_fn)
        selected_crops = planted_crops if planted_crops is not None else obsolete_crops.keys()

        crops = {
            crop_name: Crop(
                **{
                    f.name: _create_object_from_dict(get_type_hints(Crop)[f.name], obsolete_crop)
                    for f in fields(Crop)
                }
            )
            for crop_name, obsolete_crop in obsolete_crops.items() if crop_name in selected_crops
        }

    generate_crop_file(Path(new_fn), crops)


def _read_obsolete_operation_file(file_path: str | Path, tillage_tools: list, fertilizers: list) -> list[dict]:
    with open(Path(file_path)) as f:
        lines = f.read().splitlines()
    lines = iter([line for line in lines if not line.strip().startswith('#') and line.strip()])

    operations = []
    operation_dict = {}
    while True:
        try:
            line = next(lines).strip()
        except StopIteration:
            operations.append({operation: operation_dict})
            break

        field_name = line.split()[0].lower()
        if field_name in OPERATION_TYPES:
            if operation_dict:
                if operation == 'tillage':
                    _add_tillage_tool(operation_dict, tillage_tools)
                elif operation == 'fixed_fertilization':
                    _add_fertilizer(operation_dict, fertilizers)
                operations.append({operation: operation_dict})

            operation = field_name
            operation_dict = {}
        else:
            operation_dict[field_name] = line.split()[1]

    return operations


def _convert_obsolete_operations(obsolete_operations: list[dict], crops: dict[str, dict], planted_crops: list[str]) -> list:
    new_operations = []
    for obsolete_op in obsolete_operations:
        operation_type = list(obsolete_op)[0]
        operation = obsolete_op[operation_type]
        if operation_type == 'planting':
            crop = crops[operation['crop']]
            planted_crops.append(operation['crop'])
            new_operations.append(_create_object_from_dict(Planting, operation | crop))
        elif operation_type == 'tillage':
            new_operations.append(_create_object_from_dict(Tillage, operation))
        elif operation_type == 'fixed_fertilization':
            new_operations.append(_create_object_from_dict(FixedFertilization, operation))
        else:
            raise ValueError(f"Unknown operation type: {operation}")
    return new_operations


def _write_resource_file(file_path: Path, resources: list, resource_class: type) -> None:
    contents = [
        '\n'.join(format_field(r, f, (20, 12)) for f in fields(resource_class))
        for r in resources
    ]

    Path(file_path).write_text('\n\n'.join(contents) + '\n')


def convert_obsolete_operation_files(operation_fns: dict[str, str], crop_fn: str | Path, input_dir: str | Path) -> set[str]:
    crop_dict = _read_obsolete_crop_file(crop_fn)
    planted_crops = []
    tillage_tools: list[TillageTool] = []
    fertilizers: list[Fertilizer] = []

    for obsolete_fn, new_fn in operation_fns.items():
        print(f'{obsolete_fn} -> {new_fn}')
        obsolete_operations = _read_obsolete_operation_file(obsolete_fn, tillage_tools, fertilizers)

        new_operations = _convert_obsolete_operations(obsolete_operations, crop_dict, planted_crops)
        generate_operation_file(Path(new_fn), new_operations)

    print(f'Planted crops: {set(planted_crops)}')
    _write_resource_file(Path(input_dir) / 'tillage_tools.txt', tillage_tools, TillageTool)
    _write_resource_file(Path(input_dir) /'fertilizers.txt', fertilizers, Fertilizer)

    return set(planted_crops)


def _read_obsolete_soil_file(file_path: str | Path) -> tuple[list, int, float, str]:
    with open(Path(file_path)) as f:
        lines = f.read().splitlines()

    data_lnos = [i for i, line in enumerate(lines) if line.strip() and not line.strip().startswith('#')]
    comment_lnos = [i for i, line in enumerate(lines) if line.strip().startswith('#') and (i < data_lnos[0] or i > data_lnos[-1])]

    data_lines = [lines[i].strip() for i in data_lnos]
    comment_lines = [f'# {lines[i][1:].strip()}' for i in comment_lnos]

    for i, line in enumerate(data_lines):
        if line.split()[0].lower() == 'curve_number':
            curve_number = int(line.split()[1])
        elif line.split()[0].lower() == 'slope':
            slope = float(line.split()[1])
        elif line.split()[0].lower() == 'layer':
            header_row = i

    obsolete_soil_layers = [
        {data_lines[header_row].split()[i].lower(): float(line.split()[i]) for i in range(1, len(line.split()))}
        for line in data_lines[header_row + 1:]
    ]

    return obsolete_soil_layers, curve_number, slope, '\n'.join(comment_lines)


def convert_obsolete_soil_files(soil_fns: dict, *, keep_profile: bool=True) -> None:
    for obsolete_fn, new_fn in soil_fns.items():
        print(f'{obsolete_fn} -> {new_fn}')
        _convert_obsolete_soil_file(obsolete_fn, new_fn, keep_profile)


def _convert_obsolete_soil_file(obsolete_fn: str | Path, new_fn: str | Path, keep_profile: bool) -> None:
    f = Path(obsolete_fn)
    obsolete_soil_layers, curve_number, slope, comments = _read_obsolete_soil_file(f)

    cumulative_depth = 0.0
    for layer in obsolete_soil_layers:
        layer['top'] = cumulative_depth
        layer['bottom'] = layer['top'] + layer['thick']
        cumulative_depth = layer['bottom']
        layer['soc'] = layer['organic'] * SOC_FRACTION if 'soc' not in layer else layer['soc']

    soil_layers = [
        _create_object_from_dict(SoilLayer, layer, override_defaults={'bulk_density': 'bd', 'coarse_fragments': 'rock', 'pH': 'ph'})
        for layer in obsolete_soil_layers
    ]

    layers = soil_layers if keep_profile else DEFAULT_PROFILE
    generate_soil_file(Path(new_fn), soil_layers, layers=layers, curve_number=curve_number, slope=slope, desc=comments)
