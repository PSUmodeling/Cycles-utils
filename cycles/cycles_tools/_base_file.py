from __future__ import annotations
import fiona
import geopandas as gpd
import pandas as pd
import types
from dataclasses import fields
from pathlib import Path
from typing import Union, Any

def _format_block(label: str, block) -> str:
    lines = [f'## {label.replace("_", " ").upper()} ##']
    for f in fields(block):
        val = getattr(block, f.name)
        if isinstance(val, float) and val == -999.0:
            val = '-999'
        description = f.metadata.get('description', '')
        lines.append(f'{f.name.upper():<28}{val:<8}# {description}' if description else f'{f.name.upper():<28}{val}')
    lines.append('')

    return '\n'.join(lines)


def write_file(fn: Path, config) -> None:
    content = '\n'.join(
        _format_block(f.name, getattr(config, f.name))
        for f in fields(config)
        if getattr(config, f.name) is not None
    )
    fn.write_text(content)


def resolve_dict_values(user_dict: dict, simulation: dict[str, Any] | None) -> dict:
    return {key: func(simulation) if callable(func) else func for key, func in user_dict.items()}


def extract(dc_class, resolved: dict) -> dict:
    return {f.name: resolved[f.name] for f in fields(dc_class) if f.name in resolved}


def parse_value(raw: str, name: str, hint: type) -> int | float | str:
    hint = unwrap_optional(hint)
    if raw.split()[0].lower() == name.lower():
        if hint is int: return int(raw.split()[1])
        if hint is float: return float(raw.split()[1])
        return raw.split()[1]
    else:
        raise ValueError(f"Expected field name '{name}' not found in line: {raw}")


def unwrap_optional(t) -> type:
    origin = getattr(t, '__origin__', None)
    if origin is Union or origin is types.UnionType or isinstance(t, types.UnionType):
        return next(arg for arg in t.__args__ if arg is not type(None))
    return t


def read_geospatial_file(file_path: str | Path) -> gpd.GeoDataFrame:
    file_path = Path(file_path)
    ext = file_path.suffix.lstrip('.').lower()
    match ext:
        case 'shp':
            return gpd.read_file(file_path)
        case 'kml':
            return _read_kml(file_path)
        case _:
            raise ValueError(f"Unsupported boundary format: '.{ext}'")


def _read_kml(file_path: Path) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        pd.concat(
            [gpd.read_file(file_path, driver='KML', layer=layer) for layer in fiona.listlayers(file_path)],
            ignore_index=True,
        )
    )
